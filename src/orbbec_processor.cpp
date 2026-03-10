#include "orbbec_processor.hpp"
#include "logger.hpp"
#include <chrono>
#include <cstring>

// ============================================================================
//  Construction / Destruction
// ============================================================================

OrbbecProcessor::OrbbecProcessor() {
    // ── Create pipeline & obtain device ────────────────────────────────
    pipeline_ = std::make_shared<ob::Pipeline>();
    device_   = pipeline_->getDevice();
    config_   = std::make_shared<ob::Config>();

    // ── Enable Depth stream (640x576 @ 30fps) ─────────────────────────
    auto depthProfiles = pipeline_->getStreamProfileList(OB_SENSOR_DEPTH);
    auto depthProfile  = depthProfiles->getVideoStreamProfile(
        640, 576, OB_FORMAT_ANY, 30);
    config_->enableStream(depthProfile);
    Log::info("Orbbec", "Depth profile: " + std::to_string(depthProfile->width()) + "x"
             + std::to_string(depthProfile->height()) + " @ " + std::to_string(depthProfile->fps()) + " fps");

    // ── Try to enable Color stream (1920x1080 @ 30fps, MJPG) ──────────
    hasColor_ = false;
    try {
        auto colorProfiles = pipeline_->getStreamProfileList(OB_SENSOR_COLOR);
        auto colorProfile  = colorProfiles->getVideoStreamProfile(
            1920, 1080, OB_FORMAT_MJPG, 30); // @code-review: OB_FORMAT_MJPG might can be optimized to other formats which can be displayed and stored immediately without decoding, if supported by the camera
        config_->enableStream(colorProfile);

        Log::info("Orbbec", "Color profile: " + std::to_string(colorProfile->width()) + "x"
                 + std::to_string(colorProfile->height()) + " @ " + std::to_string(colorProfile->fps())
                 + " fps  format=" + std::to_string(static_cast<int>(colorProfile->format())));

        // Log color intrinsics / distortion for reference
        if (colorProfile) {
            auto intr = colorProfile->getIntrinsic();
            auto dist = colorProfile->getDistortion();
            colorIntrinsics_.fx     = intr.fx;
            colorIntrinsics_.fy     = intr.fy;
            colorIntrinsics_.cx     = intr.cx;
            colorIntrinsics_.cy     = intr.cy;
            colorIntrinsics_.width  = colorProfile->width();
            colorIntrinsics_.height = colorProfile->height();
            std::ostringstream iss;
            iss << "Intrinsics fx=" << intr.fx << " fy=" << intr.fy
                << " cx=" << intr.cx << " cy=" << intr.cy
                << "  Distortion k1=" << dist.k1 << " k2=" << dist.k2
                << " k3=" << dist.k3 << " p1=" << dist.p1 << " p2=" << dist.p2;
            Log::info("Orbbec", iss.str());
        }
        hasColor_ = true;
    }
    catch (...) {
        Log::warn("Orbbec", "Color sensor not available.");
        hasColor_ = false;
    }

    // ── Multi-device hardware sync (SECONDARY) ────────────────────────
    // Mirrors the Python config:
    //   sync_config.mode                    = OBMultiDeviceSyncMode.SECONDARY
    //   sync_config.color_delay_us          = 0
    //   sync_config.depth_delay_us          = 0
    //   sync_config.trigger_to_image_delay  = 0
    //   sync_config.trigger_out_enable      = True
    //   sync_config.trigger_out_delay_us    = 0
    //   sync_config.frames_per_trigger      = 1
    try {
        OBMultiDeviceSyncConfig syncCfg = device_->getMultiDeviceSyncConfig();
        syncCfg.syncMode             = OB_MULTI_DEVICE_SYNC_MODE_SECONDARY;
        syncCfg.colorDelayUs         = 0;
        syncCfg.depthDelayUs         = 0;
        syncCfg.trigger2ImageDelayUs = 0;
        syncCfg.triggerOutEnable     = true;
        syncCfg.triggerOutDelayUs    = 0;
        syncCfg.framesPerTrigger     = 1;
        device_->setMultiDeviceSyncConfig(syncCfg);
        Log::info("Orbbec", "Multi-device sync set to SECONDARY.");
    }
    catch (const std::exception &e) {
        Log::error("Orbbec", std::string("Failed to set multi-device sync: ") + e.what());
    }

    // ── Start pipeline (no frame-sync so that hardware sync dominates) ─
    pipeline_->disableFrameSync(); // @code-review: no software sync needed to acquire accurate timestamps
    pipeline_->start(config_);

    // ── Software Align filter (Depth→Color) ────────────────────────
    alignFilter_ = std::make_unique<ob::Align>(OB_STREAM_COLOR);

    Log::info("Orbbec", "Initialised.");
}

OrbbecProcessor::~OrbbecProcessor() {
    stop();
    if (pipeline_) {
        pipeline_->stop();
    }
    Log::info("Orbbec", "Destroyed.");
}

// ============================================================================
//  Thread management
// ============================================================================

void OrbbecProcessor::start() {
    if (running_.load()) return;

    stopRequested_.store(false);
    running_.store(true);
    workerThread_ = std::thread(&OrbbecProcessor::processingLoop, this);
    Log::info("Orbbec", "Worker thread started.");
}

void OrbbecProcessor::stop() {
    if (!running_.load()) return;

    stopRequested_.store(true);
    if (workerThread_.joinable()) {
        workerThread_.join();
    }
    running_.store(false);
    Log::info("Orbbec", "Worker thread stopped.");
}

// ============================================================================
//  Thread-safe frame accessor
// ============================================================================

bool OrbbecProcessor::getLatestFrame(OrbbecFrameData &out) {
    std::lock_guard<std::mutex> lock(frameMutex_);
    if (frameQueue_.empty()) return false;

    out = frameQueue_.back();   // peek at newest
    return true;
}

bool OrbbecProcessor::popFrame(OrbbecFrameData &out) {
    std::lock_guard<std::mutex> lock(frameMutex_);
    if (frameQueue_.empty()) return false;

    out = std::move(frameQueue_.front());
    frameQueue_.pop_front();
    return true;
}

size_t OrbbecProcessor::queueSize() const {
    std::lock_guard<std::mutex> lock(frameMutex_);
    return frameQueue_.size();
}

// ============================================================================
//  Processing loop (runs on worker thread)
// ============================================================================

// @code-review: core producer thread
void OrbbecProcessor::processingLoop() {
    // ── FPS measurement ─────────────────────────────────────────────────
    uint64_t fpsFrameCount = 0;
    auto     fpsStartTime  = std::chrono::steady_clock::now();
    constexpr double FPS_LOG_INTERVAL_SEC = 5.0;

    while (!stopRequested_.load()) {
        try {
            // Wait up to 200 ms for a frameset (> 1 frame period at 30fps)
            auto frameSet = pipeline_->waitForFrames(200);
            if (!frameSet) continue;

            // ── Pre-check: both depth & color must be present ──────────
            // The Align filter crashes (pFrame is nullptr) if either
            // frame is missing from the frameset.  This happens when
            // the device is in SECONDARY sync mode and no trigger signal
            // has been received yet, or when one stream simply drops.
            auto rawDepth = frameSet->depthFrame();
            auto rawColor = frameSet->colorFrame();

            if (!rawDepth || !rawColor) {
                // Nothing useful to process – silently retry.
                continue;
            }

            // ── Align Depth → Color ────────────────────────────────────
            std::shared_ptr<ob::Frame> alignedFrame;
            try {
                alignedFrame = alignFilter_->process(frameSet);
            }
            catch (const std::exception &e) {
                Log::error("Orbbec", std::string("Align failed: ") + e.what());
                continue;
            }
            if (!alignedFrame) continue;

            auto aligned = alignedFrame->as<ob::FrameSet>();
            if (!aligned) continue;

            auto depthFrame = aligned->depthFrame();
            auto colorFrame = aligned->colorFrame();

            if (!depthFrame || !colorFrame) continue;

            // ── Timestamps ─────────────────────────────────────────────
            uint64_t oColorTs = colorFrame->timeStampUs();
            uint64_t oDepthTs = depthFrame->timeStampUs();

            // First-frame logging (mirrors the Python version)
            if (!firstFrameLogged_) {
                Log::info("Orbbec", "First color frame - device_ts=" + std::to_string(oColorTs)
                         + " us  sys_ts=" + std::to_string(colorFrame->systemTimeStampUs()) + " us");
                Log::info("Orbbec", "First depth frame - device_ts=" + std::to_string(oDepthTs)
                         + " us  sys_ts=" + std::to_string(depthFrame->systemTimeStampUs()) + " us");
                firstFrameLogged_ = true;
            }

            // ── Color → raw bytes ──────────────────────────────────────
            uint32_t cW = colorFrame->width();
            uint32_t cH = colorFrame->height();
            const uint8_t *cData = static_cast<const uint8_t *>(colorFrame->data());
            uint32_t       cSize = colorFrame->dataSize();

            if (!cData || cSize == 0) continue;

            std::vector<uint8_t> colorBuf(cData, cData + cSize);

            // ── Depth → uint16 buffer ──────────────────────────────────
            uint32_t dW = depthFrame->width();
            uint32_t dH = depthFrame->height();
            const uint16_t *dData = static_cast<const uint16_t *>(depthFrame->data());
            uint32_t        dSize = depthFrame->dataSize();

            if (!dData || dSize == 0) continue;

            size_t dPixels = static_cast<size_t>(dW) * dH;
            std::vector<uint16_t> depthBuf(dData, dData + dPixels);

            // ── Publish to queue ───────────────────────────────────
            {
                OrbbecFrameData frame;
                frame.colorData       = std::move(colorBuf);
                frame.colorWidth      = cW;
                frame.colorHeight     = cH;

                frame.depthData       = std::move(depthBuf);
                frame.depthWidth      = dW;
                frame.depthHeight     = dH;

                frame.colorTimestampUs = oColorTs;
                frame.depthTimestampUs = oDepthTs;

                frame.valid = true;

                std::lock_guard<std::mutex> lock(frameMutex_);
                if (frameQueue_.size() >= MAX_QUEUE) {
                    frameQueue_.pop_front();  // drop oldest
                }
                frameQueue_.push_back(std::move(frame));
                newFrameReady_.store(true);
            }

            // ── FPS measurement ────────────────────────────────────────
            fpsFrameCount++;
            auto now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(now - fpsStartTime).count();
            if (elapsed >= FPS_LOG_INTERVAL_SEC) {
                orbFps_ = fpsFrameCount / elapsed;
                fpsFrameCount = 0;
                fpsStartTime  = now;
            }

        }
        catch (const std::exception &e) {
            Log::error("Orbbec", std::string("Processing loop error: ") + e.what());
        }
    }
}
