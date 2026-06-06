#include "orbbec_processor.hpp"
#include "logger.hpp"
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <vector>

namespace {

int64_t steadyNowUs() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

int64_t readMetadataOrDefault(const std::shared_ptr<ob::Frame> &frame,
                              OBFrameMetadataType type) {
    if (!frame) return -1;
    try {
        if (frame->hasMetadata(type)) {
            return frame->getMetadataValue(type);
        }
    }
    catch (...) {
    }
    return -1;
}

std::string metadataToHex(const std::shared_ptr<ob::Frame> &frame) {
    if (!frame) return "";
    try {
        const uint32_t size = frame->getMetadataSize();
        const uint8_t *metadata = frame->getMetadata();
        if (!metadata || size == 0) return "";

        std::ostringstream oss;
        oss << std::hex << std::setfill('0');
        for (uint32_t i = 0; i < size; ++i) {
            oss << std::setw(2) << static_cast<int>(metadata[i]);
        }
        return oss.str();
    }
    catch (...) {
        return "";
    }
}

OrbbecMetadataSnapshot readMetadataSnapshot(const std::shared_ptr<ob::Frame> &frame) {
    OrbbecMetadataSnapshot meta;
    meta.values.fill(-1);
    if (frame) {
        try {
            meta.metadataSize = frame->getMetadataSize();
        }
        catch (...) {
            meta.metadataSize = 0;
        }
        meta.metadataHex = metadataToHex(frame);
        for (uint32_t i = 0; i < static_cast<uint32_t>(OB_FRAME_METADATA_TYPE_COUNT); ++i) {
            const auto type = static_cast<OBFrameMetadataType>(i);
            meta.values[i] = readMetadataOrDefault(frame, type);
        }
    }
    meta.timestamp = readMetadataOrDefault(frame, OB_FRAME_METADATA_TYPE_TIMESTAMP);
    meta.sensorTimestamp = readMetadataOrDefault(frame, OB_FRAME_METADATA_TYPE_SENSOR_TIMESTAMP);
    meta.frameNumber = readMetadataOrDefault(frame, OB_FRAME_METADATA_TYPE_FRAME_NUMBER);
    meta.autoExposure = readMetadataOrDefault(frame, OB_FRAME_METADATA_TYPE_AUTO_EXPOSURE);
    meta.exposure = readMetadataOrDefault(frame, OB_FRAME_METADATA_TYPE_EXPOSURE);
    meta.gain = readMetadataOrDefault(frame, OB_FRAME_METADATA_TYPE_GAIN);
    meta.actualFrameRate = readMetadataOrDefault(frame, OB_FRAME_METADATA_TYPE_ACTUAL_FRAME_RATE);
    meta.frameRate = readMetadataOrDefault(frame, OB_FRAME_METADATA_TYPE_FRAME_RATE);
    meta.gpioInputData = readMetadataOrDefault(frame, OB_FRAME_METADATA_TYPE_GPIO_INPUT_DATA);
    return meta;
}

std::string metadataSummary(const char *name, const OrbbecMetadataSnapshot &meta) {
    std::ostringstream oss;
    int availableCount = 0;
    for (int64_t value : meta.values) {
        if (value != -1) {
            ++availableCount;
        }
    }
    oss << name << " meta: size=" << meta.metadataSize
        << " hex=" << meta.metadataHex
        << " available=" << availableCount
        << " sensor_ts=" << meta.sensorTimestamp
        << " timestamp=" << meta.timestamp
        << " frame_number=" << meta.frameNumber
        << " exposure=" << meta.exposure
        << " gain=" << meta.gain
        << " fps=" << meta.actualFrameRate
        << " gpio=" << meta.gpioInputData;
    return oss.str();
}

std::string permissionToString(OBPermissionType permission) {
    switch (permission) {
        case OB_PERMISSION_DENY: return "deny";
        case OB_PERMISSION_READ: return "read";
        case OB_PERMISSION_WRITE: return "write";
        case OB_PERMISSION_READ_WRITE: return "read_write";
        case OB_PERMISSION_ANY: return "any";
        default: return std::to_string(static_cast<int>(permission));
    }
}

std::string propertyTypeToString(OBPropertyType type) {
    switch (type) {
        case OB_BOOL_PROPERTY: return "bool";
        case OB_INT_PROPERTY: return "int";
        case OB_FLOAT_PROPERTY: return "float";
        case OB_STRUCT_PROPERTY: return "struct";
        default: return std::to_string(static_cast<int>(type));
    }
}

bool canRead(OBPermissionType permission) {
    return permission == OB_PERMISSION_READ || permission == OB_PERMISSION_READ_WRITE;
}

bool canWrite(OBPermissionType permission) {
    return permission == OB_PERMISSION_WRITE || permission == OB_PERMISSION_READ_WRITE;
}

std::string intRangeToString(const OBIntPropertyRange &range) {
    std::ostringstream oss;
    oss << "cur=" << range.cur << " min=" << range.min << " max=" << range.max
        << " step=" << range.step << " def=" << range.def;
    return oss.str();
}

std::string boolRangeToString(const OBBoolPropertyRange &range) {
    std::ostringstream oss;
    oss << "cur=" << range.cur << " def=" << range.def;
    return oss.str();
}

} // namespace

// ============================================================================
//  Construction / Destruction
// ============================================================================

OrbbecProcessor::OrbbecProcessor(const OrbbecColorControlConfig &colorControl)
    : colorControl_(colorControl) {
    // ── Create pipeline & obtain device ────────────────────────────────
    pipeline_ = std::make_shared<ob::Pipeline>();
    device_   = pipeline_->getDevice();
    config_   = std::make_shared<ob::Config>();

    try {
        const bool globalTimestampSupported = device_->isGlobalTimestampSupported();
        Log::info("Orbbec", std::string("Global timestamp supported: ")
                 + (globalTimestampSupported ? "true" : "false"));
        if (globalTimestampSupported) {
            device_->enableGlobalTimestamp(true);
            Log::info("Orbbec", "Global timestamp enabled.");
        }
    }
    catch (const std::exception &e) {
        Log::warn("Orbbec", std::string("Global timestamp setup failed: ") + e.what());
    }

    if (colorControl_.logSupportedProperties) {
        logSupportedDeviceProperties();
    }
    applyColorControl();

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
        Log::info("Orbbec", "Color profile request: 1920x1080 @ 30 fps format="
                  + colorControl_.colorFormatName + " ("
                  + std::to_string(static_cast<int>(colorControl_.colorFormat)) + ")");
        std::shared_ptr<ob::VideoStreamProfile> colorProfile;
        try {
            colorProfile = colorProfiles->getVideoStreamProfile(
                1920, 1080, colorControl_.colorFormat, 30);
        }
        catch (const std::exception &e) {
            if (colorControl_.colorFormat == OB_FORMAT_MJPG) {
                throw;
            }
            Log::warn("Orbbec", "Requested color format is not available: "
                      + std::string(e.what()) + ". Falling back to MJPG.");
            colorProfile = colorProfiles->getVideoStreamProfile(
                1920, 1080, OB_FORMAT_MJPG, 30);
        }
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

    // ── Start pipeline with RGB/depth frame synchronization ─
    config_->setFrameAggregateOutputMode(OB_FRAME_AGGREGATE_OUTPUT_ALL_TYPE_FRAME_REQUIRE);
    pipeline_->enableFrameSync();
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

void OrbbecProcessor::logSupportedDeviceProperties() const {
    if (!device_) return;
    try {
        const int count = device_->getSupportedPropertyCount();
        Log::info("Orbbec", "Supported property count: " + std::to_string(count));
        for (int i = 0; i < count; ++i) {
            const auto item = device_->getSupportedProperty(static_cast<uint32_t>(i));
            const int id = static_cast<int>(item.id);
            const bool colorExposureRelated =
                id == static_cast<int>(OB_PROP_COLOR_AUTO_EXPOSURE_BOOL) ||
                id == static_cast<int>(OB_PROP_COLOR_EXPOSURE_INT) ||
                id == static_cast<int>(OB_PROP_COLOR_GAIN_INT) ||
                id == static_cast<int>(OB_PROP_COLOR_AUTO_WHITE_BALANCE_BOOL) ||
                id == static_cast<int>(OB_PROP_COLOR_WHITE_BALANCE_INT) ||
                id == static_cast<int>(OB_PROP_COLOR_AUTO_EXPOSURE_PRIORITY_INT) ||
                id == static_cast<int>(OB_PROP_COLOR_POWER_LINE_FREQUENCY_INT) ||
                id == static_cast<int>(OB_PROP_COLOR_AE_MAX_EXPOSURE_INT) ||
                id == static_cast<int>(OB_PROP_COLOR_MAXIMAL_GAIN_INT) ||
                id == static_cast<int>(OB_PROP_COLOR_MAXIMAL_SHUTTER_INT) ||
                id == static_cast<int>(OB_PROP_COLOR_SHUTTER_INT) ||
                id == static_cast<int>(OB_PROP_COLOR_HDR_BOOL);
            if (!colorExposureRelated) continue;

            std::ostringstream oss;
            oss << "Property id=" << id
                << " name=" << (item.name ? item.name : "")
                << " type=" << propertyTypeToString(item.type)
                << " permission=" << permissionToString(item.permission);
            try {
                if (canRead(item.permission)) {
                    if (item.type == OB_BOOL_PROPERTY) {
                        oss << " range{" << boolRangeToString(device_->getBoolPropertyRange(item.id)) << "}"
                            << " value=" << device_->getBoolProperty(item.id);
                    }
                    else if (item.type == OB_INT_PROPERTY) {
                        oss << " range{" << intRangeToString(device_->getIntPropertyRange(item.id)) << "}"
                            << " value=" << device_->getIntProperty(item.id);
                    }
                }
                oss << " writable=" << (canWrite(item.permission) ? "true" : "false");
            }
            catch (const std::exception &e) {
                oss << " read_failed=" << e.what();
            }
            Log::info("Orbbec", oss.str());
        }
    }
    catch (const std::exception &e) {
        Log::warn("Orbbec", std::string("Supported property logging failed: ") + e.what());
    }
}

void OrbbecProcessor::applyColorControl() {
    if (!device_) return;

    auto setBool = [this](OBPropertyID id, bool value, const std::string &name) {
        try {
            if (!device_->isPropertySupported(id, OB_PERMISSION_WRITE) &&
                !device_->isPropertySupported(id, OB_PERMISSION_READ_WRITE)) {
                Log::warn("Orbbec", name + " is not writable on this device.");
                return;
            }
            device_->setBoolProperty(id, value);
            Log::info("Orbbec", "Set " + name + "=" + std::to_string(value));
        }
        catch (const std::exception &e) {
            Log::warn("Orbbec", "Failed to set " + name + ": " + e.what());
        }
    };
    auto setInt = [this](OBPropertyID id, int32_t value, const std::string &name) {
        try {
            if (!device_->isPropertySupported(id, OB_PERMISSION_WRITE) &&
                !device_->isPropertySupported(id, OB_PERMISSION_READ_WRITE)) {
                Log::warn("Orbbec", name + " is not writable on this device.");
                return;
            }
            try {
                const auto range = device_->getIntPropertyRange(id);
                Log::info("Orbbec", name + " range before set: " + intRangeToString(range));
            }
            catch (...) {
            }
            device_->setIntProperty(id, value);
            Log::info("Orbbec", "Set " + name + "=" + std::to_string(value));
        }
        catch (const std::exception &e) {
            Log::warn("Orbbec", "Failed to set " + name + ": " + e.what());
        }
    };

    if (colorControl_.autoExposure) {
        setBool(OB_PROP_COLOR_AUTO_EXPOSURE_BOOL, *colorControl_.autoExposure, "color_auto_exposure");
    }
    if (colorControl_.exposure) {
        setInt(OB_PROP_COLOR_EXPOSURE_INT, *colorControl_.exposure, "color_exposure");
    }
    if (colorControl_.gain) {
        setInt(OB_PROP_COLOR_GAIN_INT, *colorControl_.gain, "color_gain");
    }
    if (colorControl_.autoWhiteBalance) {
        setBool(OB_PROP_COLOR_AUTO_WHITE_BALANCE_BOOL, *colorControl_.autoWhiteBalance, "color_auto_white_balance");
    }
    if (colorControl_.whiteBalance) {
        setInt(OB_PROP_COLOR_WHITE_BALANCE_INT, *colorControl_.whiteBalance, "color_white_balance");
    }
    if (colorControl_.autoExposurePriority) {
        setInt(OB_PROP_COLOR_AUTO_EXPOSURE_PRIORITY_INT, *colorControl_.autoExposurePriority, "color_auto_exposure_priority");
    }
    if (colorControl_.powerLineFrequency) {
        setInt(OB_PROP_COLOR_POWER_LINE_FREQUENCY_INT, *colorControl_.powerLineFrequency, "color_power_line_frequency");
    }

    Log::info("Orbbec", "Color property snapshot after config: auto_exposure="
              + std::to_string(readBoolPropertyOrDefault(OB_PROP_COLOR_AUTO_EXPOSURE_BOOL))
              + " exposure=" + std::to_string(readIntPropertyOrDefault(OB_PROP_COLOR_EXPOSURE_INT))
              + " gain=" + std::to_string(readIntPropertyOrDefault(OB_PROP_COLOR_GAIN_INT))
              + " auto_wb=" + std::to_string(readBoolPropertyOrDefault(OB_PROP_COLOR_AUTO_WHITE_BALANCE_BOOL))
              + " wb=" + std::to_string(readIntPropertyOrDefault(OB_PROP_COLOR_WHITE_BALANCE_INT))
              + " ae_priority=" + std::to_string(readIntPropertyOrDefault(OB_PROP_COLOR_AUTO_EXPOSURE_PRIORITY_INT))
              + " power_line_frequency=" + std::to_string(readIntPropertyOrDefault(OB_PROP_COLOR_POWER_LINE_FREQUENCY_INT)));
}

int32_t OrbbecProcessor::readIntPropertyOrDefault(OBPropertyID propertyId, int32_t defaultValue) const {
    if (!device_) return defaultValue;
    try {
        if (device_->isPropertySupported(propertyId, OB_PERMISSION_READ) ||
            device_->isPropertySupported(propertyId, OB_PERMISSION_READ_WRITE)) {
            return device_->getIntProperty(propertyId);
        }
    }
    catch (...) {
    }
    return defaultValue;
}

int32_t OrbbecProcessor::readBoolPropertyOrDefault(OBPropertyID propertyId, int32_t defaultValue) const {
    if (!device_) return defaultValue;
    try {
        if (device_->isPropertySupported(propertyId, OB_PERMISSION_READ) ||
            device_->isPropertySupported(propertyId, OB_PERMISSION_READ_WRITE)) {
            return device_->getBoolProperty(propertyId) ? 1 : 0;
        }
    }
    catch (...) {
    }
    return defaultValue;
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
    constexpr uint64_t PROPERTY_SAMPLE_INTERVAL_FRAMES = 30;
    int32_t cachedColorAutoExposure = readBoolPropertyOrDefault(OB_PROP_COLOR_AUTO_EXPOSURE_BOOL);
    int32_t cachedColorExposure = readIntPropertyOrDefault(OB_PROP_COLOR_EXPOSURE_INT);
    int32_t cachedColorGain = readIntPropertyOrDefault(OB_PROP_COLOR_GAIN_INT);
    int32_t cachedColorAutoWhiteBalance = readBoolPropertyOrDefault(OB_PROP_COLOR_AUTO_WHITE_BALANCE_BOOL);
    int32_t cachedColorWhiteBalance = readIntPropertyOrDefault(OB_PROP_COLOR_WHITE_BALANCE_INT);
    int32_t cachedColorAutoExposurePriority = readIntPropertyOrDefault(OB_PROP_COLOR_AUTO_EXPOSURE_PRIORITY_INT);
    int32_t cachedColorPowerLineFrequency = readIntPropertyOrDefault(OB_PROP_COLOR_POWER_LINE_FREQUENCY_INT);

    while (!stopRequested_.load()) {
        try {
            // Wait up to 200 ms for a frameset (> 1 frame period at 30fps)
            auto frameSet = pipeline_->waitForFrames(200);
            if (!frameSet) continue;
            const int64_t hostArrivalUs = steadyNowUs();

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
            const auto colorMeta = readMetadataSnapshot(colorFrame);
            const auto depthMeta = readMetadataSnapshot(depthFrame);
            const auto rawColorMeta = readMetadataSnapshot(rawColor);
            const auto rawDepthMeta = readMetadataSnapshot(rawDepth);
            const int64_t rawDeltaUs = static_cast<int64_t>(rawColor->timeStampUs())
                                     - static_cast<int64_t>(rawDepth->timeStampUs());
            const int64_t alignedDeltaUs = static_cast<int64_t>(oColorTs)
                                         - static_cast<int64_t>(oDepthTs);

            // First-frame logging (mirrors the Python version)
            if (!firstFrameLogged_) {
                Log::info("Orbbec", "First raw pair - color_idx=" + std::to_string(rawColor->index())
                         + " depth_idx=" + std::to_string(rawDepth->index())
                         + " color_ts=" + std::to_string(rawColor->timeStampUs())
                         + " depth_ts=" + std::to_string(rawDepth->timeStampUs())
                         + " delta=" + std::to_string(rawDeltaUs) + " us");
                Log::info("Orbbec", "First color frame - device_ts=" + std::to_string(oColorTs)
                         + " idx=" + std::to_string(colorFrame->index())
                         + " us  sys_ts=" + std::to_string(colorFrame->systemTimeStampUs()) + " us");
                Log::info("Orbbec", "First depth frame - device_ts=" + std::to_string(oDepthTs)
                         + " idx=" + std::to_string(depthFrame->index())
                         + " us  sys_ts=" + std::to_string(depthFrame->systemTimeStampUs()) + " us");
                Log::info("Orbbec", "First color global_ts=" + std::to_string(colorFrame->globalTimeStampUs())
                         + " raw_color_global_ts=" + std::to_string(rawColor->globalTimeStampUs()));
                Log::info("Orbbec", metadataSummary("aligned color", colorMeta));
                Log::info("Orbbec", metadataSummary("aligned depth", depthMeta));
                Log::info("Orbbec", metadataSummary("raw color", rawColorMeta));
                Log::info("Orbbec", metadataSummary("raw depth", rawDepthMeta));
                Log::info("Orbbec", "First aligned pair delta=" + std::to_string(alignedDeltaUs) + " us");
                firstFrameLogged_ = true;
            }
            if (fpsFrameCount % 30 == 0 && std::llabs(alignedDeltaUs) > 5000) {
                Log::warn("Orbbec", "RGB/depth timestamp delta is large: raw_delta="
                         + std::to_string(rawDeltaUs) + " us aligned_delta="
                         + std::to_string(alignedDeltaUs) + " us");
            }
            if (fpsFrameCount % PROPERTY_SAMPLE_INTERVAL_FRAMES == 0) {
                cachedColorAutoExposure = readBoolPropertyOrDefault(OB_PROP_COLOR_AUTO_EXPOSURE_BOOL, cachedColorAutoExposure);
                cachedColorExposure = readIntPropertyOrDefault(OB_PROP_COLOR_EXPOSURE_INT, cachedColorExposure);
                cachedColorGain = readIntPropertyOrDefault(OB_PROP_COLOR_GAIN_INT, cachedColorGain);
                cachedColorAutoWhiteBalance = readBoolPropertyOrDefault(OB_PROP_COLOR_AUTO_WHITE_BALANCE_BOOL, cachedColorAutoWhiteBalance);
                cachedColorWhiteBalance = readIntPropertyOrDefault(OB_PROP_COLOR_WHITE_BALANCE_INT, cachedColorWhiteBalance);
                cachedColorAutoExposurePriority = readIntPropertyOrDefault(OB_PROP_COLOR_AUTO_EXPOSURE_PRIORITY_INT, cachedColorAutoExposurePriority);
                cachedColorPowerLineFrequency = readIntPropertyOrDefault(OB_PROP_COLOR_POWER_LINE_FREQUENCY_INT, cachedColorPowerLineFrequency);
            }

            // ── Color → raw bytes ──────────────────────────────────────
            uint32_t cW = colorFrame->width();
            uint32_t cH = colorFrame->height();
            OBFormat cFormat = colorFrame->format();
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
                frame.colorFormat     = cFormat;

                frame.depthData       = std::move(depthBuf);
                frame.depthWidth      = dW;
                frame.depthHeight     = dH;

                frame.colorTimestampUs = oColorTs;
                frame.depthTimestampUs = oDepthTs;
                frame.colorSystemTimestampUs = colorFrame->systemTimeStampUs();
                frame.depthSystemTimestampUs = depthFrame->systemTimeStampUs();
                frame.colorGlobalTimestampUs = colorFrame->globalTimeStampUs();
                frame.depthGlobalTimestampUs = depthFrame->globalTimeStampUs();

                frame.colorFrameIndex = colorFrame->index();
                frame.depthFrameIndex = depthFrame->index();
                frame.rawColorFrameIndex = rawColor->index();
                frame.rawDepthFrameIndex = rawDepth->index();
                frame.rawColorTimestampUs = rawColor->timeStampUs();
                frame.rawDepthTimestampUs = rawDepth->timeStampUs();
                frame.rawColorSystemTimestampUs = rawColor->systemTimeStampUs();
                frame.rawDepthSystemTimestampUs = rawDepth->systemTimeStampUs();
                frame.rawColorGlobalTimestampUs = rawColor->globalTimeStampUs();
                frame.rawDepthGlobalTimestampUs = rawDepth->globalTimeStampUs();
                frame.hostArrivalTimestampUs = hostArrivalUs;
                frame.colorMetadata = colorMeta;
                frame.depthMetadata = depthMeta;
                frame.rawColorMetadata = rawColorMeta;
                frame.rawDepthMetadata = rawDepthMeta;
                frame.colorAutoExposure = cachedColorAutoExposure;
                frame.colorExposure = cachedColorExposure;
                frame.colorGain = cachedColorGain;
                frame.colorAutoWhiteBalance = cachedColorAutoWhiteBalance;
                frame.colorWhiteBalance = cachedColorWhiteBalance;
                frame.colorAutoExposurePriority = cachedColorAutoExposurePriority;
                frame.colorPowerLineFrequency = cachedColorPowerLineFrequency;

                frame.valid = true;
                frame.producedSeq = producedFrameCount_.fetch_add(1) + 1;

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
