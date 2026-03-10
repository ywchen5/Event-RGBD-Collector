#include "prophesee_processor.hpp"
#include "logger.hpp"
#include <chrono>
#include <cstring>
#include <algorithm>

// ============================================================================
//  Construction / Destruction
// ============================================================================

PropheseeProcessor::PropheseeProcessor(int accumulationTimeUs,
                                       const std::string &filePath)
    : accumulationTimeUs_(accumulationTimeUs)
    , filePath_(filePath)
{
    // ── Open the camera ────────────────────────────────────────────────
    if (filePath_.empty()) {
        cam_ = Metavision::Camera::from_first_available();
    } else {
        cam_ = Metavision::Camera::from_file(filePath_);
    }

    // ── Sensor geometry ────────────────────────────────────────────────
    auto &geometry = cam_.geometry();
    width_  = geometry.width();
    height_ = geometry.height();
    Log::info("Prophesee", "Sensor geometry: " + std::to_string(width_) + " x " + std::to_string(height_));

    // ── Biases (diff_on = 76, diff_off = 20) ──────────────────────────
    try {
        auto *biases = cam_.biases().get_facility();
        if (biases) {
            biases->set("bias_diff_on",  76);
            biases->set("bias_diff_off", 20);
            Log::info("Prophesee", "Biases set: diff_on=76, diff_off=20");
        }
    }
    catch (const std::exception &e) {
        Log::warn("Prophesee", std::string("Could not set biases: ") + e.what());
    }

    // ── ERC (Event Rate Control) – 20 M events/sec ────────────────────
    try {
        auto &erc = cam_.erc_module();
        erc.enable(true);
        erc.set_cd_event_rate(20000000); // @code-review: control the max event rate
        Log::info("Prophesee", "ERC enabled at 20M events/sec");
    }
    catch (const std::exception &e) {
        Log::warn("Prophesee", std::string("Could not configure ERC: ") + e.what());
    }

    // ── External trigger (MAIN channel, rising edge) ───────────────────
    try {
        auto *trigger_in = cam_.get_device().get_facility<Metavision::I_TriggerIn>();
        if (trigger_in) {
            trigger_in->enable(Metavision::I_TriggerIn::Channel::Main);
            Log::info("Prophesee", "Trigger-in enabled on MAIN channel");
        }
    }
    catch (const std::exception &e) {
        Log::warn("Prophesee", std::string("Could not enable trigger-in: ") + e.what());
    }

    Log::info("Prophesee", "Initialised.");
}

PropheseeProcessor::~PropheseeProcessor() {
    stop();
    Log::info("Prophesee", "Destroyed.");
}

// ============================================================================
//  Thread management
// ============================================================================

void PropheseeProcessor::start() {
    if (running_.load()) return;

    stopRequested_.store(false);
    running_.store(true);
    workerThread_ = std::thread(&PropheseeProcessor::processingLoop, this);
    Log::info("Prophesee", "Worker thread started.");
}

void PropheseeProcessor::stop() {
    if (!running_.load()) return;

    stopRequested_.store(true);
    if (workerThread_.joinable()) {
        workerThread_.join();
    }
    running_.store(false);
    Log::info("Prophesee", "Worker thread stopped.");
}

// ============================================================================
//  Thread-safe accessors
// ============================================================================

bool PropheseeProcessor::getLatestSlice(EventSliceData &out) {
    std::lock_guard<std::mutex> lock(sliceMutex_);
    if (sliceQueue_.empty()) return false;

    out = sliceQueue_.back();   // peek at newest
    return true;
}

bool PropheseeProcessor::popSlice(EventSliceData &out) {
    std::lock_guard<std::mutex> lock(sliceMutex_);
    if (sliceQueue_.empty()) return false;

    out = std::move(sliceQueue_.front());
    sliceQueue_.pop_front();
    return true;
}

size_t PropheseeProcessor::sliceQueueSize() const {
    std::lock_guard<std::mutex> lock(sliceMutex_);
    return sliceQueue_.size();
}

bool PropheseeProcessor::getLatestFrame(CdFrameData &out) {
    if (!newFrameReady_.load()) return false;

    std::lock_guard<std::mutex> lock(frameMutex_);
    out = frameFront_;
    newFrameReady_.store(false);
    return true;
}

// ============================================================================
//  Processing loop (runs on worker thread)
// ============================================================================

void PropheseeProcessor::processingLoop() {
    // ── PeriodicFrameGenerationAlgorithm for visualisation ─────────────
    Metavision::PeriodicFrameGenerationAlgorithm frameGen(
        width_, height_, accumulationTimeUs_, /* fps */ 30);

    frameGen.set_output_callback(
        [this](Metavision::timestamp ts, cv::Mat &frame) {
            // Convert OpenCV Mat to raw byte buffer and publish
            CdFrameData fData;
            fData.width  = static_cast<uint32_t>(frame.cols);
            fData.height = static_cast<uint32_t>(frame.rows);
            fData.ts     = ts;

            size_t bytes = frame.total() * frame.elemSize();
            fData.frameData.resize(bytes);
            std::memcpy(fData.frameData.data(), frame.data, bytes);
            fData.valid = true;

            {
                std::lock_guard<std::mutex> lock(frameMutex_);
                frameFront_ = std::move(fData);
                newFrameReady_.store(true);
            }
        });

    // ── CD event callback ──────────────────────────────────────────────
    //    • Feed events to PeriodicFrameGenerationAlgorithm for visualisation
    //    • Buffer raw events for trigger-based slicing
    cam_.cd().add_callback(
        [this, &frameGen](const Metavision::EventCD *begin,
                          const Metavision::EventCD *end) {
            // 1) Feed to frame generator
            frameGen.process_events(begin, end);

            // 2) Buffer for trigger slicing (only after first trigger)
            if (triggerActive_) {
                for (auto it = begin; it != end; ++it) {
                    eventBuffer_.push_back({
                        static_cast<uint16_t>(it->x),
                        static_cast<uint16_t>(it->y),
                        static_cast<int16_t>(it->p),
                        static_cast<int64_t>(it->t)
                    });
                }
            }
        });

    // ── External-trigger callback ──────────────────────────────────────
    //    Mirrors the Python trigger-slicing logic:
    //      - First rising edge → set active, record timestamp
    //      - Subsequent rising edges → flush buffer as a complete slice,
    //        reset buffer for the next interval
    cam_.ext_trigger().add_callback(
        [this](const Metavision::EventExtTrigger *begin,
               const Metavision::EventExtTrigger *end) {
            for (auto it = begin; it != end; ++it) {
                if (it->p != 1) continue;   // only rising edges

                int64_t currentTrigTs = static_cast<int64_t>(it->t);
                
                // @code-review: maybe a crucial part!!!
                // ── Minimum-interval filter ─────────────────────────
                //    Reject triggers that arrive less than MIN_TRIGGER_INTERVAL_US
                //    after the last accepted trigger.  This eliminates
                //    spurious double-triggers (bounce, dual-edge, etc.)
                if (triggerActive_ &&
                    (currentTrigTs - lastTriggerTs_) < MIN_TRIGGER_INTERVAL_US) {
                    trigRejected_++;
                    continue;  // skip this trigger
                }

                trigAccepted_++;
                // @code-review: here the triggerActive_ design indicates that
                //    the trigger can only be activated once, so do not test the case where
                //    the trigger is deactivated and then reactivated for now.
                //    For future extension, this restarting mechanism can be used to improve the robustness of the system
                //    But for now, it is not essential for our system.
                if (!triggerActive_) {
                    // ── First trigger activation ───────────────────────
                    triggerActive_ = true;
                    lastTriggerTs_ = currentTrigTs;

                    // Any events in the buffer that are AFTER the trigger
                    // belong to the first slice interval → keep them.
                    // Events before → discard.
                    auto keep = std::remove_if(
                        eventBuffer_.begin(), eventBuffer_.end(),
                        [currentTrigTs](const CdEvent &e) {
                            return e.t <= currentTrigTs;
                        });
                    eventBuffer_.erase(keep, eventBuffer_.end());

                    Log::info("Prophesee", "First trigger at " + std::to_string(currentTrigTs) + " us");
                } else {
                    // ── Subsequent trigger → slice the buffered events ──
                    // Partition: events with t <= currentTrigTs go into
                    // this slice; the rest stay for the next one.
                    std::vector<CdEvent> sliceEvents;
                    std::vector<CdEvent> remaining;
                    sliceEvents.reserve(eventBuffer_.size());

                    for (auto &ev : eventBuffer_) {
                        if (ev.t <= currentTrigTs) {
                            sliceEvents.push_back(ev);
                        } else {
                            remaining.push_back(ev);
                        }
                    }
                    eventBuffer_ = std::move(remaining);

                    // Publish slice
                    if (!sliceEvents.empty()) {
                        EventSliceData slice;
                        slice.events  = std::move(sliceEvents);
                        slice.startTs = lastTriggerTs_;
                        slice.endTs   = currentTrigTs;
                        slice.valid   = true;

                        {
                            std::lock_guard<std::mutex> lock(sliceMutex_);
                            if (sliceQueue_.size() >= MAX_SLICE_QUEUE) {
                                sliceQueue_.pop_front();  // drop oldest
                            }
                            sliceQueue_.push_back(std::move(slice));
                            newSliceReady_.store(true);
                        }
                    }

                    lastTriggerTs_ = currentTrigTs;
                }
            }
        });

    // ── Start the camera streaming ─────────────────────────────────────
    cam_.start();
    Log::info("Prophesee", "Camera streaming started.");

    // ── FPS measurement ────────────────────────────────────────────────
    uint64_t sliceCount = 0;
    uint64_t eventCount = 0;
    auto     fpsStart   = std::chrono::steady_clock::now();
    constexpr double FPS_LOG_INTERVAL_SEC = 5.0;

    // ── Spin while running ─────────────────────────────────────────────
    while (!stopRequested_.load() && cam_.is_running()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));

        // Periodic slice-rate log
        if (newSliceReady_.load()) {
            sliceCount++;
            // Count events from the latest slice
            {
                std::lock_guard<std::mutex> lock(sliceMutex_);
                if (!sliceQueue_.empty()) {
                    eventCount += sliceQueue_.back().events.size();
                }
            }
        }
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - fpsStart).count();
        if (elapsed >= FPS_LOG_INTERVAL_SEC) {
            evsFps_ = sliceCount / elapsed;
            eventRate_ = eventCount / elapsed;
            sliceCount = 0;
            eventCount = 0;
            fpsStart   = now;
        }
    }

    // ── Cleanup ────────────────────────────────────────────────────────
    cam_.stop();
    Log::info("Prophesee", "Camera streaming stopped.");
}
