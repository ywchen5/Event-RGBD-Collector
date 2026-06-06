#include "sync_processor.hpp"
#include "logger.hpp"
#include <chrono>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <limits>
#include <sstream>

namespace {
constexpr double PI = 3.14159265358979323846;
}

// ============================================================================
//  Construction / Destruction
// ============================================================================

SyncProcessor::SyncProcessor(OrbbecProcessor &orbbec,
                             PropheseeProcessor &prophesee)
    : orbbec_(orbbec)
    , prophesee_(prophesee)
    , startTime_(std::chrono::steady_clock::now())
{
    Log::info("Sync", "Initialised.");
}

SyncProcessor::~SyncProcessor() {
    stop();
    Log::info("Sync", "Destroyed.");
}

// ============================================================================
//  Thread management
// ============================================================================

void SyncProcessor::start() {
    if (running_.load()) return;

    stopRequested_.store(false);
    running_.store(true);
    workerThread_ = std::thread(&SyncProcessor::syncLoop, this);
    Log::info("Sync", "Sync thread started.");
}

void SyncProcessor::stop() {
    if (!running_.load()) return;

    stopRequested_.store(true);
    if (workerThread_.joinable()) {
        workerThread_.join();
    }
    running_.store(false);
    Log::info("Sync", "Sync thread stopped.  Pairs=" + std::to_string(totalPairCount_)
             + "  OrbDrops=" + std::to_string(orbDropCount_)
             + "  EvsDrops=" + std::to_string(evsDropCount_));
}

// ============================================================================
//  Thread-safe accessor
// ============================================================================

bool SyncProcessor::getLatestPair(SyncedPair &out) {
    if (!newPairReady_.load()) return false;

    std::lock_guard<std::mutex> lock(pairMutex_);
    out = pairFront_;
    newPairReady_.store(false);
    return true;
}

bool SyncProcessor::popPair(SyncedPair &out) {
    std::lock_guard<std::mutex> lock(pairQueueMutex_);
    if (pairQueue_.empty()) return false;

    out = std::move(pairQueue_.front());
    pairQueue_.pop_front();
    return true;
}

size_t SyncProcessor::pairQueueSize() const {
    std::lock_guard<std::mutex> lock(pairQueueMutex_);
    return pairQueue_.size();
}

void SyncProcessor::setCallback(PairCallback cb) {
    std::lock_guard<std::mutex> lock(cbMutex_);
    callback_ = std::move(cb);
}

// @code-review: here the bootstrap alignment procedure also indicates that we 
// can not do re-activation on trigger for now. The system has to be restarted
// if the trigger is deactivated and reactivated.
// for future extension, the restarting mechanism can be implemented 
// by extend the function of bootstrap procedure

// ============================================================================
//  Bootstrap alignment
// ============================================================================
//
//  At startup the two devices power up at different times.  One may have
//  accumulated many frames before the other delivers its first.  We
//  cannot compare raw timestamps across device clocks (they are in
//  completely different domains 鈥?e.g. Orbbec ~45 billion us vs
//  Prophesee ~16 million us).
//
//  Strategy:
//    1. Wait until both buffers have 鈮?MIN_BOOTSTRAP_SAMPLES entries,
//       which proves both devices are actively streaming.
//    2. In this diagnostic branch, keep startup buffers intact.
//    3. Initialise deltaOrbToEvs_ from the first available pair.
//    4. Record only a few pairs so startup ordering can be inspected.
//
bool SyncProcessor::bootstrapAlignment() {
    if (orbBuf_.size() < MIN_BOOTSTRAP_SAMPLES ||
        evsBuf_.size() < MIN_BOOTSTRAP_SAMPLES) {
        return false;
    }

    if (orbBuf_.front().hostArrivalTimestampUs <= 0 ||
        evsBuf_.front().triggerEndHostReceiptUs <= 0) {
        Log::warn("Sync", "Waiting for host phase timestamps before bootstrap.");
        return false;
    }

    const size_t orbInitial = orbBuf_.size();
    const size_t evsInitial = evsBuf_.size();
    const auto initialOrbFront = orbBuf_.front();
    const auto initialEvsFront = evsBuf_.front();

    size_t orbDiscarded = 0;
    const size_t evsDiscarded = 0;

    startupHostPhaseUs_ =
        initialOrbFront.hostArrivalTimestampUs -
        initialEvsFront.triggerEndHostReceiptUs;

    if (startupHostPhaseUs_ < RGB_PHASE_THRESHOLD_US) {
        // In this delivery phase E_k visually matches RGB_{k+1}.
        if (orbBuf_.size() < 2) {
            return false;
        }
        orbBuf_.pop_front();
        orbDiscarded = 1;
        orbDropCount_++;
        appliedRgbFrameOffset_ = 1;
    } else {
        appliedRgbFrameOffset_ = 0;
    }

    if (orbBuf_.empty()) {
        return false;
    }

    const auto orbFront = orbBuf_.front();
    const auto evsFront = evsBuf_.front();

    int64_t orbTs = static_cast<int64_t>(orbFront.colorTimestampUs);
    int64_t evsTs = evsFront.startTs;
    deltaOrbToEvs_ = evsTs - orbTs;
    initialDelta_ = deltaOrbToEvs_;

    Log::LogBlock blk("Bootstrap Alignment");
    blk.kv("Mode", "host-phase-auto-rgb-offset");
    blk.kv("Startup host phase", std::to_string(startupHostPhaseUs_) + " us");
    blk.kv("Phase threshold", std::to_string(RGB_PHASE_THRESHOLD_US) + " us");
    blk.kv("Applied RGB frame offset", appliedRgbFrameOffset_);
    blk.kv("Min samples", MIN_BOOTSTRAP_SAMPLES);
    blk.kv("Orb initial", orbInitial);
    blk.kv("Evs initial", evsInitial);
    blk.kv("Producer Orb frames", orbbec_.producedFrameCount());
    blk.kv("Producer Evs triggers", prophesee_.trigAccepted());
    blk.kv("Producer Evs slices", prophesee_.slicesProduced());
    blk.kv("Orb discarded", orbDiscarded);
    blk.kv("Evs discarded", evsDiscarded);
    blk.kv("Initial Orb front", "seq=" + std::to_string(initialOrbFront.producedSeq)
           + " color_idx=" + std::to_string(initialOrbFront.colorFrameIndex)
           + " host=" + std::to_string(initialOrbFront.hostArrivalTimestampUs) + " us");
    blk.kv("Initial Evs front", "slice=" + std::to_string(initialEvsFront.sliceSeq)
           + " trig=" + std::to_string(initialEvsFront.triggerStartSeq)
           + "->" + std::to_string(initialEvsFront.triggerEndSeq)
           + " trigger_end_host="
           + std::to_string(initialEvsFront.triggerEndHostReceiptUs) + " us");
    blk.kv("Orb selected", "seq=" + std::to_string(orbFront.producedSeq)
           + " color_idx=" + std::to_string(orbFront.colorFrameIndex)
           + " ts=" + std::to_string(orbFront.colorTimestampUs) + " us");
    blk.kv("Evs selected", "slice=" + std::to_string(evsFront.sliceSeq)
           + " trig=" + std::to_string(evsFront.triggerStartSeq)
           + "->" + std::to_string(evsFront.triggerEndSeq)
           + " ts=" + std::to_string(evsFront.startTs)
           + "->" + std::to_string(evsFront.endTs) + " us");
    blk.kv("Seq delta evs-orb", static_cast<int64_t>(evsFront.sliceSeq)
           - static_cast<int64_t>(orbFront.producedSeq));
    blk.kv("OrbTs", std::to_string(orbTs) + " us");
    blk.kv("EvsTs", std::to_string(evsTs) + " us");
    blk.kv("Delta (evs-orb)", std::to_string(deltaOrbToEvs_) + " us");
    Log::banner(blk.title(), blk.body());

    aligned_ = true;
    return true;
}

// ============================================================================
//  Sync loop
// ============================================================================

void SyncProcessor::syncLoop() {
    Log::info("Sync", "Waiting for both sensors...");

    auto lastMonitor = std::chrono::steady_clock::now();
    constexpr double MONITOR_INTERVAL_SEC = 5.0;

    while (!stopRequested_.load()) {
        bool hasData = false;

        // 鈹€鈹€ 1. Drain Orbbec queue into local deque 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        {
            OrbbecFrameData tmp;
            while (orbbec_.popFrame(tmp)) {
                if (tmp.valid) {
                    const uint64_t orbProducedSeq = tmp.producedSeq;
                    const uint64_t colorFrameIndex = tmp.colorFrameIndex;
                    const uint64_t depthFrameIndex = tmp.depthFrameIndex;
                    const uint64_t rawColorFrameIndex = tmp.rawColorFrameIndex;
                    const uint64_t rawDepthFrameIndex = tmp.rawDepthFrameIndex;
                    const uint64_t colorTimestampUs = tmp.colorTimestampUs;
                    const uint64_t latestEventTriggers = prophesee_.trigAccepted();
                    const uint64_t latestEventSlices = prophesee_.slicesProduced();
                    const size_t orbBufBefore = orbBuf_.size();
                    const size_t evsBufBefore = evsBuf_.size();
                    uint64_t syncLatestSliceSeq = 0;
                    uint64_t syncLatestTriggerStartSeq = 0;
                    uint64_t syncLatestTriggerEndSeq = 0;
                    int64_t syncLatestEventStartTs = 0;
                    int64_t syncLatestEventEndTs = 0;
                    if (!evsBuf_.empty()) {
                        const auto& latest = evsBuf_.back();
                        syncLatestSliceSeq = latest.sliceSeq;
                        syncLatestTriggerStartSeq = latest.triggerStartSeq;
                        syncLatestTriggerEndSeq = latest.triggerEndSeq;
                        syncLatestEventStartTs = latest.startTs;
                        syncLatestEventEndTs = latest.endTs;
                    }

                    if (orbBuf_.size() >= MAX_BUFFER) {
                        orbBuf_.pop_front();
                        orbDropCount_++;
                    }
                    orbBuf_.push_back(std::move(tmp));
                    if (orbEnqueueLogCount_ < 24) {
                        ++orbEnqueueLogCount_;
                        Log::info("Sync", "Orb enqueue #" + std::to_string(orbEnqueueLogCount_)
                                  + " orb_produced_seq=" + std::to_string(orbProducedSeq)
                                  + " color_idx=" + std::to_string(colorFrameIndex)
                                  + " depth_idx=" + std::to_string(depthFrameIndex)
                                  + " raw_color_idx=" + std::to_string(rawColorFrameIndex)
                                  + " raw_depth_idx=" + std::to_string(rawDepthFrameIndex)
                                  + " color_ts=" + std::to_string(colorTimestampUs)
                                  + " orb_buf_before=" + std::to_string(orbBufBefore)
                                  + " orb_buf_after=" + std::to_string(orbBuf_.size())
                                  + " producer_event_triggers=" + std::to_string(latestEventTriggers)
                                  + " producer_event_slices=" + std::to_string(latestEventSlices)
                                  + " sync_evs_buf_before_event_drain=" + std::to_string(evsBufBefore)
                                  + " sync_latest_slice_seq=" + std::to_string(syncLatestSliceSeq)
                                  + " sync_latest_trigger_seq=" + std::to_string(syncLatestTriggerStartSeq)
                                  + "->" + std::to_string(syncLatestTriggerEndSeq)
                                  + " sync_latest_event_ts=" + std::to_string(syncLatestEventStartTs)
                                  + "->" + std::to_string(syncLatestEventEndTs));
                    }
                    hasData = true;
                }
            }
        }

        // 鈹€鈹€ 2. Drain Prophesee queue into local deque 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        {
            EventSliceData tmp;
            while (prophesee_.popSlice(tmp)) {
                if (tmp.valid) {
                    if (evsBuf_.size() >= MAX_BUFFER) {
                        evsBuf_.pop_front();
                        evsDropCount_++;
                    }
                    evsBuf_.push_back(std::move(tmp));
                    hasData = true;
                }
            }
        }

        // 鈹€鈹€ 2.5  Bootstrap: wait for enough data, then align 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        if (!aligned_) {
            if (!bootstrapAlignment()) {
                // Not ready yet 鈥?log status occasionally
                auto now = std::chrono::steady_clock::now();
                double elapsed = std::chrono::duration<double>(now - lastMonitor).count();
                if (elapsed >= MONITOR_INTERVAL_SEC) {
                    Log::info("Sync", "Waiting for bootstrap鈥? OrbBuf:" + std::to_string(orbBuf_.size())
                             + "  EvsBuf:" + std::to_string(evsBuf_.size()));
                    lastMonitor = now;
                }

                if (!hasData) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
                continue;   // skip pairing until aligned
            }
            // Bootstrap just succeeded 鈥?fall through to pairing
        }

        // 鈹€鈹€ 3. Nearest-timestamp pairing 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        //
        //    The event camera produces N trigger slices per Orbbec
        //    frame (N >= 1, typically 2 at 60 Hz triggers vs 30 fps).
        //    For each Orbbec frame, find the event slice with the
        //    closest mapped timestamp, pair them, discard extras.
        //    No REALIGN / RE-BOOTSTRAP needed.
        //
        while (!orbBuf_.empty() && !evsBuf_.empty()) {

            int64_t orbTs       = static_cast<int64_t>(orbBuf_.front().colorTimestampUs);
            int64_t pairDeltaOrbToEvs = deltaOrbToEvs_;
            int64_t mappedOrbTs = orbTs + pairDeltaOrbToEvs;

            // --- Find closest event slice ------------------------------------
            size_t  bestIdx  = 0;
            int64_t bestDiff = std::abs(evsBuf_[0].startTs - mappedOrbTs);

            for (size_t i = 1; i < evsBuf_.size(); ++i) {
                int64_t diff = std::abs(evsBuf_[i].startTs - mappedOrbTs);
                if (diff < bestDiff) {
                    bestDiff = diff;
                    bestIdx  = i;
                } else {
                    break;   // timestamps are monotonic
                }
            }

            // If the best candidate is the very last entry AND still
            // in the future, wait for more slices before committing.
            if (bestIdx == evsBuf_.size() - 1 &&
                evsBuf_[bestIdx].startTs > mappedOrbTs + FRAME_INTERVAL_US) {
                break;
            }

            // Discard earlier event slices (extras from N:1 ratio)
            for (size_t i = 0; i < bestIdx; ++i) {
                evsBuf_.pop_front();
                evsDropCount_++;
            }

            // Pop matched pair
            auto evsData = std::move(evsBuf_.front());  evsBuf_.pop_front();
            auto orbData = std::move(orbBuf_.front());  orbBuf_.pop_front();

            int64_t evsTs     = evsData.startTs;
            int64_t clockDiff = evsTs - mappedOrbTs;

            // EMA drift tracking
            deltaOrbToEvs_ += static_cast<int64_t>(clockDiff * DRIFT_ALPHA);
            diffSamples_.push_back(std::abs(clockDiff));

            // Build synchronised pair
            SyncedPair pair;
            pair.orbbec      = std::move(orbData);
            pair.events      = std::move(evsData);
            pair.seqNum      = nextSeq_++;
            pair.clockDiffUs = clockDiff;
            pair.deltaOrbToEvsUs = pairDeltaOrbToEvs;
            pair.mappedColorTimestampUs = static_cast<int64_t>(pair.orbbec.colorTimestampUs) + pairDeltaOrbToEvs;
            pair.mappedDepthTimestampUs = static_cast<int64_t>(pair.orbbec.depthTimestampUs) + pairDeltaOrbToEvs;
            pair.appliedRgbFrameOffset = appliedRgbFrameOffset_;
            pair.valid       = true;

            if (pair.seqNum < 12) {
                Log::info("Sync", "Pair produced #" + std::to_string(pair.seqNum)
                          + " color_idx=" + std::to_string(pair.orbbec.colorFrameIndex)
                          + " depth_idx=" + std::to_string(pair.orbbec.depthFrameIndex)
                          + " raw_color_idx=" + std::to_string(pair.orbbec.rawColorFrameIndex)
                          + " raw_depth_idx=" + std::to_string(pair.orbbec.rawDepthFrameIndex)
                          + " color_ts=" + std::to_string(pair.orbbec.colorTimestampUs)
                          + " mapped_color_ts=" + std::to_string(pair.mappedColorTimestampUs)
                          + " event_trigger_seq=" + std::to_string(pair.events.triggerStartSeq)
                          + "->" + std::to_string(pair.events.triggerEndSeq)
                          + " event_start=" + std::to_string(pair.events.startTs)
                          + " event_end=" + std::to_string(pair.events.endTs)
                          + " events=" + std::to_string(pair.events.events.size())
                          + " best_idx=" + std::to_string(bestIdx)
                          + " applied_rgb_offset=" + std::to_string(pair.appliedRgbFrameOffset)
                          + " clock_diff=" + std::to_string(clockDiff) + " us");
            }

            pairCount_++;
            totalPairCount_++;

            CdFrameData cdTmp;
            if (prophesee_.getLatestFrame(cdTmp) && cdTmp.valid) {
                pair.cdFrame = std::move(cdTmp);
            }

            {
                std::lock_guard<std::mutex> lock(pairMutex_);
                pairFront_ = pair;
                newPairReady_.store(true);
            }

            {
                std::lock_guard<std::mutex> lock(pairQueueMutex_);
                if (pairQueue_.size() >= MAX_PAIR_QUEUE) {
                    pairQueue_.pop_front();
                    pairQueueDropCount_.fetch_add(1);
                }
                pairQueue_.push_back(pair);
            }

            {
                std::lock_guard<std::mutex> lock(cbMutex_);
                if (callback_) {
                    callback_(pair);
                }
            }
        }

        // 鈹€鈹€ 4. Stall guard 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        if (orbBuf_.size() > MAX_LEAD && evsBuf_.empty()) {
            orbBuf_.pop_front();
            orbDropCount_++;
        }
        if (evsBuf_.size() > MAX_LEAD && orbBuf_.empty()) {
            evsBuf_.pop_front();
            evsDropCount_++;
        }

        // 鈹€鈹€ 5. Periodic monitoring 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - lastMonitor).count();
        if (elapsed >= MONITOR_INTERVAL_SEC) {
            int64_t drift = deltaOrbToEvs_ - initialDelta_;
            double avgDiff = 0.0, maxDiff = 0.0;
            if (!diffSamples_.empty()) {
                avgDiff = static_cast<double>(
                    std::accumulate(diffSamples_.begin(), diffSamples_.end(), int64_t(0)))
                    / diffSamples_.size();
                maxDiff = static_cast<double>(
                    *std::max_element(diffSamples_.begin(), diffSamples_.end()));
                diffSamples_.clear();
            }

            double pairRate = pairCount_ / elapsed;

            // 鈹€鈹€ Compute FOVs (degrees) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
            // FOV = 2 * atan(sensor_half_size / focal_length) * 180/pi
            // Event camera intrinsics (hardcoded from calibration)
            constexpr double EVT_FX = 1697.0566;
            constexpr int    EVT_W  = 1280;
            double evtFovH = 2.0 * std::atan(EVT_W / (2.0 * EVT_FX)) * 180.0 / PI;

            // Orbbec RGB intrinsics (from camera API)
            auto orbIntr = orbbec_.colorIntrinsics();
            double orbFovH = 0;
            if (orbIntr.fx > 0 && orbIntr.width > 0) {
                orbFovH = 2.0 * std::atan(orbIntr.width / (2.0 * orbIntr.fx)) * 180.0 / PI;
            }

            // 鈹€鈹€ Elapsed time since system start 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
            auto totalElapsed = std::chrono::steady_clock::now() - startTime_;
            int totalSec = static_cast<int>(std::chrono::duration<double>(totalElapsed).count());
            int mins = totalSec / 60;
            int secs = totalSec % 60;
            std::string elapsedStr = std::to_string(mins) + "m " + std::to_string(secs) + "s";

            // 鈹€鈹€ Build consolidated status banner 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
            Log::LogBlock blk(Log::timestamp() + "  System Status");
            blk.section("Runtime");
            blk.kv("Uptime", elapsedStr);
            blk.sep();
            blk.section("Orbbec (RGB)");
            blk.kvf("FPS", orbbec_.fps(), 1);
            blk.kvf("FOV", orbFovH, 1, " deg");
            blk.section("Prophesee (Event)");
            blk.kvf("Event rate", prophesee_.eventRate() / 1e6, 2, " M ev/s");
            blk.kvf("FOV", evtFovH, 1, " deg");
            blk.kv("Trig accepted", prophesee_.trigAccepted());
            blk.kv("Trig rejected", prophesee_.trigRejected());
            blk.sep();
            blk.section("Sync");
            blk.kv("OrbBuf", orbBuf_.size());
            blk.kv("EvsBuf", evsBuf_.size());
            blk.kv("PairQueue", pairQueueSize());
            blk.kv("Drift", std::to_string(drift) + " us");
            blk.kvf("AvgClockDiff", avgDiff, 1, " us");
            blk.kvf("MaxClockDiff", maxDiff, 1, " us");
            blk.sep();
            blk.section("Drops");
            blk.kv("OrbDrop", orbDropCount_);
            blk.kv("EvsSkip", evsDropCount_);
            blk.kv("PairQueueDrop", pairQueueDropCount_.load());

            if (orbBuf_.size() > MAX_LEAD || evsBuf_.size() > MAX_LEAD) {
                blk.sep();
                blk.kv("IMBALANCE", "ORB:" + std::to_string(orbBuf_.size())
                       + " EVS:" + std::to_string(evsBuf_.size()));
            }

            Log::banner(blk.title(), blk.body());

            pairCount_ = 0;
            lastMonitor = now;
        }

        if (!hasData) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
}
