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
                             PropheseeProcessor &prophesee,
                             int64_t rgbEventVisualOffsetUs)
    : orbbec_(orbbec)
    , prophesee_(prophesee)
    , rgbEventVisualOffsetUs_(rgbEventVisualOffsetUs)
    , startTime_(std::chrono::steady_clock::now())
{
    Log::info("Sync", "Initialised. RGB-event visual offset: "
              + std::to_string(rgbEventVisualOffsetUs_) + " us.");
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
//  completely different domains — e.g. Orbbec ~45 billion us vs
//  Prophesee ~16 million us).
//
//  Strategy:
//    1. Wait until both buffers have ≥ MIN_BOOTSTRAP_SAMPLES entries,
//       which proves both devices are actively streaming.
//    2. Align the common startup window by discarding old front samples
//       only.  If one side already has newer surplus samples, keep them
//       queued for future frames instead of throwing them away.
//    3. Move the LAST entry in the common startup window to the front.
//    4. Initialise deltaOrbToEvs_ from that pair.
//    5. The first few pairs after this may still have a residual
//       offset, but the EMA drift-tracking (alpha=0.1) converges
//       within ~10 frames.
//
bool SyncProcessor::bootstrapAlignment() {
    size_t requiredEvsSamples = MIN_BOOTSTRAP_SAMPLES;
    if (rgbEventVisualOffsetUs_ > 0) {
        requiredEvsSamples += static_cast<size_t>(rgbEventVisualOffsetUs_ / FRAME_INTERVAL_US) + 2;
    }

    if (orbBuf_.size() < MIN_BOOTSTRAP_SAMPLES ||
        evsBuf_.size() < requiredEvsSamples) {
        return false;   // not enough data yet — at least one side is still warming up
    }

    const size_t orbInitialSize = orbBuf_.size();
    const size_t evsInitialSize = evsBuf_.size();

    size_t orbDiscarded = 0;
    size_t evsDiscarded = 0;
    const size_t commonCount = std::min(orbBuf_.size(), evsBuf_.size());

    const size_t visualOffsetFrames =
        rgbEventVisualOffsetUs_ > 0
            ? static_cast<size_t>((rgbEventVisualOffsetUs_ + FRAME_INTERVAL_US - 1) / FRAME_INTERVAL_US)
            : 0;
    const size_t evsBootstrapKeep = std::min(commonCount, visualOffsetFrames + 1);

    // Move the common-window tail to the Orbbec front.  Any newer Orbbec
    // samples after that tail remain queued for future event slices.
    for (size_t i = 0; i + 1 < commonCount; ++i) {
        orbBuf_.pop_front();
        orbDiscarded++;
        orbDropCount_++;
    }

    // Keep enough earlier event slices for a positive visual offset, plus
    // any newer event surplus after the common-window tail.  For example,
    // with common tail E5 and --rgb-event-offset-frames 2, keep E3,E4,E5
    // at the front and also retain E6,E7 if they already arrived.
    const size_t evsFrontDiscard = commonCount - evsBootstrapKeep;
    for (size_t i = 0; i < evsFrontDiscard; ++i) {
        evsBuf_.pop_front();
        evsDiscarded++;
        evsDropCount_++;
    }

    // Initialise delta from the trigger-sequence-aligned event, not from
    // an earlier visual-offset candidate at the front.
    int64_t orbTs = static_cast<int64_t>(orbBuf_.front().colorTimestampUs);
    const size_t evsAnchorIdx = evsBootstrapKeep - 1;
    int64_t evsTs = evsBuf_[evsAnchorIdx].startTs;
    deltaOrbToEvs_ = evsTs - orbTs;
    initialDelta_  = deltaOrbToEvs_;

    {
        Log::LogBlock blk("Bootstrap Alignment");
        blk.kv("Orb initial", orbInitialSize);
        blk.kv("Evs initial", evsInitialSize);
        blk.kv("Common count", commonCount);
        blk.kv("Orb discarded", orbDiscarded);
        blk.kv("Evs discarded", evsDiscarded);
        blk.kv("Visual offset frames", visualOffsetFrames);
        blk.kv("Evs kept", evsBuf_.size());
        blk.kv("Orb selected frame", orbBuf_.front().colorFrameIndex);
        blk.kv("Evs anchor seq", std::to_string(evsBuf_[evsAnchorIdx].triggerStartSeq)
               + "->" + std::to_string(evsBuf_[evsAnchorIdx].triggerEndSeq));
        blk.kv("Evs first kept seq", std::to_string(evsBuf_.front().triggerStartSeq)
               + "->" + std::to_string(evsBuf_.front().triggerEndSeq));
        blk.kv("Evs last kept seq", std::to_string(evsBuf_.back().triggerStartSeq)
               + "->" + std::to_string(evsBuf_.back().triggerEndSeq));
        blk.kv("OrbTs", std::to_string(orbTs) + " us");
        blk.kv("EvsTs", std::to_string(evsTs) + " us");
        blk.kv("Delta (evs-orb)", std::to_string(deltaOrbToEvs_) + " us");
        Log::banner(blk.title(), blk.body());
    }

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

        // ── 1. Drain Orbbec queue into local deque ──────────────────
        {
            OrbbecFrameData tmp;
            while (orbbec_.popFrame(tmp)) {
                if (tmp.valid) {
                    if (orbBuf_.size() >= MAX_BUFFER) {
                        orbBuf_.pop_front();
                        orbDropCount_++;
                    }
                    orbBuf_.push_back(std::move(tmp));
                    hasData = true;
                }
            }
        }

        // ── 2. Drain Prophesee queue into local deque ───────────────
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

        // ── 2.5  Bootstrap: wait for enough data, then align ────────
        if (!aligned_) {
            if (!bootstrapAlignment()) {
                // Not ready yet – log status occasionally
                auto now = std::chrono::steady_clock::now();
                double elapsed = std::chrono::duration<double>(now - lastMonitor).count();
                if (elapsed >= MONITOR_INTERVAL_SEC) {
                    Log::info("Sync", "Waiting for bootstrap…  OrbBuf:" + std::to_string(orbBuf_.size())
                             + "  EvsBuf:" + std::to_string(evsBuf_.size()));
                    lastMonitor = now;
                }

                if (!hasData) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
                continue;   // skip pairing until aligned
            }
            // Bootstrap just succeeded – fall through to pairing
        }

        // ── 3. Nearest-timestamp pairing ────────────────────────────
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
            int64_t visualMappedOrbTs = mappedOrbTs - rgbEventVisualOffsetUs_;

            // --- Find closest event slice ------------------------------------
            size_t  bestIdx  = 0;
            int64_t bestDiff = std::abs(evsBuf_[0].startTs - visualMappedOrbTs);

            for (size_t i = 1; i < evsBuf_.size(); ++i) {
                int64_t diff = std::abs(evsBuf_[i].startTs - visualMappedOrbTs);
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
                evsBuf_[bestIdx].startTs > visualMappedOrbTs + FRAME_INTERVAL_US) {
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
            int64_t visualClockDiff = evsTs - visualMappedOrbTs;

            // EMA drift tracking
            deltaOrbToEvs_ += static_cast<int64_t>(visualClockDiff * DRIFT_ALPHA);
            diffSamples_.push_back(std::abs(visualClockDiff));

            // Build synchronised pair
            SyncedPair pair;
            pair.orbbec      = std::move(orbData);
            pair.events      = std::move(evsData);
            pair.seqNum      = nextSeq_++;
            pair.clockDiffUs = clockDiff;
            pair.deltaOrbToEvsUs = pairDeltaOrbToEvs;
            pair.mappedColorTimestampUs = static_cast<int64_t>(pair.orbbec.colorTimestampUs) + pairDeltaOrbToEvs;
            pair.mappedDepthTimestampUs = static_cast<int64_t>(pair.orbbec.depthTimestampUs) + pairDeltaOrbToEvs;
            pair.rgbEventVisualOffsetUs = rgbEventVisualOffsetUs_;
            pair.visualMappedColorTimestampUs = pair.mappedColorTimestampUs - rgbEventVisualOffsetUs_;
            pair.visualClockDiffUs = visualClockDiff;
            pair.valid       = true;

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

        // ── 4. Stall guard ─────────────────────────────────────────────
        if (orbBuf_.size() > MAX_LEAD && evsBuf_.empty()) {
            orbBuf_.pop_front();
            orbDropCount_++;
        }
        if (evsBuf_.size() > MAX_LEAD && orbBuf_.empty()) {
            evsBuf_.pop_front();
            evsDropCount_++;
        }

        // ── 5. Periodic monitoring ─────────────────────────────────────
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

            // ── Compute FOVs (degrees) ──────────────────────────────────
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

            // ── Elapsed time since system start ─────────────────────────
            auto totalElapsed = std::chrono::steady_clock::now() - startTime_;
            int totalSec = static_cast<int>(std::chrono::duration<double>(totalElapsed).count());
            int mins = totalSec / 60;
            int secs = totalSec % 60;
            std::string elapsedStr = std::to_string(mins) + "m " + std::to_string(secs) + "s";

            // ── Build consolidated status banner ────────────────────────
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
