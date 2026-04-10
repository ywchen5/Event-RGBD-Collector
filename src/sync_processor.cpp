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
//    2. Flush all but the LAST entry on each side — the most recent
//       data is the only data that can possibly be co-triggered.
//    3. Initialise deltaOrbToEvs_ from that pair.
//    4. The first few pairs after this may still have a residual
//       offset, but the EMA drift-tracking (alpha=0.1) converges
//       within ~10 frames.
//
bool SyncProcessor::bootstrapAlignment() {
    if (orbBuf_.size() < MIN_BOOTSTRAP_SAMPLES ||
        evsBuf_.size() < MIN_BOOTSTRAP_SAMPLES) {
        return false;   // not enough data yet — at least one side is still warming up
    }

    // ── Flush all but the newest on each side ───────────────────────────
    size_t orbDiscarded = 0;
    while (orbBuf_.size() > 1) {
        orbBuf_.pop_front();
        orbDiscarded++;
        orbDropCount_++;
    }

    size_t evsDiscarded = 0;
    while (evsBuf_.size() > 1) {
        evsBuf_.pop_front();
        evsDiscarded++;
        evsDropCount_++;
    }

    // ── Initialise delta from the newest pair ───────────────────────────
    int64_t orbTs = static_cast<int64_t>(orbBuf_.front().colorTimestampUs);
    int64_t evsTs = evsBuf_.front().startTs;
    deltaOrbToEvs_ = evsTs - orbTs;
    initialDelta_  = deltaOrbToEvs_;

    {
        Log::LogBlock blk("Bootstrap Alignment");
        blk.kv("Orb discarded", orbDiscarded);
        blk.kv("Evs discarded", evsDiscarded);
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
            int64_t mappedOrbTs = orbTs + deltaOrbToEvs_;

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
            blk.kv("Drift", std::to_string(drift) + " us");
            blk.kvf("AvgClockDiff", avgDiff, 1, " us");
            blk.kvf("MaxClockDiff", maxDiff, 1, " us");
            blk.sep();
            blk.section("Drops");
            blk.kv("OrbDrop", orbDropCount_);
            blk.kv("EvsSkip", evsDropCount_);

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
