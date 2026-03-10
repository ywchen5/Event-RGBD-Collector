#pragma once

#include <iostream>
#include <thread>
#include <mutex>
#include <atomic>
#include <deque>
#include <vector>
#include <cstdint>
#include <functional>

#include "orbbec_processor.hpp"
#include "prophesee_processor.hpp"

/**
 * @brief A synchronised pair: one Orbbec frame matched with one event slice.
 *
 *        Because both sensors are driven by the *same* hardware trigger,
 *        the Nth Orbbec frameset and the Nth event slice are physically
 *        co-captured.  The SyncProcessor simply pairs them by arrival
 *        order, which is far more robust than timestamp-domain matching
 *        across two independent device clocks.
 */
struct SyncedPair {
    OrbbecFrameData  orbbec;
    EventSliceData   events;

    // Optional CD visualisation frame closest in time
    CdFrameData      cdFrame;

    // Sequence number (0-based) – monotonically increasing
    uint64_t         seqNum = 0;

    // Cross-clock diff (for monitoring only)
    //   = events.startTs − (orbbec.colorTimestampUs + deltaOrbToEvs)
    int64_t          clockDiffUs = 0;

    bool valid = false;
};

/**
 * @brief SyncProcessor – runs in a dedicated thread, continuously drains
 *        the OrbbecProcessor and PropheseeProcessor front-buffers and
 *        pairs them using a robust sequence-number strategy.
 *
 *  Design rationale
 *  ────────────────
 *  Both cameras share the same hardware trigger (Orbbec in SECONDARY
 *  mode, Prophesee trigger-in on MAIN channel).  At 30 fps every
 *  ~33.3 ms a trigger fires, producing:
 *
 *      • Exactly 1 Orbbec frameset  (depth + color)
 *      • Exactly 1 event slice      (events between two trigger edges)
 *
 *  They always arrive in the same order.  Therefore the simplest and
 *  most robust matching is: pair #N with #N.
 *
 *  A small ring-buffer on each side absorbs jitter in case one sensor
 *  delivers slightly faster than the other.  We keep a clock-offset
 *  estimate (delta_orb_to_evs) for *monitoring only* – it is never
 *  used for the actual matching decision.
 *
 *  Strategy hierarchy (in order of priority):
 *  ──────────────────────────────────────────
 *  1. **Sequence pairing** – always consume the oldest from each buffer.
 *  2. **Stall guard**      – if one buffer grows > MAX_LEAD while the
 *                            other is empty, the leading sensor is too
 *                            far ahead; drop its oldest frame to stay
 *                            bounded.
 *  3. **Clock monitoring** – track cross-clock diff for diagnostics.
 */
class SyncProcessor {
public:
    /**
     * @param orbbec     Reference to a running OrbbecProcessor.
     * @param prophesee  Reference to a running PropheseeProcessor.
     */
    SyncProcessor(OrbbecProcessor &orbbec, PropheseeProcessor &prophesee);
    ~SyncProcessor();

    /// Start the sync thread.  Non-blocking.
    void start();

    /// Request stop and join.
    void stop();

    /// Thread-safe: copy the latest synchronised pair.
    /// @return true if a new pair was available since the last call.
    bool getLatestPair(SyncedPair &out);

    /// Check running state.
    bool isRunning() const { return running_.load(); }

    /// Register a user callback that fires on every new pair (called from
    /// the sync thread – keep it short or copy the data and return).
    using PairCallback = std::function<void(const SyncedPair &)>;
    void setCallback(PairCallback cb);

private:
    void syncLoop();
 
    /// Bootstrap: find the first truly matching pair by timestamp
    /// nearest-neighbour, discard startup junk, and initialise delta.
    /// Returns true once alignment is established.
    bool bootstrapAlignment();

    // ── References to the two producers ────────────────────────────────
    OrbbecProcessor    &orbbec_;
    PropheseeProcessor &prophesee_;

    // ── Bootstrap state ────────────────────────────────────────────────
    //    Before aligned_==true we are in startup phase: accumulate data
    //    from both sides, then find the first real match.  This handles
    //    the case where one device starts much earlier than the other.
    bool     aligned_               = false;
    static constexpr size_t MIN_BOOTSTRAP_SAMPLES = 5;  // need ≥5 on each side

    // ── Internal FIFO buffers ──────────────────────────────────────────
    //    Drain the per-sensor front-buffers into these deques so that
    //    we never miss a frame even if the main thread is slow.
    static constexpr size_t MAX_BUFFER = 120;   // ~4 s @ 30 fps
    static constexpr size_t MAX_LEAD   = 10;    // stall-guard threshold
    std::deque<OrbbecFrameData> orbBuf_;
    std::deque<EventSliceData>  evsBuf_;

    // ── Clock-offset estimate ──────────────────────────────────────────
    //    deltaOrbToEvs_ maps Orbbec timestamps onto the event timeline.
    //    Initialised during bootstrap from a true timestamp match, then
    //    maintained every pair via EMA so clockDiff stays near zero.
    int64_t  deltaOrbToEvs_      = 0;     // µs
    int64_t  initialDelta_       = 0;
    static constexpr double DRIFT_ALPHA = 0.1;

    // ── Nearest-timestamp pairing ───────────────────────────────────────
    //    For each Orbbec frame, pick the event slice with the closest
    //    mapped timestamp.  Extra slices (from N:1 trigger ratio) are
    //    silently discarded.  No REALIGN / RE-BOOTSTRAP needed.
    static constexpr int64_t FRAME_INTERVAL_US = 33333;  // 30 fps

    // ── Sequence counter ───────────────────────────────────────────────
    uint64_t nextSeq_ = 0;

    // ── Front-buffer for the consumer ──────────────────────────────────
    SyncedPair        pairFront_;
    std::mutex        pairMutex_;
    std::atomic<bool> newPairReady_{false};

    // ── Optional user callback ─────────────────────────────────────────
    PairCallback      callback_;
    std::mutex        cbMutex_;

    // ── Threading ──────────────────────────────────────────────────────
    std::thread       workerThread_;
    std::atomic<bool> running_{false};
    std::atomic<bool> stopRequested_{false};

    // ── Diagnostics ────────────────────────────────────────────────────
    uint64_t pairCount_      = 0;
    uint64_t totalPairCount_ = 0;   // never reset
    uint64_t orbDropCount_   = 0;
    uint64_t evsDropCount_   = 0;
    std::vector<int64_t> diffSamples_;  // for periodic average / max
    std::chrono::steady_clock::time_point startTime_;
};
