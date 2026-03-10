#pragma once

#include <iostream>
#include <thread>
#include <mutex>
#include <atomic>
#include <vector>
#include <deque>
#include <cstdint>

// Metavision SDK
#include <metavision/sdk/driver/camera.h>
#include <metavision/sdk/core/algorithms/periodic_frame_generation_algorithm.h>
#include <metavision/hal/facilities/i_trigger_in.h>
#include <metavision/hal/facilities/i_ll_biases.h>
#include <metavision/hal/facilities/i_erc_module.h>

/**
 * @brief A single CD event (x, y, polarity, timestamp).
 *        Mirrors Metavision::EventCD but stored in a plain struct
 *        so the caller doesn't need to include the SDK headers.
 */
// @code-review: a single event representation
struct CdEvent {
    uint16_t x;
    uint16_t y;
    int16_t  p;      // 0 or 1
    int64_t  t;      // microseconds
};

/**
 * @brief A trigger-sliced batch of events – all events that occurred
 *        between two consecutive rising-edge trigger signals.
 *        Mirrors the Python slice_queue.put({'event_volume', 'start_ts', 'end_ts'}).
 */
struct EventSliceData {
    std::vector<CdEvent> events;
    int64_t startTs = 0;   // µs – timestamp of the slice-start trigger
    int64_t endTs   = 0;   // µs – timestamp of the slice-end trigger
    bool valid      = false;
};

/**
 * @brief A periodic CD frame produced by PeriodicFrameGenerationAlgorithm
 *        for visualisation purposes.
 */
struct CdFrameData {
    std::vector<uint8_t> frameData;   // BGR or mono pixel buffer
    uint32_t width  = 0;
    uint32_t height = 0;
    int64_t  ts     = 0;              // µs – frame timestamp
    bool valid      = false;
};

/**
 * @brief PropheseeProcessor – runs in a dedicated thread, continuously
 *        processes events from a Prophesee event camera and provides:
 *
 *        1. Trigger-sliced event batches (EventSliceData) – for downstream
 *           algorithms that need precise trigger-synchronised data.
 *        2. Periodic CD frames (CdFrameData) – for real-time visualisation.
 *
 *        Configuration mirrors the Python PropheseeCamera class:
 *          - Biases: diff_on=76, diff_off=20
 *          - ERC enabled at 20 M events/sec
 *          - External trigger on MAIN channel
 *          - PeriodicFrameGenerationAlgorithm (accumulation 33 000 µs, 30 fps)
 *          - Trigger-based event slicing between consecutive rising edges
 */
class PropheseeProcessor {
public:
    /**
     * @param accumulationTimeUs  Accumulation time for frame generation (default 33 000 µs ≈ 30 fps).
     * @param filePath            Optional RAW file path.  If empty, the first available live camera is used.
     */
    explicit PropheseeProcessor(int accumulationTimeUs = 33000,
                                const std::string &filePath = "");
    ~PropheseeProcessor();

    /// Start the processing thread.  Non-blocking.
    void start();

    /// Request the processing thread to stop and wait for it to join.
    void stop();

    // @code-review: the same logic for both frame and event slices.
    /// Thread-safe: copy the latest trigger-sliced event batch (peek).
    /// @return true if a new slice was available since the last call.
    bool getLatestSlice(EventSliceData &out);

    /// Thread-safe: pop the oldest queued slice (FIFO).  Used by SyncProcessor.
    /// @return true if a slice was available.
    bool popSlice(EventSliceData &out);

    /// Number of slices currently queued.
    size_t sliceQueueSize() const;

    /// Thread-safe: copy the latest CD visualisation frame.
    /// @return true if a new frame was available since the last call.
    bool getLatestFrame(CdFrameData &out);

    /// Check whether the processing thread is running.
    bool isRunning() const { return running_.load(); }

    /// Sensor geometry
    int sensorWidth()  const { return width_;  }
    int sensorHeight() const { return height_; }

private:
    /// Main loop running on the worker thread.
    void processingLoop();

    // ── Camera / SDK objects ───────────────────────────────────────────
    Metavision::Camera cam_;
    int width_  = 0;
    int height_ = 0;
    int accumulationTimeUs_;
    std::string filePath_;

    // ── Stats (readable from SyncProcessor monitor) ─────────────────────
    std::atomic<double> evsFps_{0.0};
    std::atomic<double> eventRate_{0.0};  // events per second
public:
    double fps() const { return evsFps_.load(); }
    double eventRate() const { return eventRate_.load(); }
private:

    // ── Threading ──────────────────────────────────────────────────────
    std::thread       workerThread_;
    std::atomic<bool> running_{false};
    std::atomic<bool> stopRequested_{false};

    // ── Trigger-sliced events (bounded queue) ──────────────────────────
    static constexpr size_t MAX_SLICE_QUEUE = 60;  // ~2 s @ 30 fps
    std::deque<EventSliceData> sliceQueue_;
    mutable std::mutex         sliceMutex_;
    std::atomic<bool>          newSliceReady_{false};

    // ── CD visualisation frame (front-buffer pattern) ──────────────────
    CdFrameData       frameFront_;
    std::mutex        frameMutex_;
    std::atomic<bool> newFrameReady_{false};

    // ── Trigger-slicing state (owned by the worker thread – no mutex) ──
    std::vector<CdEvent> eventBuffer_;
    int64_t              lastTriggerTs_ = 0;
    bool                 triggerActive_ = false;

    // ── Trigger diagnostics ────────────────────────────────────────────
    //    Min interval filter: reject any rising edge closer than half
    //    a frame period to the previous accepted trigger.  This filters
    //    out double-triggers from bounce, dual-edge, or other causes.
    static constexpr int64_t MIN_TRIGGER_INTERVAL_US = 20000;  // 20 ms (~50 Hz max)
    std::atomic<uint64_t> trigAccepted_{0};
    std::atomic<uint64_t> trigRejected_{0};
public:
    uint64_t trigAccepted() const { return trigAccepted_.load(); }
    uint64_t trigRejected() const { return trigRejected_.load(); }
private:
};
