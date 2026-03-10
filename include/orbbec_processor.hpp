#pragma once

#include <iostream>
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <vector>
#include <deque>
#include <cstring>

// OrbbecSDK
#include "libobsensor/ObSensor.hpp"
#include "libobsensor/hpp/Error.hpp"

/**
 * @brief Struct to hold a single frame result from the Orbbec camera.
 *        Mirrors the Python OrbbecCamera.get_frames() return values.
 */
struct OrbbecFrameData {
    // Color image (BGR, 3-channel, row-major)
    std::vector<uint8_t> colorData;
    uint32_t colorWidth  = 0;
    uint32_t colorHeight = 0;

    // Depth image (uint16, row-major, unit: raw depth units)
    std::vector<uint16_t> depthData;
    uint32_t depthWidth  = 0;
    uint32_t depthHeight = 0;

    // Timestamps (microseconds, device clock)
    uint64_t colorTimestampUs = 0;
    uint64_t depthTimestampUs = 0;

    bool valid = false;
};

/**
 * @brief OrbbecProcessor – runs in a dedicated thread, continuously grabs
 *        frames from the Orbbec depth camera and makes the latest result
 *        available via a thread-safe accessor.
 *
 *        Configuration mirrors the Python OrbbecCamera class:
 *          - Depth + Color streams (default profiles)
 *          - Multi-device sync (SECONDARY mode, hardware trigger)
 *          - Software Align (D2C via OB_STREAM_COLOR)
 */
class OrbbecProcessor {
public:
    OrbbecProcessor();
    ~OrbbecProcessor();

    /// Start the processing thread.  Non-blocking.
    void start();

    /// Request the processing thread to stop and wait for it to join.
    void stop();

    /// Thread-safe: copy the latest frame data into @p out (peek, marks consumed).
    /// @return true if new data was available since the last call.
    bool getLatestFrame(OrbbecFrameData &out);

    /// Thread-safe: pop the oldest queued frame (FIFO).  Used by SyncProcessor.
    /// @return true if a frame was available.
    bool popFrame(OrbbecFrameData &out);

    /// Number of frames currently queued.
    size_t queueSize() const;

    /// Check whether the processing thread is running.
    bool isRunning() const { return running_.load(); }

private:
    /// The actual processing loop executed on the worker thread.
    void processingLoop();

    // ── OrbbecSDK objects ──────────────────────────────────────────────
    std::shared_ptr<ob::Pipeline>          pipeline_;
    std::shared_ptr<ob::Config>            config_;
    std::shared_ptr<ob::Device>            device_;
    std::unique_ptr<ob::Align>             alignFilter_;

    bool hasColor_ = false;

    // ── Color intrinsics (for FOV computation) ─────────────────────────
    struct Intrinsics {
        float fx = 0, fy = 0, cx = 0, cy = 0;
        uint32_t width = 0, height = 0;
    };
    Intrinsics colorIntrinsics_;

public:
    Intrinsics colorIntrinsics() const { return colorIntrinsics_; }
private:

    // ── Stats (readable from SyncProcessor monitor) ─────────────────────
    std::atomic<double> orbFps_{0.0};
public:
    double fps() const { return orbFps_.load(); }
private:

    // ── Threading ──────────────────────────────────────────────────────
    std::thread       workerThread_;
    std::atomic<bool> running_{false};
    std::atomic<bool> stopRequested_{false};

    // Bounded queue: the worker pushes, consumer pops / peeks.
    static constexpr size_t MAX_QUEUE = 60;   // ~2 s @ 30 fps
    std::deque<OrbbecFrameData> frameQueue_;
    mutable std::mutex          frameMutex_;
    std::atomic<bool>           newFrameReady_{false};

    bool firstFrameLogged_ = false;
};
