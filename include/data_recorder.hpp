#pragma once

#include <string>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <deque>
#include <cstdint>
#include <chrono>
#include <filesystem>

#include "sync_processor.hpp"

/**
 * @brief DataRecorder – asynchronously writes synced pairs to disk using
 *        the same directory layout as VISUAL_FRAMEWORK:
 *
 *        output/session_YYYYMMDD_HHMMSS_<uuid8>/
 *            event/
 *                000000.h5        – HDF5 structured array  {x:u2, y:u2, p:i2, t:i8}
 *                000001.h5           with attrs start_ts, end_ts
 *                …
 *            frame/
 *                000000_rgb.jpg   – JPEG quality 95
 *                000000_depth.png – 16-bit PNG
 *                000001_rgb.jpg
 *                000001_depth.png
 *                …
 *
 *  Architecture (v2 – throughput-optimised):
 *    - Two separate thread pools: one for HDF5 (serialised via mutex,
 *      but multiple threads help pipeline-overlap with image work),
 *      and one for images (fully parallel, no mutex needed).
 *    - Adaptive compression: gzip-4 for ≤100 K events, gzip-1 for
 *      ≤500 K, uncompressed above that (CPU can't keep up at 30 fps).
 *    - Back-pressure monitoring: periodic stats log, throttled drop
 *      warnings (at most 1 per second).
 */
class DataRecorder {
public:
    /**
     * @param outputRoot      Root output directory (default: ./output).
     * @param numHdf5Workers  Number of HDF5 writer threads.
     * @param numImageWorkers Number of image writer threads.
     */
    explicit DataRecorder(const std::string &outputRoot = "./output",
                          int numHdf5Workers  = 4,
                          int numImageWorkers = 6);
    ~DataRecorder();

    /// Start the writer threads.
    void start();

    /// Signal stop and wait for all queued items to flush.
    void stop();

    /// Enqueue a synced pair for writing.  Thread-safe.
    void enqueue(const SyncedPair &pair);

    /// Session directory path (available after start()).
    std::string sessionPath() const { return sessionPath_.string(); }

private:
    struct WriteTask {
        uint64_t                idx;
        OrbbecFrameData         orbbec;
        EventSliceData          events;
        // For image writer: carry event slice timestamps (µs)
        uint64_t                evStartTs = 0;
        uint64_t                evEndTs   = 0;
    };

    // ── HDF5 writer pool (serialised internally via g_hdf5Mutex) ───────
    void hdf5WorkerLoop();
    void writeEventHdf5(const WriteTask &task);

    // ── Image writer pool (fully parallel) ─────────────────────────────
    void imageWorkerLoop();
    void writeImages(const WriteTask &task);

    // Session paths
    std::filesystem::path outputRoot_;
    std::filesystem::path sessionPath_;
    std::filesystem::path eventDir_;
    std::filesystem::path frameDir_;

    // HDF5 thread pool
    int numHdf5Workers_;
    std::vector<std::thread> hdf5Workers_;
    std::deque<WriteTask> hdf5Queue_;
    std::mutex hdf5QueueMutex_;
    std::condition_variable hdf5QueueCv_;

    // Image thread pool
    int numImageWorkers_;
    std::vector<std::thread> imageWorkers_;
    std::deque<WriteTask> imageQueue_;
    std::mutex imageQueueMutex_;
    std::condition_variable imageQueueCv_;

    std::atomic<bool> stopRequested_{false};
    std::atomic<bool> running_{false};

    // Record index (monotonically increasing)
    std::atomic<uint64_t> recordIdx_{0};

    // ── Back-pressure stats ────────────────────────────────────────────
    std::atomic<uint64_t> hdf5Drops_{0};
    std::atomic<uint64_t> imagDrops_{0};
    std::atomic<uint64_t> totalWritten_{0};
    std::chrono::steady_clock::time_point lastDropWarnTime_;
    std::mutex dropWarnMutex_;

    // Protect concurrent append to frame/images.txt
    std::mutex imagesTxtMutex_;

    // At most 1 drop-warning per second
    void emitDropWarning(const char *pool, size_t queueDepth);

    static constexpr size_t MAX_HDF5_QUEUE  = 200;
    static constexpr size_t MAX_IMAGE_QUEUE = 400;
};
