#include "data_recorder.hpp"
#include "logger.hpp"

#include <iostream>
#include <iomanip>
#include <sstream>
#include <chrono>
#include <random>
#include <algorithm>

// OpenCV for image writing
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core.hpp>

// HDF5 C++ API
#include <H5Cpp.h>

// HDF5 is NOT thread-safe by default.  All HDF5 calls must be serialised.
static std::mutex g_hdf5Mutex;

// ============================================================================
//  Construction / Destruction
// ============================================================================

DataRecorder::DataRecorder(const std::string &outputRoot,
                           int numHdf5Workers, int numImageWorkers)
    : outputRoot_(outputRoot)
    , numHdf5Workers_(numHdf5Workers)
    , numImageWorkers_(numImageWorkers)
{
}

DataRecorder::~DataRecorder() {
    stop();
}

// ============================================================================
//  Thread management
// ============================================================================

void DataRecorder::start() {
    if (running_.load()) return;

    // ── Create session directory ────────────────────────────────────────
    auto now = std::chrono::system_clock::now();
    auto tt  = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &tt);
#else
    localtime_r(&tt, &tm);
#endif

    std::ostringstream tss;
    tss << std::put_time(&tm, "%Y%m%d_%H%M%S");

    // 8-char random hex id (like Python uuid4()[:8])
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint32_t> dist(0, 0xFFFFFFFF);
    std::ostringstream idss;
    idss << std::hex << std::setfill('0') << std::setw(8) << dist(gen);

    std::string sessionName = "session_" + tss.str() + "_" + idss.str();
    sessionPath_ = outputRoot_ / sessionName;
    eventDir_    = sessionPath_ / "event";
    frameDir_    = sessionPath_ / "frame";

    std::filesystem::create_directories(eventDir_);
    std::filesystem::create_directories(frameDir_);

    Log::info("Recorder", "Session folder: " + sessionPath_.string());

    // ── Launch thread pools ─────────────────────────────────────────────
    stopRequested_.store(false);
    running_.store(true);
    recordIdx_.store(0);
    hdf5Drops_.store(0);
    imagDrops_.store(0);
    totalWritten_.store(0);
    lastDropWarnTime_ = std::chrono::steady_clock::now();

    hdf5Workers_.reserve(numHdf5Workers_);
    for (int i = 0; i < numHdf5Workers_; ++i) {
        hdf5Workers_.emplace_back(&DataRecorder::hdf5WorkerLoop, this);
    }

    imageWorkers_.reserve(numImageWorkers_);
    for (int i = 0; i < numImageWorkers_; ++i) {
        imageWorkers_.emplace_back(&DataRecorder::imageWorkerLoop, this);
    }

    Log::info("Recorder", "Started with " + std::to_string(numHdf5Workers_)
             + " HDF5 + " + std::to_string(numImageWorkers_) + " image writer threads.");
}

void DataRecorder::stop() {
    if (!running_.load()) return;

    {
        std::lock_guard<std::mutex> lk1(hdf5QueueMutex_);
        std::lock_guard<std::mutex> lk2(imageQueueMutex_);
        Log::info("Recorder", "Stopping - flushing " + std::to_string(hdf5Queue_.size())
             + " HDF5 + " + std::to_string(imageQueue_.size()) + " image items...");
    }

    stopRequested_.store(true);
    hdf5QueueCv_.notify_all();
    imageQueueCv_.notify_all();

    for (auto &w : hdf5Workers_) {
        if (w.joinable()) w.join();
    }
    hdf5Workers_.clear();

    for (auto &w : imageWorkers_) {
        if (w.joinable()) w.join();
    }
    imageWorkers_.clear();

    running_.store(false);

    Log::info("Recorder", "Stopped.  Written: " + std::to_string(totalWritten_.load())
             + "  HDF5-drops: " + std::to_string(hdf5Drops_.load())
             + "  Image-drops: " + std::to_string(imagDrops_.load()));
}

// ============================================================================
//  Drop warning (throttled: max 1 per second)
// ============================================================================

void DataRecorder::emitDropWarning(const char *pool, size_t queueDepth) {
    std::lock_guard<std::mutex> lk(dropWarnMutex_);
    auto now = std::chrono::steady_clock::now();
    if (std::chrono::duration_cast<std::chrono::seconds>(now - lastDropWarnTime_).count() >= 1) {
        lastDropWarnTime_ = now;
        Log::warn("Recorder", std::string(pool) + " queue full (depth="
                 + std::to_string(queueDepth) + "), dropping oldest.  HDF5-drops="
                 + std::to_string(hdf5Drops_.load()) + "  Img-drops="
                 + std::to_string(imagDrops_.load()));
    }
}

// ============================================================================
//  Enqueue – pushes to BOTH queues (HDF5 and image, independently)
// ============================================================================

void DataRecorder::enqueue(const SyncedPair &pair) {
    if (!running_.load()) return;

    WriteTask task;
    task.idx    = recordIdx_.fetch_add(1);
    task.orbbec = pair.orbbec;      // copy
    task.events = pair.events;      // copy

    // ── Push to HDF5 queue ──────────────────────────────────────────────
    {
        std::lock_guard<std::mutex> lk(hdf5QueueMutex_);
        if (hdf5Queue_.size() >= MAX_HDF5_QUEUE) {
            hdf5Queue_.pop_front();
            hdf5Drops_.fetch_add(1);
            emitDropWarning("HDF5", MAX_HDF5_QUEUE);
        }
        hdf5Queue_.push_back(task);   // copy (events data)
    }
    hdf5QueueCv_.notify_one();

    // ── Push to image queue (share the orbbec data, clear events to save memory) ─
    {
        WriteTask imgTask;
        imgTask.idx    = task.idx;
        imgTask.orbbec = std::move(task.orbbec);
        // imgTask.events left empty – not needed for images

        std::lock_guard<std::mutex> lk(imageQueueMutex_);
        if (imageQueue_.size() >= MAX_IMAGE_QUEUE) {
            imageQueue_.pop_front();
            imagDrops_.fetch_add(1);
            emitDropWarning("Image", MAX_IMAGE_QUEUE);
        }
        imageQueue_.push_back(std::move(imgTask));
    }
    imageQueueCv_.notify_one();
}

// ============================================================================
//  HDF5 worker loop
// ============================================================================

void DataRecorder::hdf5WorkerLoop() {
    while (true) {
        WriteTask task;
        {
            std::unique_lock<std::mutex> lk(hdf5QueueMutex_);
            hdf5QueueCv_.wait(lk, [this] {
                return !hdf5Queue_.empty() || stopRequested_.load();
            });

            if (hdf5Queue_.empty() && stopRequested_.load()) return;
            if (hdf5Queue_.empty()) continue;

            task = std::move(hdf5Queue_.front());
            hdf5Queue_.pop_front();
        }

        writeEventHdf5(task);
    }
}

// ============================================================================
//  Image worker loop
// ============================================================================

void DataRecorder::imageWorkerLoop() {
    while (true) {
        WriteTask task;
        {
            std::unique_lock<std::mutex> lk(imageQueueMutex_);
            imageQueueCv_.wait(lk, [this] {
                return !imageQueue_.empty() || stopRequested_.load();
            });

            if (imageQueue_.empty() && stopRequested_.load()) return;
            if (imageQueue_.empty()) continue;

            task = std::move(imageQueue_.front());
            imageQueue_.pop_front();
        }

        writeImages(task);
        totalWritten_.fetch_add(1);
    }
}

// ============================================================================
//  HDF5 event writing (adaptive compression)
// ============================================================================

void DataRecorder::writeEventHdf5(const WriteTask &task) {
    std::ostringstream oss;
    oss << std::setfill('0') << std::setw(6) << task.idx;
    std::string idxStr = oss.str();

    try {
        std::string h5Path = (eventDir_ / (idxStr + ".h5")).string();

        std::lock_guard<std::mutex> h5lock(g_hdf5Mutex);

        // Define the compound data type matching Python's structured array
        H5::CompType eventType(sizeof(uint16_t) * 2 + sizeof(int16_t) + sizeof(int64_t));
        size_t offset = 0;
        eventType.insertMember("x", offset, H5::PredType::NATIVE_UINT16);
        offset += sizeof(uint16_t);
        eventType.insertMember("y", offset, H5::PredType::NATIVE_UINT16);
        offset += sizeof(uint16_t);
        eventType.insertMember("p", offset, H5::PredType::NATIVE_INT16);
        offset += sizeof(int16_t);
        eventType.insertMember("t", offset, H5::PredType::NATIVE_INT64);

        // Memory type matching actual CdEvent struct (may have padding)
        H5::CompType memType(sizeof(CdEvent));
        memType.insertMember("x", offsetof(CdEvent, x), H5::PredType::NATIVE_UINT16);
        memType.insertMember("y", offsetof(CdEvent, y), H5::PredType::NATIVE_UINT16);
        memType.insertMember("p", offsetof(CdEvent, p), H5::PredType::NATIVE_INT16);
        memType.insertMember("t", offsetof(CdEvent, t), H5::PredType::NATIVE_INT64);

        hsize_t nEvents = task.events.events.size();
        hsize_t dims[1] = {nEvents};
        H5::DataSpace dataspace(1, dims);

        // ── Adaptive compression ────────────────────────────────────────
        //  ≤100K events → gzip level 4 (fast, good ratio)
        //  ≤500K events → gzip level 1 (minimal CPU cost)
        //  >500K events → NO compression (raw throughput priority)
        H5::DSetCreatPropList plist;
        if (nEvents > 0) {
            hsize_t chunkSize;
            if (nEvents <= 100000) {
                chunkSize = std::min(nEvents, hsize_t(8192));
                plist.setChunk(1, &chunkSize);
                plist.setDeflate(4);
            } else if (nEvents <= 500000) {
                chunkSize = std::min(nEvents, hsize_t(32768));
                plist.setChunk(1, &chunkSize);
                plist.setDeflate(1);
            } else {
                // No compression – just chunked for large writes
                chunkSize = std::min(nEvents, hsize_t(65536));
                plist.setChunk(1, &chunkSize);
                // no setDeflate → uncompressed
            }
        }

        H5::H5File file(h5Path, H5F_ACC_TRUNC);
        H5::DataSet dataset = file.createDataSet("events", eventType, dataspace, plist);

        if (nEvents > 0) {
            dataset.write(task.events.events.data(), memType);
        }

        // Attributes – stored on the FILE (matching Python: h5f.attrs[...])
        {
            H5::DataSpace attrSpace(H5S_SCALAR);
            auto attrType = H5::PredType::NATIVE_INT64;

            int64_t startTs = task.events.startTs;
            int64_t endTs   = task.events.endTs;

            H5::Attribute a1 = file.createAttribute("start_ts", attrType, attrSpace);
            a1.write(attrType, &startTs);

            H5::Attribute a2 = file.createAttribute("end_ts", attrType, attrSpace);
            a2.write(attrType, &endTs);
        }

        file.close();
    }
    catch (const H5::Exception &e) {
        Log::error("Recorder", "HDF5 error for " + idxStr + ": " + e.getDetailMsg());
    }
    catch (const std::exception &e) {
        Log::error("Recorder", "Event write error for " + idxStr + ": " + e.what());
    }
}

// ============================================================================
//  Image writing (RGB JPEG + Depth PNG) – no mutex needed
// ============================================================================

void DataRecorder::writeImages(const WriteTask &task) {
    std::ostringstream oss;
    oss << std::setfill('0') << std::setw(6) << task.idx;
    std::string idxStr = oss.str();

    // ── 1. Write RGB JPEG ──────────────────────────────────────────────
    try {
        if (!task.orbbec.colorData.empty() &&
            task.orbbec.colorWidth > 0 && task.orbbec.colorHeight > 0) {

            size_t rawBgrSize = static_cast<size_t>(task.orbbec.colorWidth)
                              * task.orbbec.colorHeight * 3;

            cv::Mat rgb;
            if (task.orbbec.colorData.size() < rawBgrSize) {
                // Compressed MJPG → decode first
                cv::Mat jpgBuf(1, static_cast<int>(task.orbbec.colorData.size()),
                               CV_8UC1,
                               const_cast<uint8_t *>(task.orbbec.colorData.data()));
                rgb = cv::imdecode(jpgBuf, cv::IMREAD_COLOR);
            } else {
                rgb = cv::Mat(task.orbbec.colorHeight, task.orbbec.colorWidth,
                              CV_8UC3,
                              const_cast<uint8_t *>(task.orbbec.colorData.data()));
            }

            if (!rgb.empty()) {
                std::string jpgPath = (frameDir_ / (idxStr + "_rgb.jpg")).string();
                std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 95};
                cv::imwrite(jpgPath, rgb, params);
            }
        }
    }
    catch (const std::exception &e) {
        Log::error("Recorder", "RGB write error for " + idxStr + ": " + e.what());
    }

    // ── 2. Write Depth PNG (16-bit) ────────────────────────────────────
    try {
        if (!task.orbbec.depthData.empty() &&
            task.orbbec.depthWidth > 0 && task.orbbec.depthHeight > 0) {

            cv::Mat depth(task.orbbec.depthHeight, task.orbbec.depthWidth,
                          CV_16UC1, const_cast<uint16_t *>(task.orbbec.depthData.data()));

            std::string pngPath = (frameDir_ / (idxStr + "_depth.png")).string();
            cv::imwrite(pngPath, depth);
        }
    }
    catch (const std::exception &e) {
        Log::error("Recorder", "Depth write error for " + idxStr + ": " + e.what());
    }
}
