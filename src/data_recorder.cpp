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
#include <fstream>

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
    pairDrops_.store(0);
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
             + "  Pair-drops: " + std::to_string(pairDrops_.load()));
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
                 + std::to_string(queueDepth) + "), dropping synced pair.  Pair-drops="
                 + std::to_string(pairDrops_.load()));
    }
}

// ============================================================================
//  Enqueue – pushes to BOTH queues (HDF5 and image, independently)
// ============================================================================

void DataRecorder::enqueue(const SyncedPair &pair) {
    if (!running_.load()) return;

    const uint64_t idx = recordIdx_.fetch_add(1);

    WriteTask hdf5Task;
    hdf5Task.idx = idx;
    hdf5Task.events = pair.events;      // copy full event payload
    hdf5Task.eventCount = pair.events.events.size();
    hdf5Task.eventStartTs = pair.events.startTs;
    hdf5Task.eventEndTs = pair.events.endTs;

    WriteTask imgTask;
    imgTask.idx    = idx;
    imgTask.orbbec = pair.orbbec;       // copy image payload
    imgTask.eventCount = hdf5Task.eventCount;
    imgTask.eventStartTs = hdf5Task.eventStartTs;
    imgTask.eventEndTs = hdf5Task.eventEndTs;
    imgTask.events.startTs = pair.events.startTs;
    imgTask.events.endTs = pair.events.endTs;
    imgTask.events.triggerStartSeq = pair.events.triggerStartSeq;
    imgTask.events.triggerEndSeq = pair.events.triggerEndSeq;
    imgTask.events.sliceSeq = pair.events.sliceSeq;
    imgTask.events.triggerStartHostReceiptUs = pair.events.triggerStartHostReceiptUs;
    imgTask.events.triggerEndHostReceiptUs = pair.events.triggerEndHostReceiptUs;
    imgTask.events.valid = pair.events.valid;
    imgTask.seqNum = pair.seqNum;
    imgTask.clockDiffUs = pair.clockDiffUs;
    imgTask.deltaOrbToEvsUs = pair.deltaOrbToEvsUs;
    imgTask.mappedColorTimestampUs = pair.mappedColorTimestampUs;
    imgTask.mappedDepthTimestampUs = pair.mappedDepthTimestampUs;
    imgTask.appliedRgbFrameOffset = pair.appliedRgbFrameOffset;
    // imgTask.events.events left empty: image/CSV metadata does not need it.

    const char *fullPool = nullptr;
    size_t fullDepth = 0;
    {
        std::scoped_lock lock(hdf5QueueMutex_, imageQueueMutex_);
        if (hdf5Queue_.size() >= MAX_HDF5_QUEUE) {
            fullPool = "HDF5";
            fullDepth = MAX_HDF5_QUEUE;
        } else if (imageQueue_.size() >= MAX_IMAGE_QUEUE) {
            fullPool = "Image";
            fullDepth = MAX_IMAGE_QUEUE;
        } else {
            hdf5Queue_.push_back(std::move(hdf5Task));
            imageQueue_.push_back(std::move(imgTask));
        }
    }

    if (fullPool) {
        pairDrops_.fetch_add(1);
        emitDropWarning(fullPool, fullDepth);
        return;
    }

    hdf5QueueCv_.notify_one();
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

    // (no local image flags needed in HDF5 writer)

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

    // local flags/paths for files written in this task
    bool wroteRgb = false;
    bool wroteDepth = false;
    std::string writtenRgbPath;
    std::string writtenDepthPath;

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
                wroteRgb = true;
                writtenRgbPath = jpgPath;
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
            wroteDepth = true;
            writtenDepthPath = pngPath;
        }
    }
    catch (const std::exception &e) {
        Log::error("Recorder", "Depth write error for " + idxStr + ": " + e.what());
    }

    // Append images.txt with the actual Orbbec timestamps for each image.
    // A still image has one SDK timestamp rather than an interval, so the
    // start/end fields are both set to that image's own timestamp.
    try {
        std::lock_guard<std::mutex> lk(imagesTxtMutex_);
        std::string listPath = (frameDir_ / "images.txt").string();
        std::ofstream ofs(listPath, std::ios::app);
        if (ofs) {
            if (wroteRgb) {
                uint64_t ts = task.orbbec.colorTimestampUs;
                ofs << std::filesystem::path(writtenRgbPath).filename().string()
                    << " " << ts << " " << ts << "\n";
            }
            if (wroteDepth) {
                uint64_t ts = task.orbbec.depthTimestampUs;
                ofs << std::filesystem::path(writtenDepthPath).filename().string()
                    << " " << ts << " " << ts << "\n";
            }
            ofs.close();
        } else {
            Log::error("Recorder", "Failed to open images.txt for append: " + listPath);
        }
    }
    catch (const std::exception &e) {
        Log::error("Recorder", "images.txt write error for " + idxStr + ": " + e.what());
    }

    // Append a machine-readable timestamp diagnostic file.  The mapped_* fields
    // are Orbbec hardware timestamps transformed into the event-camera timeline.
    try {
        std::lock_guard<std::mutex> lk(imagesTxtMutex_);
        std::filesystem::path csvPath = frameDir_ / "sync_timestamps.csv";
        const bool writeHeader = !std::filesystem::exists(csvPath);
        std::ofstream ofs(csvPath.string(), std::ios::app);
        if (ofs) {
            if (writeHeader) {
                ofs << "idx,seq_num,rgb_file,depth_file,"
                    << "color_idx,depth_idx,raw_color_idx,raw_depth_idx,orbbec_produced_seq,"
                    << "event_slice_seq,event_trigger_start_seq,event_trigger_end_seq,event_count,"
                    << "applied_rgb_frame_offset,"
                    << "orbbec_host_arrival_us,event_trigger_start_host_us,event_trigger_end_host_us,"
                    << "orbbec_host_minus_trigger_start_us,orbbec_host_minus_trigger_end_us,"
                    << "rgb_raw_ts_us,depth_raw_ts_us,"
                    << "rgb_mapped_event_ts_us,depth_mapped_event_ts_us,"
                    << "event_start_ts_us,event_end_ts_us,"
                    << "delta_orbbec_to_event_us,event_minus_rgb_mapped_us,"
                    << "depth_minus_rgb_raw_us,depth_minus_rgb_mapped_us\n";
            }
            const int64_t depthMinusRgbRaw =
                static_cast<int64_t>(task.orbbec.depthTimestampUs)
                - static_cast<int64_t>(task.orbbec.colorTimestampUs);
            const int64_t depthMinusRgbMapped =
                task.mappedDepthTimestampUs - task.mappedColorTimestampUs;
            const int64_t orbHostMinusTriggerStart =
                task.orbbec.hostArrivalTimestampUs - task.events.triggerStartHostReceiptUs;
            const int64_t orbHostMinusTriggerEnd =
                task.orbbec.hostArrivalTimestampUs - task.events.triggerEndHostReceiptUs;
            ofs << task.idx << ","
                << task.seqNum << ","
                << (wroteRgb ? std::filesystem::path(writtenRgbPath).filename().string() : "") << ","
                << (wroteDepth ? std::filesystem::path(writtenDepthPath).filename().string() : "") << ","
                << task.orbbec.colorFrameIndex << ","
                << task.orbbec.depthFrameIndex << ","
                << task.orbbec.rawColorFrameIndex << ","
                << task.orbbec.rawDepthFrameIndex << ","
                << task.orbbec.producedSeq << ","
                << task.events.sliceSeq << ","
                << task.events.triggerStartSeq << ","
                << task.events.triggerEndSeq << ","
                << task.eventCount << ","
                << task.appliedRgbFrameOffset << ","
                << task.orbbec.hostArrivalTimestampUs << ","
                << task.events.triggerStartHostReceiptUs << ","
                << task.events.triggerEndHostReceiptUs << ","
                << orbHostMinusTriggerStart << ","
                << orbHostMinusTriggerEnd << ","
                << task.orbbec.colorTimestampUs << ","
                << task.orbbec.depthTimestampUs << ","
                << task.mappedColorTimestampUs << ","
                << task.mappedDepthTimestampUs << ","
                << task.eventStartTs << ","
                << task.eventEndTs << ","
                << task.deltaOrbToEvsUs << ","
                << task.clockDiffUs << ","
                << depthMinusRgbRaw << ","
                << depthMinusRgbMapped << "\n";
        } else {
            Log::error("Recorder", "Failed to open sync_timestamps.csv for append: " + csvPath.string());
        }
    }
    catch (const std::exception &e) {
        Log::error("Recorder", "sync_timestamps.csv write error for " + idxStr + ": " + e.what());
    }
}
