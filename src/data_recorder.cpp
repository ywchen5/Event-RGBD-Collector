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
#include <opencv2/imgproc.hpp>

// HDF5 C++ API
#include <H5Cpp.h>
#include <fstream>

// HDF5 is NOT thread-safe by default.  All HDF5 calls must be serialised.
static std::mutex g_hdf5Mutex;

namespace {

void writeInt64Attr(H5::H5File &file,
                    const H5::DataType &attrType,
                    const H5::DataSpace &attrSpace,
                    const std::string &name,
                    int64_t value) {
    H5::Attribute attr = file.createAttribute(name, attrType, attrSpace);
    attr.write(attrType, &value);
}

void writeStringAttr(H5::H5File &file,
                     const H5::DataSpace &attrSpace,
                     const std::string &name,
                     const std::string &value) {
    H5::StrType attrType(H5::PredType::C_S1, H5T_VARIABLE);
    H5::Attribute attr = file.createAttribute(name, attrType, attrSpace);
    const char *cValue = value.c_str();
    attr.write(attrType, &cValue);
}

void appendMetadataHeader(std::ostream &os, const std::string &prefix) {
    os << prefix << "_metadata_size,"
       << prefix << "_metadata_hex,";
    os << prefix << "_meta_timestamp,"
       << prefix << "_meta_sensor_timestamp,"
       << prefix << "_meta_frame_number,"
       << prefix << "_meta_auto_exposure,"
       << prefix << "_meta_exposure,"
       << prefix << "_meta_gain,"
       << prefix << "_meta_actual_frame_rate,"
       << prefix << "_meta_frame_rate,"
       << prefix << "_meta_gpio_input_data,";
    for (uint32_t i = 0; i < static_cast<uint32_t>(OB_FRAME_METADATA_TYPE_COUNT); ++i) {
        os << prefix << "_meta_type_" << i << ",";
    }
}

void appendMetadataRow(std::ostream &os, const OrbbecMetadataSnapshot &meta) {
    os << meta.metadataSize << ","
       << meta.metadataHex << ","
       << meta.timestamp << ","
       << meta.sensorTimestamp << ","
       << meta.frameNumber << ","
       << meta.autoExposure << ","
       << meta.exposure << ","
       << meta.gain << ","
       << meta.actualFrameRate << ","
       << meta.frameRate << ","
       << meta.gpioInputData << ",";
    for (int64_t value : meta.values) {
        os << value << ",";
    }
}

} // namespace

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
    task.eventStartTs = pair.events.startTs;
    task.eventEndTs = pair.events.endTs;
    task.eventTriggerStartSeq = pair.events.triggerStartSeq;
    task.eventTriggerEndSeq = pair.events.triggerEndSeq;
    task.seqNum = pair.seqNum;
    task.clockDiffUs = pair.clockDiffUs;
    task.deltaOrbToEvsUs = pair.deltaOrbToEvsUs;
    task.mappedColorTimestampUs = pair.mappedColorTimestampUs;
    task.mappedDepthTimestampUs = pair.mappedDepthTimestampUs;
    task.rgbEventVisualOffsetUs = pair.rgbEventVisualOffsetUs;
    task.visualMappedColorTimestampUs = pair.visualMappedColorTimestampUs;
    task.visualClockDiffUs = pair.visualClockDiffUs;

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
        imgTask.eventStartTs = task.eventStartTs;
        imgTask.eventEndTs = task.eventEndTs;
        imgTask.eventTriggerStartSeq = task.eventTriggerStartSeq;
        imgTask.eventTriggerEndSeq = task.eventTriggerEndSeq;
        imgTask.seqNum = task.seqNum;
        imgTask.clockDiffUs = task.clockDiffUs;
        imgTask.deltaOrbToEvsUs = task.deltaOrbToEvsUs;
        imgTask.mappedColorTimestampUs = task.mappedColorTimestampUs;
        imgTask.mappedDepthTimestampUs = task.mappedDepthTimestampUs;
        imgTask.rgbEventVisualOffsetUs = task.rgbEventVisualOffsetUs;
        imgTask.visualMappedColorTimestampUs = task.visualMappedColorTimestampUs;
        imgTask.visualClockDiffUs = task.visualClockDiffUs;
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
            int64_t triggerStartSeq = static_cast<int64_t>(task.events.triggerStartSeq);
            int64_t triggerEndSeq = static_cast<int64_t>(task.events.triggerEndSeq);

            H5::Attribute a1 = file.createAttribute("start_ts", attrType, attrSpace);
            a1.write(attrType, &startTs);

            H5::Attribute a2 = file.createAttribute("end_ts", attrType, attrSpace);
            a2.write(attrType, &endTs);

            H5::Attribute aTrigStart = file.createAttribute("trigger_start_seq", attrType, attrSpace);
            aTrigStart.write(attrType, &triggerStartSeq);

            H5::Attribute aTrigEnd = file.createAttribute("trigger_end_seq", attrType, attrSpace);
            aTrigEnd.write(attrType, &triggerEndSeq);

            int64_t visualOffsetUs = task.rgbEventVisualOffsetUs;
            int64_t visualMappedColorTs = task.visualMappedColorTimestampUs;

            H5::Attribute a3 = file.createAttribute("rgb_event_visual_offset_us", attrType, attrSpace);
            a3.write(attrType, &visualOffsetUs);

            H5::Attribute a4 = file.createAttribute("rgb_visual_mapped_event_ts_us", attrType, attrSpace);
            a4.write(attrType, &visualMappedColorTs);

            int64_t colorFrameIndex = static_cast<int64_t>(task.orbbec.colorFrameIndex);
            int64_t depthFrameIndex = static_cast<int64_t>(task.orbbec.depthFrameIndex);
            int64_t rawColorFrameIndex = static_cast<int64_t>(task.orbbec.rawColorFrameIndex);
            int64_t rawDepthFrameIndex = static_cast<int64_t>(task.orbbec.rawDepthFrameIndex);

            H5::Attribute a5 = file.createAttribute("orbbec_color_frame_index", attrType, attrSpace);
            a5.write(attrType, &colorFrameIndex);

            H5::Attribute a6 = file.createAttribute("orbbec_depth_frame_index", attrType, attrSpace);
            a6.write(attrType, &depthFrameIndex);

            H5::Attribute a7 = file.createAttribute("orbbec_raw_color_frame_index", attrType, attrSpace);
            a7.write(attrType, &rawColorFrameIndex);

            H5::Attribute a8 = file.createAttribute("orbbec_raw_depth_frame_index", attrType, attrSpace);
            a8.write(attrType, &rawDepthFrameIndex);

            writeInt64Attr(file, attrType, attrSpace, "orbbec_color_global_ts_us",
                           static_cast<int64_t>(task.orbbec.colorGlobalTimestampUs));
            writeInt64Attr(file, attrType, attrSpace, "orbbec_depth_global_ts_us",
                           static_cast<int64_t>(task.orbbec.depthGlobalTimestampUs));
            writeInt64Attr(file, attrType, attrSpace, "orbbec_raw_color_global_ts_us",
                           static_cast<int64_t>(task.orbbec.rawColorGlobalTimestampUs));
            writeInt64Attr(file, attrType, attrSpace, "orbbec_raw_depth_global_ts_us",
                           static_cast<int64_t>(task.orbbec.rawDepthGlobalTimestampUs));
            writeInt64Attr(file, attrType, attrSpace, "orbbec_color_format",
                           static_cast<int64_t>(task.orbbec.colorFormat));
            writeInt64Attr(file, attrType, attrSpace, "orbbec_color_prop_auto_exposure",
                           task.orbbec.colorAutoExposure);
            writeInt64Attr(file, attrType, attrSpace, "orbbec_color_prop_exposure",
                           task.orbbec.colorExposure);
            writeInt64Attr(file, attrType, attrSpace, "orbbec_color_prop_gain",
                           task.orbbec.colorGain);
            writeInt64Attr(file, attrType, attrSpace, "orbbec_color_prop_auto_white_balance",
                           task.orbbec.colorAutoWhiteBalance);
            writeInt64Attr(file, attrType, attrSpace, "orbbec_color_prop_white_balance",
                           task.orbbec.colorWhiteBalance);
            writeInt64Attr(file, attrType, attrSpace, "orbbec_color_prop_auto_exposure_priority",
                           task.orbbec.colorAutoExposurePriority);
            writeInt64Attr(file, attrType, attrSpace, "orbbec_color_prop_power_line_frequency",
                           task.orbbec.colorPowerLineFrequency);

            writeInt64Attr(file, attrType, attrSpace, "orbbec_color_metadata_size",
                           static_cast<int64_t>(task.orbbec.colorMetadata.metadataSize));
            writeInt64Attr(file, attrType, attrSpace, "orbbec_depth_metadata_size",
                           static_cast<int64_t>(task.orbbec.depthMetadata.metadataSize));
            writeInt64Attr(file, attrType, attrSpace, "orbbec_raw_color_metadata_size",
                           static_cast<int64_t>(task.orbbec.rawColorMetadata.metadataSize));
            writeInt64Attr(file, attrType, attrSpace, "orbbec_raw_depth_metadata_size",
                           static_cast<int64_t>(task.orbbec.rawDepthMetadata.metadataSize));

            writeStringAttr(file, attrSpace, "orbbec_color_metadata_hex",
                            task.orbbec.colorMetadata.metadataHex);
            writeStringAttr(file, attrSpace, "orbbec_depth_metadata_hex",
                            task.orbbec.depthMetadata.metadataHex);
            writeStringAttr(file, attrSpace, "orbbec_raw_color_metadata_hex",
                            task.orbbec.rawColorMetadata.metadataHex);
            writeStringAttr(file, attrSpace, "orbbec_raw_depth_metadata_hex",
                            task.orbbec.rawDepthMetadata.metadataHex);

            writeInt64Attr(file, attrType, attrSpace, "orbbec_color_meta_timestamp",
                           task.orbbec.colorMetadata.timestamp);
            writeInt64Attr(file, attrType, attrSpace, "orbbec_color_meta_sensor_timestamp",
                           task.orbbec.colorMetadata.sensorTimestamp);
            writeInt64Attr(file, attrType, attrSpace, "orbbec_color_meta_frame_number",
                           task.orbbec.colorMetadata.frameNumber);
            writeInt64Attr(file, attrType, attrSpace, "orbbec_color_meta_exposure",
                           task.orbbec.colorMetadata.exposure);
            writeInt64Attr(file, attrType, attrSpace, "orbbec_color_meta_gpio_input_data",
                           task.orbbec.colorMetadata.gpioInputData);

            writeInt64Attr(file, attrType, attrSpace, "orbbec_depth_meta_timestamp",
                           task.orbbec.depthMetadata.timestamp);
            writeInt64Attr(file, attrType, attrSpace, "orbbec_depth_meta_sensor_timestamp",
                           task.orbbec.depthMetadata.sensorTimestamp);
            writeInt64Attr(file, attrType, attrSpace, "orbbec_depth_meta_frame_number",
                           task.orbbec.depthMetadata.frameNumber);
            writeInt64Attr(file, attrType, attrSpace, "orbbec_depth_meta_exposure",
                           task.orbbec.depthMetadata.exposure);
            writeInt64Attr(file, attrType, attrSpace, "orbbec_depth_meta_gpio_input_data",
                           task.orbbec.depthMetadata.gpioInputData);

            writeInt64Attr(file, attrType, attrSpace, "orbbec_raw_color_meta_timestamp",
                           task.orbbec.rawColorMetadata.timestamp);
            writeInt64Attr(file, attrType, attrSpace, "orbbec_raw_color_meta_sensor_timestamp",
                           task.orbbec.rawColorMetadata.sensorTimestamp);
            writeInt64Attr(file, attrType, attrSpace, "orbbec_raw_color_meta_frame_number",
                           task.orbbec.rawColorMetadata.frameNumber);
            writeInt64Attr(file, attrType, attrSpace, "orbbec_raw_color_meta_exposure",
                           task.orbbec.rawColorMetadata.exposure);
            writeInt64Attr(file, attrType, attrSpace, "orbbec_raw_color_meta_gpio_input_data",
                           task.orbbec.rawColorMetadata.gpioInputData);

            writeInt64Attr(file, attrType, attrSpace, "orbbec_raw_depth_meta_timestamp",
                           task.orbbec.rawDepthMetadata.timestamp);
            writeInt64Attr(file, attrType, attrSpace, "orbbec_raw_depth_meta_sensor_timestamp",
                           task.orbbec.rawDepthMetadata.sensorTimestamp);
            writeInt64Attr(file, attrType, attrSpace, "orbbec_raw_depth_meta_frame_number",
                           task.orbbec.rawDepthMetadata.frameNumber);
            writeInt64Attr(file, attrType, attrSpace, "orbbec_raw_depth_meta_exposure",
                           task.orbbec.rawDepthMetadata.exposure);
            writeInt64Attr(file, attrType, attrSpace, "orbbec_raw_depth_meta_gpio_input_data",
                           task.orbbec.rawDepthMetadata.gpioInputData);
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
            if (task.orbbec.colorFormat == OB_FORMAT_MJPG ||
                (task.orbbec.colorFormat == OB_FORMAT_UNKNOWN &&
                 task.orbbec.colorData.size() < rawBgrSize)) {
                // Compressed MJPG → decode first
                cv::Mat jpgBuf(1, static_cast<int>(task.orbbec.colorData.size()),
                               CV_8UC1,
                               const_cast<uint8_t *>(task.orbbec.colorData.data()));
                rgb = cv::imdecode(jpgBuf, cv::IMREAD_COLOR);
            } else if (task.orbbec.colorFormat == OB_FORMAT_RGB) {
                cv::Mat raw(task.orbbec.colorHeight, task.orbbec.colorWidth,
                            CV_8UC3,
                            const_cast<uint8_t *>(task.orbbec.colorData.data()));
                cv::cvtColor(raw, rgb, cv::COLOR_RGB2BGR);
            } else if (task.orbbec.colorFormat == OB_FORMAT_YUYV ||
                       task.orbbec.colorFormat == OB_FORMAT_YUY2) {
                cv::Mat raw(task.orbbec.colorHeight, task.orbbec.colorWidth,
                            CV_8UC2,
                            const_cast<uint8_t *>(task.orbbec.colorData.data()));
                cv::cvtColor(raw, rgb, cv::COLOR_YUV2BGR_YUY2);
            } else if (task.orbbec.colorFormat == OB_FORMAT_UYVY) {
                cv::Mat raw(task.orbbec.colorHeight, task.orbbec.colorWidth,
                            CV_8UC2,
                            const_cast<uint8_t *>(task.orbbec.colorData.data()));
                cv::cvtColor(raw, rgb, cv::COLOR_YUV2BGR_UYVY);
            } else if (task.orbbec.colorFormat == OB_FORMAT_NV12) {
                cv::Mat raw(task.orbbec.colorHeight + task.orbbec.colorHeight / 2,
                            task.orbbec.colorWidth, CV_8UC1,
                            const_cast<uint8_t *>(task.orbbec.colorData.data()));
                cv::cvtColor(raw, rgb, cv::COLOR_YUV2BGR_NV12);
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
                    << "rgb_raw_ts_us,depth_raw_ts_us,"
                    << "rgb_system_ts_us,depth_system_ts_us,"
                    << "rgb_global_ts_us,depth_global_ts_us,"
                    << "rgb_frame_index,depth_frame_index,"
                    << "raw_rgb_frame_index,raw_depth_frame_index,"
                    << "raw_rgb_ts_us,raw_depth_ts_us,"
                    << "raw_rgb_system_ts_us,raw_depth_system_ts_us,"
                    << "raw_rgb_global_ts_us,raw_depth_global_ts_us,"
                    << "rgb_color_format,"
                    << "rgb_prop_auto_exposure,rgb_prop_exposure,rgb_prop_gain,"
                    << "rgb_prop_auto_white_balance,rgb_prop_white_balance,"
                    << "rgb_prop_auto_exposure_priority,rgb_prop_power_line_frequency,";
                appendMetadataHeader(ofs, "rgb");
                appendMetadataHeader(ofs, "depth");
                appendMetadataHeader(ofs, "raw_rgb");
                appendMetadataHeader(ofs, "raw_depth");
                ofs
                    << "rgb_mapped_event_ts_us,depth_mapped_event_ts_us,"
                    << "rgb_event_visual_offset_us,rgb_visual_mapped_event_ts_us,"
                    << "event_start_ts_us,event_end_ts_us,event_trigger_start_seq,event_trigger_end_seq,"
                    << "delta_orbbec_to_event_us,event_minus_rgb_mapped_us,event_minus_rgb_visual_mapped_us,"
                    << "depth_minus_rgb_raw_us,depth_minus_rgb_mapped_us\n";
            }
            const int64_t depthMinusRgbRaw =
                static_cast<int64_t>(task.orbbec.depthTimestampUs)
                - static_cast<int64_t>(task.orbbec.colorTimestampUs);
            const int64_t depthMinusRgbMapped =
                task.mappedDepthTimestampUs - task.mappedColorTimestampUs;
            ofs << task.idx << ","
                << task.seqNum << ","
                << (wroteRgb ? std::filesystem::path(writtenRgbPath).filename().string() : "") << ","
                << (wroteDepth ? std::filesystem::path(writtenDepthPath).filename().string() : "") << ","
                << task.orbbec.colorTimestampUs << ","
                << task.orbbec.depthTimestampUs << ","
                << task.orbbec.colorSystemTimestampUs << ","
                << task.orbbec.depthSystemTimestampUs << ","
                << task.orbbec.colorGlobalTimestampUs << ","
                << task.orbbec.depthGlobalTimestampUs << ","
                << task.orbbec.colorFrameIndex << ","
                << task.orbbec.depthFrameIndex << ","
                << task.orbbec.rawColorFrameIndex << ","
                << task.orbbec.rawDepthFrameIndex << ","
                << task.orbbec.rawColorTimestampUs << ","
                << task.orbbec.rawDepthTimestampUs << ","
                << task.orbbec.rawColorSystemTimestampUs << ","
                << task.orbbec.rawDepthSystemTimestampUs << ","
                << task.orbbec.rawColorGlobalTimestampUs << ","
                << task.orbbec.rawDepthGlobalTimestampUs << ","
                << static_cast<int>(task.orbbec.colorFormat) << ","
                << task.orbbec.colorAutoExposure << ","
                << task.orbbec.colorExposure << ","
                << task.orbbec.colorGain << ","
                << task.orbbec.colorAutoWhiteBalance << ","
                << task.orbbec.colorWhiteBalance << ","
                << task.orbbec.colorAutoExposurePriority << ","
                << task.orbbec.colorPowerLineFrequency << ",";
            appendMetadataRow(ofs, task.orbbec.colorMetadata);
            appendMetadataRow(ofs, task.orbbec.depthMetadata);
            appendMetadataRow(ofs, task.orbbec.rawColorMetadata);
            appendMetadataRow(ofs, task.orbbec.rawDepthMetadata);
            ofs
                << task.mappedColorTimestampUs << ","
                << task.mappedDepthTimestampUs << ","
                << task.rgbEventVisualOffsetUs << ","
                << task.visualMappedColorTimestampUs << ","
                << task.eventStartTs << ","
                << task.eventEndTs << ","
                << task.eventTriggerStartSeq << ","
                << task.eventTriggerEndSeq << ","
                << task.deltaOrbToEvsUs << ","
                << task.clockDiffUs << ","
                << task.visualClockDiffUs << ","
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
