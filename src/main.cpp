// @code-review: for now, the system doesn't support multiple activation, so do not try this
// when the hardware trigger stopped, the system doesn't do re-initialization automatically.
// Thus just restart the system.

// And generally, the system doesn't fail, with many cases included.
// For example: start the trigger first, and just start the system. (misalign might happen because the first frame problem)
// Or, start the system first, then start the trigger. (misalign might happen because the jittering of trigger electric signal)
// On my test, both situation can be handled well. But there might still be some corner cases that we haven't tested, 
// if encountered, please let me know.

#include <iostream>
#include <thread>
#include <chrono>
#include <csignal>
#include <atomic>
#include <memory>
#include <string>
#include <cstring>

// OpenCV for visualisation
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

// Processing threads
#include "orbbec_processor.hpp"
#include "prophesee_processor.hpp"
#include "sync_processor.hpp"
#include "data_recorder.hpp"
#include "logger.hpp"

// ── Global stop flag for graceful shutdown ──────────────────────────────────
static std::atomic<bool> g_stop{false}; // @code-review: control the shutdown flow better rather than a global flag
static void signalHandler(int) { g_stop.store(true); }

// ── Command-line helpers ────────────────────────────────────────────────
// @code-review: cmd identifiers functions
static bool hasFlag(int argc, char *argv[], const char *flag) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], flag) == 0) return true;
    }
    return false;
}

static std::string getFlagValue(int argc, char *argv[], const char *flag,
                                const std::string &defaultVal = "") {
    for (int i = 1; i < argc - 1; ++i) {
        if (std::strcmp(argv[i], flag) == 0) return argv[i + 1];
    }
    return defaultVal;
}

// Get first positional argument (non-flag, non-flag-value)
// @code-review: also an identifier function
static std::string getPositional(int argc, char *argv[]) {
    for (int i = 1; i < argc; ++i) {
        // Skip known flags and their values
        if (std::strcmp(argv[i], "--store") == 0) continue;
        if (std::strcmp(argv[i], "--no-display") == 0) continue;
        if (std::strcmp(argv[i], "--output") == 0 ||
            std::strcmp(argv[i], "-o") == 0) {
            ++i;  // skip the value after --output/-o
            continue;
        }
        // Not a flag → treat as positional (raw file path)
        return argv[i];
    }
    return "";
}

// ── Visualisation helpers ───────────────────────────────────────────────

/// Decode Orbbec color data (MJPG compressed) into a BGR cv::Mat.
static cv::Mat decodeColor(const OrbbecFrameData &ob) {
    if (ob.colorData.empty() || ob.colorWidth == 0) return {};
    size_t rawBgrSize = static_cast<size_t>(ob.colorWidth) * ob.colorHeight * 3;
    if (ob.colorData.size() < rawBgrSize) {
        // MJPG compressed → decode
        // @code-review: for now we use opencv for visualization, maybe the decoding can not be optimized here
        // @code-review: this will be called for every frame, a waste.
        cv::Mat buf(1, static_cast<int>(ob.colorData.size()), CV_8UC1,
                    const_cast<uint8_t *>(ob.colorData.data()));
        return cv::imdecode(buf, cv::IMREAD_COLOR);
    }
    // Raw BGR
    return cv::Mat(ob.colorHeight, ob.colorWidth, CV_8UC3,
                   const_cast<uint8_t *>(ob.colorData.data())).clone();
}

/// Convert Orbbec depth (uint16) to a JET-coloured BGR cv::Mat.
// @code-review: also an overhead just for visualization, maybe can be optimized, but the GPU transfer is also an overhead.
// @code-review: so leave it here.
static cv::Mat coloriseDepth(const OrbbecFrameData &ob) {
    if (ob.depthData.empty() || ob.depthWidth == 0) return {};
    cv::Mat raw(ob.depthHeight, ob.depthWidth, CV_16UC1,
                const_cast<uint16_t *>(ob.depthData.data()));
    // Normalise to 0-255 (clamp max ~5000 mm → 5 m viewing range)
    cv::Mat norm;
    double maxDepth = 5000.0;
    raw.convertTo(norm, CV_8UC1, 255.0 / maxDepth);
    cv::Mat coloured;
    cv::applyColorMap(norm, coloured, cv::COLORMAP_JET);
    // Make invalid (depth==0) pixels black
    for (int r = 0; r < raw.rows; ++r) {
        const uint16_t *dptr = raw.ptr<uint16_t>(r);
        cv::Vec3b *cptr = coloured.ptr<cv::Vec3b>(r);
        for (int c = 0; c < raw.cols; ++c) {
            if (dptr[c] == 0) cptr[c] = cv::Vec3b(0, 0, 0);
        }
    }
    return coloured;
}

/// Convert CdFrameData into a displayable cv::Mat.
// @code-review: generalizable code interface.
static cv::Mat cdFrameToMat(const CdFrameData &cd) {
    if (!cd.valid || cd.frameData.empty() || cd.width == 0) return {};
    // MetavisionSDK PeriodicFrameGeneration produces BGR 3-channel
    int channels = static_cast<int>(cd.frameData.size()) / (cd.width * cd.height);
    if (channels == 3) {
        return cv::Mat(cd.height, cd.width, CV_8UC3,
                       const_cast<uint8_t *>(cd.frameData.data())).clone();
    } else if (channels == 1) {
        return cv::Mat(cd.height, cd.width, CV_8UC1,
                       const_cast<uint8_t *>(cd.frameData.data())).clone();
    }
    return {};
}

/// Render a terminal-style log panel (black background, coloured text).
/// Uses very small monospace font to maximise the number of visible lines.
// @code-review: this is a simple implementation for demonstration, can be optimized and enhanced (e.g. support markdown-like formatting, or even HTML).
static cv::Mat renderLogPanel(int width, int height) {
    cv::Mat panel(height, width, CV_8UC3, cv::Scalar(15, 15, 15)); // near-black bg

    // Title bar
    cv::rectangle(panel, cv::Rect(0, 0, width, 22), cv::Scalar(40, 40, 40), cv::FILLED);
    cv::putText(panel, "LOG TERMINAL", cv::Point(8, 15),
                cv::FONT_HERSHEY_PLAIN, 0.95, cv::Scalar(0, 220, 180), 1, cv::LINE_AA);

    auto logs = Log::getLogSnapshot();

    // Font config: FONT_HERSHEY_PLAIN at scale 0.75 gives a compact monospace look
    const double fontScale = 0.75;
    const int    fontFace  = cv::FONT_HERSHEY_PLAIN;
    const int    lineH     = 13;    // pixels per line
    const int    startY    = 32;    // first line Y position
    const int    leftPad   = 4;
    const int    maxLines  = (height - startY - 2) / lineH;
    const int    maxChars  = (width - leftPad - 4) / 6;  // ~6px per char at this scale

    // Flatten log entries: long lines get word-wrapped
    struct DisplayLine {
        std::string text;
        int level;
    };
    std::vector<DisplayLine> displayLines;
    displayLines.reserve(logs.size() * 2);

    for (auto &entry : logs) {
        const std::string &t = entry.text;
        if (static_cast<int>(t.size()) <= maxChars) {
            displayLines.push_back({t, entry.level});
        } else {
            // Wrap long lines
            for (size_t pos = 0; pos < t.size(); pos += maxChars) {
                std::string chunk = t.substr(pos, maxChars);
                displayLines.push_back({(pos > 0 ? "  " : "") + chunk, entry.level});
            }
        }
    }

    // Show the most recent lines that fit
    int startIdx = 0;
    if (static_cast<int>(displayLines.size()) > maxLines)
        startIdx = static_cast<int>(displayLines.size()) - maxLines;

    for (int i = startIdx; i < static_cast<int>(displayLines.size()); ++i) {
        int y = startY + (i - startIdx) * lineH;
        // Colour by severity
        cv::Scalar col;
        switch (displayLines[i].level) {
            case 1:  col = cv::Scalar(0, 190, 255); break; // warn: orange
            case 2:  col = cv::Scalar(80, 80, 255);  break; // error: red
            default: col = cv::Scalar(180, 190, 180); break; // info: light green-grey
        }
        // Highlight section headers (lines starting with "---" title "---")
        if (displayLines[i].text.find("--- ") == 0) {
            col = cv::Scalar(255, 200, 100); // cyan-blue for headers
        }
        // Highlight separator lines (pure dashes inside body)
        else if (displayLines[i].text.find("------") != std::string::npos) {
            col = cv::Scalar(100, 100, 100); // dim grey
        }
        cv::putText(panel, displayLines[i].text, cv::Point(leftPad, y),
                    fontFace, fontScale, col, 1, cv::LINE_AA);
    }

    // Scrollbar indicator on right edge
    if (static_cast<int>(displayLines.size()) > maxLines) {
        int totalLines = static_cast<int>(displayLines.size());
        float viewFrac = static_cast<float>(maxLines) / totalLines;
        float posFrac  = static_cast<float>(startIdx) / totalLines;
        int barH = std::max(10, static_cast<int>(viewFrac * (height - startY)));
        int barY = startY + static_cast<int>(posFrac * (height - startY));
        cv::rectangle(panel, cv::Rect(width - 4, barY, 3, barH),
                      cv::Scalar(80, 80, 80), cv::FILLED);
    }

    return panel;
}

// main loop
int main(int argc, char *argv[]) {
    std::signal(SIGINT, signalHandler);

    // ── Parse arguments ─────────────────────────────────────────────────
    bool storeEnabled   = hasFlag(argc, argv, "--store");
    bool displayEnabled = !hasFlag(argc, argv, "--no-display");
    std::string outputRoot = getFlagValue(argc, argv, "--output",
                             getFlagValue(argc, argv, "-o", "./output"));
    std::string rawFile = getPositional(argc, argv);

    if (storeEnabled) {
        Log::info("Main", "Recording enabled.  Output root: " + outputRoot);
    }
    if (displayEnabled) {
        Log::info("Main", "Visualisation enabled.  Press 'q' or ESC to quit.");
    }

    // ════════════════════════════════════════════════════════════════════
    //  1.  Orbbec camera  –  runs in its own thread
    // ════════════════════════════════════════════════════════════════════
    OrbbecProcessor orbbec;
    orbbec.start();  // launches the processing thread

    // ════════════════════════════════════════════════════════════════════
    //  2.  Prophesee (event) camera – runs in its own thread
    // ════════════════════════════════════════════════════════════════════
    std::unique_ptr<PropheseeProcessor> prophesee;
    bool hasEventCam = false;

    try {
        if (!rawFile.empty()) {
            prophesee = std::make_unique<PropheseeProcessor>(33000, rawFile);
        } else {
            prophesee = std::make_unique<PropheseeProcessor>();
        }
        prophesee->start();  // launches the processing thread
        hasEventCam = true;
    }
    catch (const std::exception &e) {
        Log::error("Main", std::string("Prophesee camera not available: ") + e.what());
        Log::info("Main", "Running Orbbec-only mode. Press Ctrl+C to quit.");
    }

    // ════════════════════════════════════════════════════════════════════
    //  3.  Sync processor – pairs Orbbec frames with event slices
    // ════════════════════════════════════════════════════════════════════
    std::unique_ptr<SyncProcessor> sync;
    if (hasEventCam) {
        sync = std::make_unique<SyncProcessor>(orbbec, *prophesee);
        sync->start();
    }

    // ════════════════════════════════════════════════════════════════════
    //  3.5  Data recorder (--store)
    // ════════════════════════════════════════════════════════════════════
    std::unique_ptr<DataRecorder> recorder;
    if (storeEnabled && hasEventCam) {
        recorder = std::make_unique<DataRecorder>(outputRoot);
        recorder->start();
    }

    // ════════════════════════════════════════════════════════════════════
    //  4.  Main loop – consume synchronised pairs + visualisation
    // ════════════════════════════════════════════════════════════════════
    SyncedPair pair;
    uint64_t pairCount    = 0;
    uint64_t obOnlyCount  = 0;

    // Display target size – each panel size in the 2x2 grid
    const int PANEL_W = 640;
    const int PANEL_H = 360;
    const int CANVAS_W = PANEL_W * 2;  // total width  = 1280
    const int CANVAS_H = PANEL_H * 2;  // total height = 720

    // Create single combined window
    if (displayEnabled) {
        cv::namedWindow("Sensor Dashboard", cv::WINDOW_NORMAL);
        cv::resizeWindow("Sensor Dashboard", CANVAS_W, CANVAS_H);
    }

    // Persistent panel images (keep last valid frame when no new data)
    cv::Mat panelEvent(PANEL_H, PANEL_W, CV_8UC3, cv::Scalar(30, 30, 30));
    cv::Mat panelRGB  (PANEL_H, PANEL_W, CV_8UC3, cv::Scalar(30, 30, 30));
    cv::Mat panelDepth(PANEL_H, PANEL_W, CV_8UC3, cv::Scalar(30, 30, 30));

    // Draw placeholder text on initial panels
    cv::putText(panelEvent, "Event Camera - Waiting...", cv::Point(150, 180),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(100, 100, 100), 1);
    cv::putText(panelRGB,   "RGB Camera - Waiting...",   cv::Point(170, 180),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(100, 100, 100), 1);
    cv::putText(panelDepth, "Depth Camera - Waiting...", cv::Point(150, 180),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(100, 100, 100), 1);

    while (!g_stop.load()) {
        // If the event camera was opened and has stopped, exit the loop
        if (hasEventCam && !prophesee->isRunning()) break;

        bool gotFrame = false;   // track whether we got new data this iteration

        if (hasEventCam && sync) {
            // ── Synced mode: consume matched pairs ──────────────────────
            if (sync->getLatestPair(pair) && pair.valid) {
                gotFrame = true;
                pairCount++;

                // Record to disk if --store is active
                if (recorder) {
                    recorder->enqueue(pair);
                }

                // ── Update panel images ─────────────────────────────────
                if (displayEnabled) {
                    cv::Mat evFrame = cdFrameToMat(pair.cdFrame);
                    if (!evFrame.empty()) {
                        cv::resize(evFrame, panelEvent, cv::Size(PANEL_W, PANEL_H));
                    }

                    cv::Mat rgb = decodeColor(pair.orbbec);
                    if (!rgb.empty()) {
                        cv::resize(rgb, panelRGB, cv::Size(PANEL_W, PANEL_H));
                    }

                    cv::Mat depthJet = coloriseDepth(pair.orbbec);
                    if (!depthJet.empty()) {
                        cv::resize(depthJet, panelDepth, cv::Size(PANEL_W, PANEL_H));
                    }
                }
            }
        } else {
            // ── Orbbec-only mode ────────────────────────────────────────
            OrbbecFrameData obFrame;
            if (orbbec.getLatestFrame(obFrame) && obFrame.valid) {
                gotFrame = true;
                obOnlyCount++;

                if (displayEnabled) {
                    cv::Mat rgb = decodeColor(obFrame);
                    if (!rgb.empty()) {
                        cv::resize(rgb, panelRGB, cv::Size(PANEL_W, PANEL_H));
                    }

                    cv::Mat depthJet = coloriseDepth(obFrame);
                    if (!depthJet.empty()) {
                        cv::resize(depthJet, panelDepth, cv::Size(PANEL_W, PANEL_H));
                    }
                }
            }
        }

        // ── Compose the 2×2 dashboard and show ─────────────────────────
        if (displayEnabled) {
            // Render the log terminal panel (top-right)
            cv::Mat panelLog = renderLogPanel(PANEL_W, PANEL_H);

            // Add panel labels
            auto addLabel = [](cv::Mat &img, const std::string &label) {
                cv::rectangle(img, cv::Rect(0, 0, static_cast<int>(label.size()) * 13 + 16, 26),
                              cv::Scalar(0, 0, 0), cv::FILLED);
                cv::putText(img, label, cv::Point(8, 18),
                            cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
            };
            addLabel(panelEvent, "Event Accumulation");
            addLabel(panelRGB,   "RGB Image");
            addLabel(panelDepth, "Aligned Depth (JET)");

            // Assemble 2×2 canvas:
            //   top-left  = Event  |  top-right    = Log
            //   bot-left  = RGB    |  bot-right    = Depth
            cv::Mat canvas(CANVAS_H, CANVAS_W, CV_8UC3);
            panelEvent.copyTo(canvas(cv::Rect(0,       0,       PANEL_W, PANEL_H)));
            panelLog  .copyTo(canvas(cv::Rect(PANEL_W, 0,       PANEL_W, PANEL_H)));
            panelRGB  .copyTo(canvas(cv::Rect(0,       PANEL_H, PANEL_W, PANEL_H)));
            panelDepth.copyTo(canvas(cv::Rect(PANEL_W, PANEL_H, PANEL_W, PANEL_H)));

            // Draw grid lines
            cv::line(canvas, cv::Point(PANEL_W, 0), cv::Point(PANEL_W, CANVAS_H),
                     cv::Scalar(100, 100, 100), 2);
            cv::line(canvas, cv::Point(0, PANEL_H), cv::Point(CANVAS_W, PANEL_H),
                     cv::Scalar(100, 100, 100), 2);

            cv::imshow("Sensor Dashboard", canvas);

            int key = cv::waitKey(1);
            if (key == 'q' || key == 'Q' || key == 27 /*ESC*/) {
                Log::info("Main", "Quit key pressed.");
                g_stop.store(true);
                break;
            }
        } else if (!gotFrame) {
            // No display, no new data → sleep to avoid busy-spinning
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    // Destroy windows
    if (displayEnabled) {
        cv::destroyAllWindows();
    }

    // ════════════════════════════════════════════════════════════════════
    //  5.  Cleanup
    // ════════════════════════════════════════════════════════════════════
    if (recorder) recorder->stop();
    if (sync) sync->stop();
    if (hasEventCam) prophesee->stop();
    orbbec.stop();

    Log::info("Main", "Finished. Synced pairs: " + std::to_string(pairCount)
             + "  Orbbec-only frames: " + std::to_string(obOnlyCount));
    return 0;
}