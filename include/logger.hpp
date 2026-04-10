#pragma once

#include <iostream>
#include <sstream>
#include <iomanip>
#include <string>
#include <mutex>
#include <chrono>
#include <deque>
#include <vector>

/**
 * @brief Centralised, thread-safe logging utility.
 *
 *  Usage:
 *      Log::info("TAG", "message");
 *      Log::warn("TAG", "message");
 *      Log::error("TAG", "message");
 *      Log::banner("TITLE", body_string);   // boxed block
 *
 *  All output goes through a single mutex so multi-threaded log lines
 *  never interleave.
 */
namespace Log {

// ── Global mutex for serialised output ─────────────────────────────────
inline std::mutex &ioMutex() {
    static std::mutex mtx;
    return mtx;
}

// ── Timestamp helper ───────────────────────────────────────────────────
inline std::string timestamp() {
    auto now  = std::chrono::system_clock::now();
    auto tt   = std::chrono::system_clock::to_time_t(now);
    auto ms   = std::chrono::duration_cast<std::chrono::milliseconds>(
                    now.time_since_epoch()) % 1000;
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &tt);
#else
    localtime_r(&tt, &tm);
#endif

    std::ostringstream oss;
    oss << std::put_time(&tm, "%H:%M:%S")
        << '.' << std::setfill('0') << std::setw(3) << ms.count();
    return oss.str();
}

// ── Ring buffer for on-screen log panel ─────────────────────────────────
struct LogEntry {
    std::string text;
    int level; // 0=info, 1=warn, 2=error
};

inline std::mutex &ringMutex() {
    static std::mutex mtx;
    return mtx;
}

inline std::deque<LogEntry> &ringBuffer() {
    static std::deque<LogEntry> buf;
    return buf;
}

inline constexpr size_t RING_MAX = 500;

inline void pushToRing(const std::string &text, int level) {
    std::lock_guard<std::mutex> lk(ringMutex());
    ringBuffer().push_back({text, level});
    while (ringBuffer().size() > RING_MAX)
        ringBuffer().pop_front();
}

/// Get a snapshot of the current log ring buffer (thread-safe).
inline std::vector<LogEntry> getLogSnapshot() {
    std::lock_guard<std::mutex> lk(ringMutex());
    return {ringBuffer().begin(), ringBuffer().end()};
}

// ── Single-line log ────────────────────────────────────────────────────

inline void info(const std::string &tag, const std::string &msg) {
    std::lock_guard<std::mutex> lk(ioMutex());
    std::cout << timestamp() << "  [" << tag << "]  " << msg << std::endl;
}

inline void warn(const std::string &tag, const std::string &msg) {
    std::lock_guard<std::mutex> lk(ioMutex());
    std::cerr << timestamp() << "  [" << tag << "]  [WARN] " << msg << std::endl;
}

inline void error(const std::string &tag, const std::string &msg) {
    std::lock_guard<std::mutex> lk(ioMutex());
    std::cerr << timestamp() << "  [" << tag << "]  [ERROR] " << msg << std::endl;
}

// ── Boxed banner block ─────────────────────────────────────────────────
//
//  ╔══════════════════════════════════════════════════════╗
//  ║  TITLE                                              ║
//  ╠══════════════════════════════════════════════════════╣
//  ║  body line 1                                        ║
//  ║  body line 2                                        ║
//  ╚══════════════════════════════════════════════════════╝
//
inline void banner(const std::string &title, const std::string &body,
                   int width = 60) {
    // Split body into lines
    std::vector<std::string> lines;
    std::istringstream stream(body);
    std::string line;
    while (std::getline(stream, line)) {
        lines.push_back(line);
    }

    // Compute inner width (box border takes 4 chars: "║ " + " ║")
    int inner = width - 4;
    if (inner < 20) inner = 20;

    auto pad = [&](const std::string &s) -> std::string {
        std::string out = s;
        if (static_cast<int>(out.size()) > inner) out = out.substr(0, inner);
        out.append(inner - static_cast<int>(out.size()), ' ');
        return out;
    };

    auto hbar = [&](const char *left, const char *mid, const char *right) {
        std::string bar = left;
        for (int i = 0; i < inner + 2; ++i) bar += mid;
        bar += right;
        return bar;
    };

    std::ostringstream oss;
    oss << hbar("+", "-", "+") << "\n";
    oss << "| " << pad(title) << " |\n";
    oss << hbar("+", "-", "+") << "\n";
    for (auto &l : lines) {
        oss << "| " << pad(l) << " |\n";
    }
    oss << hbar("+", "-", "+");

    // Push pure-ASCII lines into ring buffer for GUI display
    // (OpenCV putText cannot render Unicode — filter to printable ASCII only)
    auto toAscii = [](const std::string &s) {
        std::string out;
        out.reserve(s.size());
        for (unsigned char c : s) {
            if (c >= 0x20 && c <= 0x7E) out += static_cast<char>(c);
        }
        return out;
    };
    pushToRing("--- " + toAscii(title) + " ---", 0);
    for (auto &l : lines) {
        // Strip leading whitespace for compact display
        std::string trimmed = l;
        size_t pos = trimmed.find_first_not_of(' ');
        if (pos != std::string::npos && pos > 0) trimmed = trimmed.substr(pos);
        if (!trimmed.empty())
            pushToRing("  " + toAscii(trimmed), 0);
    }

    std::lock_guard<std::mutex> lk(ioMutex());
    std::cout << "\n" << oss.str() << "\n" << std::endl;
}

// ── Compact key:value row builder ──────────────────────────────────────
//    Use LogBlock to accumulate rows, then flush as one banner.
//
//    LogBlock blk("Sync Monitor");
//    blk.kv("Pairs",   pairCount);
//    blk.kv("Rate",    fps, "/s");
//    blk.sep();
//    blk.kv("Drift",   drift, " µs");
//    Log::banner(blk.title(), blk.body());
//
class LogBlock {
public:
    explicit LogBlock(const std::string &title) : title_(title) {}

    // Key-value with string
    LogBlock &kv(const std::string &key, const std::string &val) {
        oss_ << "  " << std::left << std::setw(20) << key << " : " << val << "\n";
        return *this;
    }

    // Key-value with number
    template <typename T>
    LogBlock &kv(const std::string &key, T val, const std::string &unit = "") {
        oss_ << "  " << std::left << std::setw(20) << key << " : " << val << unit << "\n";
        return *this;
    }

    // Key-value with double, fixed precision
    LogBlock &kvf(const std::string &key, double val, int prec = 1,
                  const std::string &unit = "") {
        oss_ << "  " << std::left << std::setw(20) << key << " : "
             << std::fixed << std::setprecision(prec) << val << unit << "\n";
        return *this;
    }

    // Separator line
    LogBlock &sep() {
        oss_ << "  " << std::string(36, '-') << "\n";
        return *this;
    }

    // Section sub-header
    LogBlock &section(const std::string &name) {
        oss_ << "  [" << name << "]\n";
        return *this;
    }

    const std::string &title() const { return title_; }
    std::string body() const { return oss_.str(); }

private:
    std::string title_;
    std::ostringstream oss_;
};

}  // namespace Log
