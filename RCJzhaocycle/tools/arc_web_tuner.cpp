#include "arc_detector.hpp"
#include "frame_remapper.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <signal.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cctype>
#include <csignal>
#include <cstddef>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct TuneState {
    int scale_percent = 50;
    int max_width = 400;
    int max_height = 300;
    int use_hough = 0;
    int use_ransac = 1;
    int use_mask = 0;
    int min_radius = 45;
    int max_radius = 430;
    int ring_band = 10;
    int min_arc_bins = 8;
    int dark_value_max = 118;
    int dark_offset = 34;
    int min_confidence = 18;
    int green_hue_min = 32;
    int green_hue_max = 96;
    int green_sat_min = 28;
};

struct SharedState {
    std::mutex mutex;
    TuneState tune;
    cv::Mat latest_frame;
    std::vector<unsigned char> preview_jpeg;
    std::vector<unsigned char> binary_jpeg;
    std::string status_json = "{}";
    fs::path params_dir = "params";
    fs::path snapshot_dir = "pic_web";
    bool remap_enabled = false;
    std::string remap_path;
    bool mask_enabled = false;
    std::string mask_path;
    std::atomic<bool> running{true};
};

std::atomic<bool> g_running{true};

void onSignal(int) {
    g_running = false;
}

rcj::ArcDetectorConfig makeConfig(const TuneState& state) {
    rcj::ArcDetectorConfig config;
    config.processing_scale = std::max(10, state.scale_percent) / 100.0F;
    config.processing_max_width = std::max(64, state.max_width);
    config.processing_max_height = std::max(48, state.max_height);
    config.use_hough = state.use_hough != 0;
    config.use_ransac_candidates = state.use_ransac != 0;
    config.return_binary_roi = true;
    config.min_radius = std::max(1, state.min_radius);
    config.max_radius = std::max(config.min_radius + 1, state.max_radius);
    config.ring_band = std::max(1, state.ring_band);
    config.min_arc_bins = std::max(1, state.min_arc_bins);
    config.dark_value_max = std::max(1, state.dark_value_max);
    config.dark_adaptive_offset = state.dark_offset;
    config.min_confidence = std::max(0, state.min_confidence) / 100.0F;
    config.green_hue_min = std::min(state.green_hue_min, state.green_hue_max);
    config.green_hue_max = std::max(state.green_hue_min, state.green_hue_max);
    config.green_sat_min = std::max(0, state.green_sat_min);
    return config;
}

std::string nowMinute() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
    localtime_r(&t, &tm);
    std::ostringstream out;
    out << std::put_time(&tm, "%Y-%m-%d %H:%M");
    return out.str();
}

std::string fileMinute() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
    localtime_r(&t, &tm);
    std::ostringstream out;
    out << std::put_time(&tm, "%Y%m%d_%H%M");
    return out.str();
}

std::string fileTimestamp() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    const auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() % 1000;
    std::tm tm{};
    localtime_r(&t, &tm);
    std::ostringstream out;
    out << std::put_time(&tm, "%Y%m%d_%H%M%S_") << std::setw(3) << std::setfill('0') << millis;
    return out.str();
}

std::string paramsJson(const TuneState& state, const std::string& timestamp) {
    std::ostringstream out;
    out << "{\n"
        << "  \"timestamp_minute\": \"" << timestamp << "\",\n"
        << "  \"scale_percent\": " << state.scale_percent << ",\n"
        << "  \"max_width\": " << state.max_width << ",\n"
        << "  \"max_height\": " << state.max_height << ",\n"
        << "  \"use_hough\": " << state.use_hough << ",\n"
        << "  \"use_ransac\": " << state.use_ransac << ",\n"
        << "  \"use_mask\": " << state.use_mask << ",\n"
        << "  \"min_radius\": " << state.min_radius << ",\n"
        << "  \"max_radius\": " << state.max_radius << ",\n"
        << "  \"ring_band\": " << state.ring_band << ",\n"
        << "  \"min_arc_bins\": " << state.min_arc_bins << ",\n"
        << "  \"dark_value_max\": " << state.dark_value_max << ",\n"
        << "  \"dark_offset\": " << state.dark_offset << ",\n"
        << "  \"min_confidence\": " << state.min_confidence << ",\n"
        << "  \"green_hue_min\": " << state.green_hue_min << ",\n"
        << "  \"green_hue_max\": " << state.green_hue_max << ",\n"
        << "  \"green_sat_min\": " << state.green_sat_min << "\n"
        << "}\n";
    return out.str();
}

void saveParams(const TuneState& state, const fs::path& params_dir) {
    fs::create_directories(params_dir);
    const std::string minute = nowMinute();
    const std::string json = paramsJson(state, minute);

    {
        std::ofstream latest(params_dir / "latest_params.json", std::ios::trunc);
        latest << json;
    }
    {
        std::ofstream minute_file(params_dir / ("arc_params_" + fileMinute() + ".json"), std::ios::trunc);
        minute_file << json;
    }

    const fs::path history_path = params_dir / "arc_params_history.csv";
    const bool needs_header = !fs::exists(history_path);
    std::ofstream history(history_path, std::ios::app);
    if (needs_header) {
        history << "timestamp_minute,scale_percent,max_width,max_height,use_hough,use_ransac,use_mask,min_radius,max_radius,"
                << "ring_band,min_arc_bins,dark_value_max,dark_offset,min_confidence,green_hue_min,"
                << "green_hue_max,green_sat_min\n";
    }
    history << minute << ','
            << state.scale_percent << ','
            << state.max_width << ','
            << state.max_height << ','
            << state.use_hough << ','
            << state.use_ransac << ','
            << state.use_mask << ','
            << state.min_radius << ','
            << state.max_radius << ','
            << state.ring_band << ','
            << state.min_arc_bins << ','
            << state.dark_value_max << ','
            << state.dark_offset << ','
            << state.min_confidence << ','
            << state.green_hue_min << ','
            << state.green_hue_max << ','
            << state.green_sat_min << '\n';
}

bool parseIntField(const std::string& body, const std::string& key, int& target) {
    const std::string quoted = "\"" + key + "\"";
    std::size_t pos = body.find(quoted);
    if (pos == std::string::npos) {
        return false;
    }
    pos = body.find(':', pos + quoted.size());
    if (pos == std::string::npos) {
        return false;
    }
    ++pos;
    while (pos < body.size() && std::isspace(static_cast<unsigned char>(body[pos])) != 0) {
        ++pos;
    }
    std::size_t end = pos;
    if (end < body.size() && (body[end] == '-' || body[end] == '+')) {
        ++end;
    }
    while (end < body.size() && std::isdigit(static_cast<unsigned char>(body[end])) != 0) {
        ++end;
    }
    if (end == pos) {
        return false;
    }
    target = std::stoi(body.substr(pos, end - pos));
    return true;
}

void updateTuneFromJson(const std::string& body, TuneState& tune) {
    parseIntField(body, "scale_percent", tune.scale_percent);
    parseIntField(body, "max_width", tune.max_width);
    parseIntField(body, "max_height", tune.max_height);
    parseIntField(body, "use_hough", tune.use_hough);
    parseIntField(body, "use_ransac", tune.use_ransac);
    parseIntField(body, "use_mask", tune.use_mask);
    parseIntField(body, "min_radius", tune.min_radius);
    parseIntField(body, "max_radius", tune.max_radius);
    parseIntField(body, "ring_band", tune.ring_band);
    parseIntField(body, "min_arc_bins", tune.min_arc_bins);
    parseIntField(body, "dark_value_max", tune.dark_value_max);
    parseIntField(body, "dark_offset", tune.dark_offset);
    parseIntField(body, "min_confidence", tune.min_confidence);
    parseIntField(body, "green_hue_min", tune.green_hue_min);
    parseIntField(body, "green_hue_max", tune.green_hue_max);
    parseIntField(body, "green_sat_min", tune.green_sat_min);
}

std::string tuneJson(const TuneState& tune) {
    return paramsJson(tune, nowMinute());
}

void drawDetection(cv::Mat& frame, const rcj::ArcDetection& result, double ms, double fps) {
    if (result.found) {
        cv::circle(frame, result.center, static_cast<int>(std::round(result.radius)), cv::Scalar(0, 255, 255), 2);
        cv::circle(frame, result.center, 3, cv::Scalar(0, 0, 255), -1);
        cv::rectangle(frame, result.roi_rect, cv::Scalar(255, 0, 0), 2);
    }
    const std::string text = std::string(result.found ? "FOUND" : "MISS") +
                             " ms=" + std::to_string(ms).substr(0, 5) +
                             " fps=" + std::to_string(fps).substr(0, 5) +
                             " conf=" + std::to_string(result.confidence).substr(0, 5);
    cv::rectangle(frame, cv::Rect(0, 0, frame.cols, 34), cv::Scalar(0, 0, 0), -1);
    cv::putText(frame, text, cv::Point(8, 23), cv::FONT_HERSHEY_SIMPLEX, 0.62, cv::Scalar(255, 255, 255), 2);
}

std::string jsonString(const std::string& value) {
    std::ostringstream out;
    for (const char ch : value) {
        switch (ch) {
            case '\\':
                out << "\\\\";
                break;
            case '"':
                out << "\\\"";
                break;
            case '\n':
                out << "\\n";
                break;
            case '\r':
                out << "\\r";
                break;
            case '\t':
                out << "\\t";
                break;
            default:
                out << ch;
                break;
        }
    }
    return out.str();
}

std::string statusJson(
    const rcj::ArcDetection& result,
    double ms,
    double fps,
    bool remap_enabled,
    const std::string& remap_path,
    bool mask_enabled,
    const std::string& mask_path) {
    std::ostringstream out;
    out << "{"
        << "\"found\":" << (result.found ? "true" : "false") << ','
        << "\"center_x\":" << result.center.x << ','
        << "\"center_y\":" << result.center.y << ','
        << "\"radius\":" << result.radius << ','
        << "\"angle_start\":" << result.angle_start << ','
        << "\"angle_end\":" << result.angle_end << ','
        << "\"confidence\":" << result.confidence << ','
        << "\"ms\":" << ms << ','
        << "\"fps\":" << fps << ','
        << "\"remap_enabled\":" << (remap_enabled ? "true" : "false") << ','
        << "\"remap_path\":\"" << jsonString(remap_path) << "\","
        << "\"mask_enabled\":" << (mask_enabled ? "true" : "false") << ','
        << "\"mask_path\":\"" << jsonString(mask_path) << "\""
        << "}";
    return out.str();
}

std::string saveSnapshot(SharedState& state, std::string& error) {
    cv::Mat frame;
    fs::path snapshot_dir;
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        if (state.latest_frame.empty()) {
            error = "no camera frame is available yet";
            return {};
        }
        frame = state.latest_frame.clone();
        snapshot_dir = state.snapshot_dir;
    }

    std::error_code ec;
    fs::create_directories(snapshot_dir, ec);
    if (ec) {
        error = "failed to create snapshot directory: " + snapshot_dir.string();
        return {};
    }

    const fs::path path = snapshot_dir / ("web_" + fileTimestamp() + ".png");
    if (!cv::imwrite(path.string(), frame)) {
        error = "failed to write snapshot: " + path.string();
        return {};
    }
    return path.string();
}

bool loadKeepMask(const fs::path& path, cv::Mat& keep_mask, std::string& error) {
    cv::Mat loaded = cv::imread(path.string(), cv::IMREAD_GRAYSCALE);
    if (loaded.empty()) {
        error = "failed to read mask: " + path.string();
        return false;
    }
    cv::threshold(loaded, keep_mask, 127, 255, cv::THRESH_BINARY);
    return true;
}

bool applyKeepMask(cv::Mat& image, const cv::Mat& keep_mask, std::string& error) {
    if (keep_mask.empty()) {
        return true;
    }
    if (image.size() != keep_mask.size()) {
        std::ostringstream message;
        message << "mask size " << keep_mask.cols << 'x' << keep_mask.rows
                << " does not match frame size " << image.cols << 'x' << image.rows;
        error = message.str();
        return false;
    }
    if (image.channels() == 1) {
        image.setTo(cv::Scalar(180), keep_mask == 0);
    } else {
        image.setTo(cv::Scalar(0, 180, 0), keep_mask == 0);
    }
    return true;
}

bool sendAll(int fd, const char* data, std::size_t size) {
    while (size > 0) {
        const ssize_t sent = ::send(fd, data, size, MSG_NOSIGNAL);
        if (sent <= 0) {
            return false;
        }
        data += sent;
        size -= static_cast<std::size_t>(sent);
    }
    return true;
}

bool sendAll(int fd, const std::string& data) {
    return sendAll(fd, data.data(), data.size());
}

void sendResponse(int fd, const std::string& content_type, const std::string& body, int code = 200, const std::string& status = "OK") {
    std::ostringstream header;
    header << "HTTP/1.1 " << code << ' ' << status << "\r\n"
           << "Content-Type: " << content_type << "\r\n"
           << "Content-Length: " << body.size() << "\r\n"
           << "Connection: close\r\n"
           << "Cache-Control: no-store\r\n\r\n";
    sendAll(fd, header.str());
    sendAll(fd, body);
}

std::string htmlPage() {
    return R"HTML(<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RCJ Arc Web Tuner</title>
  <style>
    body { margin: 0; font-family: system-ui, sans-serif; background: #111; color: #eee; }
    main { display: grid; grid-template-columns: minmax(0, 1fr) 360px; gap: 16px; padding: 16px; }
    img { width: 100%; background: #000; border: 1px solid #333; }
    .panel { background: #1b1b1b; padding: 14px; border-radius: 6px; }
    .row { display: grid; grid-template-columns: 145px 1fr 52px; gap: 8px; align-items: center; margin: 9px 0; }
    input[type=range] { width: 100%; }
    code { color: #9ee; }
    button { width: 100%; padding: 9px; margin-top: 10px; }
    .hint { color: #bbb; font-size: 13px; line-height: 1.35; }
    @media (max-width: 900px) { main { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
<main>
  <section>
    <img src="/stream" alt="live stream">
    <img src="/binary" alt="binary roi" style="max-width: 420px; margin-top: 10px;">
  </section>
  <aside class="panel">
    <h2>Arc Parameters</h2>
    <div id="status">loading</div>
    <div id="controls"></div>
    <button onclick="saveNow()">Save now</button>
    <button onclick="snapshotNow()">Snapshot</button>
    <p class="hint">Press <code>s</code> to save the current corrected camera frame to <code>pic_web/</code>.</p>
    <div id="snapshotStatus" class="hint"></div>
    <p>Auto-save writes <code>params/latest_params.json</code>, minute JSON, and history CSV.</p>
  </aside>
</main>
<script>
const spec = [
  ["scale_percent", 10, 100, 1], ["max_width", 64, 800, 1], ["max_height", 48, 600, 1],
  ["use_hough", 0, 1, 1], ["use_ransac", 0, 1, 1], ["use_mask", 0, 1, 1], ["min_radius", 1, 800, 1], ["max_radius", 2, 900, 1],
  ["ring_band", 1, 80, 1], ["min_arc_bins", 1, 72, 1], ["dark_value_max", 1, 255, 1],
  ["dark_offset", -80, 160, 1], ["min_confidence", 0, 100, 1],
  ["green_hue_min", 0, 179, 1], ["green_hue_max", 0, 179, 1], ["green_sat_min", 0, 255, 1]
];
let params = {};
let timer = 0;
function renderControls() {
  const root = document.getElementById("controls");
  root.innerHTML = "";
  for (const [name, min, max, step] of spec) {
    const row = document.createElement("div");
    row.className = "row";
    row.innerHTML = `<label>${name}</label><input id="${name}" type="range" min="${min}" max="${max}" step="${step}" value="${params[name]}"><span id="${name}_v">${params[name]}</span>`;
    root.appendChild(row);
    row.querySelector("input").addEventListener("input", e => {
      params[name] = parseInt(e.target.value, 10);
      document.getElementById(name + "_v").textContent = params[name];
      scheduleSave();
    });
  }
}
function scheduleSave() {
  clearTimeout(timer);
  timer = setTimeout(() => postParams(), 250);
}
async function postParams() {
  await fetch("/api/params", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(params)});
}
async function saveNow() {
  await postParams();
  await fetch("/api/save-now", {method:"POST"});
}
async function snapshotNow() {
  const status = document.getElementById("snapshotStatus");
  try {
    const response = await fetch("/api/snapshot", {method:"POST"});
    const result = await response.json();
    status.textContent = result.saved ? `saved ${result.path}` : `snapshot failed: ${result.error}`;
  } catch (e) {
    status.textContent = "snapshot failed";
  }
}
async function loadParams() {
  params = await (await fetch("/api/params")).json();
  renderControls();
}
async function pollStatus() {
  try {
    const s = await (await fetch("/api/status")).json();
    document.getElementById("status").textContent =
      `${s.found ? "FOUND" : "MISS"} ms=${s.ms.toFixed(2)} fps=${s.fps.toFixed(1)} conf=${s.confidence.toFixed(3)} radius=${s.radius.toFixed(1)} remap=${s.remap_enabled ? "on" : "off"} mask=${s.mask_enabled ? "on" : "off"}`;
  } catch (e) {}
}
loadParams();
setInterval(pollStatus, 500);
window.addEventListener("keydown", e => {
  if (e.target && ["INPUT", "TEXTAREA", "SELECT"].includes(e.target.tagName)) return;
  if (e.key === "s" || e.key === "S") {
    e.preventDefault();
    snapshotNow();
  }
});
</script>
</body>
</html>)HTML";
}

bool readRequest(int fd, std::string& request, std::string& body) {
    char buffer[4096];
    while (request.find("\r\n\r\n") == std::string::npos) {
        const ssize_t n = recv(fd, buffer, sizeof(buffer), 0);
        if (n <= 0) {
            return false;
        }
        request.append(buffer, static_cast<std::size_t>(n));
        if (request.size() > 65536) {
            return false;
        }
    }
    const std::size_t header_end = request.find("\r\n\r\n");
    body = request.substr(header_end + 4);
    int content_length = 0;
    const std::string lower = request;
    const std::string marker = "Content-Length:";
    std::size_t pos = lower.find(marker);
    if (pos != std::string::npos) {
        pos += marker.size();
        while (pos < lower.size() && std::isspace(static_cast<unsigned char>(lower[pos])) != 0) {
            ++pos;
        }
        content_length = std::stoi(lower.substr(pos));
    }
    while (static_cast<int>(body.size()) < content_length) {
        const ssize_t n = recv(fd, buffer, sizeof(buffer), 0);
        if (n <= 0) {
            return false;
        }
        body.append(buffer, static_cast<std::size_t>(n));
    }
    return true;
}

void streamJpeg(int fd, SharedState& state, bool binary) {
    const std::string header =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n"
        "Cache-Control: no-store\r\n"
        "Connection: close\r\n\r\n";
    if (!sendAll(fd, header)) {
        return;
    }

    while (state.running && g_running) {
        std::vector<unsigned char> jpg;
        {
            std::lock_guard<std::mutex> lock(state.mutex);
            jpg = binary ? state.binary_jpeg : state.preview_jpeg;
        }
        if (!jpg.empty()) {
            std::ostringstream part;
            part << "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " << jpg.size() << "\r\n\r\n";
            if (!sendAll(fd, part.str()) || !sendAll(fd, reinterpret_cast<const char*>(jpg.data()), jpg.size()) || !sendAll(fd, "\r\n")) {
                break;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(40));
    }
}

void handleClient(int fd, SharedState& state) {
    std::string request;
    std::string body;
    if (!readRequest(fd, request, body)) {
        close(fd);
        return;
    }
    std::istringstream first_line(request);
    std::string method;
    std::string path;
    first_line >> method >> path;

    if (method == "GET" && path == "/") {
        sendResponse(fd, "text/html; charset=utf-8", htmlPage());
    } else if (method == "GET" && path == "/stream") {
        streamJpeg(fd, state, false);
    } else if (method == "GET" && path == "/binary") {
        streamJpeg(fd, state, true);
    } else if (method == "GET" && path == "/api/params") {
        std::lock_guard<std::mutex> lock(state.mutex);
        sendResponse(fd, "application/json", tuneJson(state.tune));
    } else if (method == "GET" && path == "/api/status") {
        std::lock_guard<std::mutex> lock(state.mutex);
        sendResponse(fd, "application/json", state.status_json);
    } else if (method == "POST" && path == "/api/params") {
        TuneState saved;
        fs::path params_dir;
        {
            std::lock_guard<std::mutex> lock(state.mutex);
            updateTuneFromJson(body, state.tune);
            saved = state.tune;
            params_dir = state.params_dir;
        }
        saveParams(saved, params_dir);
        sendResponse(fd, "application/json", tuneJson(saved));
    } else if (method == "POST" && path == "/api/save-now") {
        TuneState saved;
        fs::path params_dir;
        {
            std::lock_guard<std::mutex> lock(state.mutex);
            saved = state.tune;
            params_dir = state.params_dir;
        }
        saveParams(saved, params_dir);
        sendResponse(fd, "application/json", "{\"saved\":true}\n");
    } else if (method == "POST" && path == "/api/snapshot") {
        std::string error;
        const std::string saved_path = saveSnapshot(state, error);
        if (saved_path.empty()) {
            sendResponse(fd, "application/json", "{\"saved\":false,\"error\":\"" + jsonString(error) + "\"}\n", 500, "Internal Server Error");
        } else {
            sendResponse(fd, "application/json", "{\"saved\":true,\"path\":\"" + jsonString(saved_path) + "\"}\n");
        }
    } else {
        sendResponse(fd, "text/plain", "not found\n", 404, "Not Found");
    }
    close(fd);
}

void captureLoop(SharedState& state, int camera_index, const rcj::FrameRemapper& remapper, const cv::Mat& keep_mask) {
    cv::VideoCapture cap(camera_index);
    if (!cap.isOpened()) {
        std::cerr << "failed to open camera " << camera_index << "\n";
        state.running = false;
        g_running = false;
        return;
    }
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 800);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 600);
    cap.set(cv::CAP_PROP_FPS, 60);

    auto last = std::chrono::steady_clock::now();
    while (state.running && g_running) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        if (remapper.enabled()) {
            cv::Mat remapped;
            std::string error;
            if (!remapper.remap(frame, remapped, &error)) {
                std::cerr << error << "\n";
                state.running = false;
                g_running = false;
                break;
            }
            frame = remapped;
        }

        TuneState tune;
        {
            std::lock_guard<std::mutex> lock(state.mutex);
            tune = state.tune;
        }

        const bool use_mask = tune.use_mask != 0 && !keep_mask.empty();
        cv::Mat detection_frame = frame.clone();
        if (use_mask) {
            std::string error;
            if (!applyKeepMask(detection_frame, keep_mask, error)) {
                std::cerr << error << "\n";
                state.running = false;
                g_running = false;
                break;
            }
        }

        rcj::ArcDetector detector(makeConfig(tune));
        const auto start = std::chrono::steady_clock::now();
        rcj::ArcDetection result = detector.detect(detection_frame);
        const auto end = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(end - start).count();
        const double fps = 1000.0 / std::max(1.0, std::chrono::duration<double, std::milli>(end - last).count());
        last = end;

        cv::Mat shown = frame.clone();
        drawDetection(shown, result, ms, fps);
        std::vector<unsigned char> preview;
        std::vector<unsigned char> binary;
        cv::imencode(".jpg", shown, preview, {cv::IMWRITE_JPEG_QUALITY, 80});
        if (!result.binary_roi.empty()) {
            cv::Mat binary_color;
            cv::cvtColor(result.binary_roi, binary_color, cv::COLOR_GRAY2BGR);
            cv::imencode(".jpg", binary_color, binary, {cv::IMWRITE_JPEG_QUALITY, 80});
        }

        {
            std::lock_guard<std::mutex> lock(state.mutex);
            state.latest_frame = frame.clone();
            state.preview_jpeg = std::move(preview);
            state.binary_jpeg = std::move(binary);
            state.status_json = statusJson(result, ms, fps, state.remap_enabled, state.remap_path, use_mask, state.mask_path);
        }
    }
}

int makeServerSocket(const std::string& bind_host, int port) {
    const int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        return -1;
    }
    int opt = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    if (inet_pton(AF_INET, bind_host.c_str(), &addr.sin_addr) != 1) {
        close(fd);
        return -1;
    }
    if (bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        close(fd);
        return -1;
    }
    if (listen(fd, 16) != 0) {
        close(fd);
        return -1;
    }
    return fd;
}

void printUsage(const char* argv0) {
    std::cerr << "usage: " << argv0 << " [--camera 0] [--bind 0.0.0.0] [--port 8080] [--params-dir params] [--snapshot-dir pic_web] [--mask config/robot_mask.png | --no-mask] [--remap config/remap.xml | --no-remap]\n";
}

}  // namespace

int main(int argc, char** argv) {
    int camera = 0;
    std::string bind_host = "0.0.0.0";
    int port = 8080;
    fs::path params_dir = "params";
    fs::path snapshot_dir = "pic_web";
    bool remap_enabled = false;
    std::string remap_path;
    bool mask_enabled = false;
    fs::path mask_path;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--camera" && i + 1 < argc) {
            camera = std::stoi(argv[++i]);
        } else if (arg == "--bind" && i + 1 < argc) {
            bind_host = argv[++i];
        } else if (arg == "--port" && i + 1 < argc) {
            port = std::stoi(argv[++i]);
        } else if (arg == "--params-dir" && i + 1 < argc) {
            params_dir = argv[++i];
        } else if (arg == "--snapshot-dir" && i + 1 < argc) {
            snapshot_dir = argv[++i];
        } else if (arg == "--mask" && i + 1 < argc) {
            mask_enabled = true;
            mask_path = argv[++i];
        } else if (arg == "--no-mask") {
            mask_enabled = false;
            mask_path.clear();
        } else if (arg == "--remap" && i + 1 < argc) {
            remap_enabled = true;
            remap_path = argv[++i];
        } else if (arg == "--no-remap") {
            remap_enabled = false;
            remap_path.clear();
        } else if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        } else {
            printUsage(argv[0]);
            return 2;
        }
    }

    signal(SIGINT, onSignal);
    signal(SIGTERM, onSignal);
    signal(SIGPIPE, SIG_IGN);

    SharedState state;
    state.params_dir = params_dir;
    state.snapshot_dir = snapshot_dir;
    state.remap_enabled = remap_enabled;
    state.remap_path = remap_path;
    state.mask_enabled = mask_enabled;
    state.mask_path = mask_path.string();
    rcj::FrameRemapper remapper;
    if (remap_enabled) {
        std::string error;
        if (!remapper.load(remap_path, &error)) {
            std::cerr << error << "\n";
            return 1;
        }
        std::cout << "remap enabled: " << remapper.path() << " size=" << remapper.mapSize().width << 'x' << remapper.mapSize().height << "\n";
    }
    cv::Mat keep_mask;
    if (mask_enabled) {
        std::string error;
        if (!loadKeepMask(mask_path, keep_mask, error)) {
            std::cerr << error << "\n";
            return 1;
        }
        std::cout << "mask enabled: " << mask_path << " size=" << keep_mask.cols << 'x' << keep_mask.rows << "\n";
    }
    saveParams(state.tune, state.params_dir);

    std::thread capture_thread(captureLoop, std::ref(state), camera, std::cref(remapper), std::cref(keep_mask));
    const int server_fd = makeServerSocket(bind_host, port);
    if (server_fd < 0) {
        std::cerr << "failed to bind " << bind_host << ':' << port << "\n";
        state.running = false;
        capture_thread.join();
        return 1;
    }

    std::cout << "arc_web_tuner listening on http://" << bind_host << ':' << port << "/\n";
    while (state.running && g_running) {
        sockaddr_in client_addr{};
        socklen_t client_len = sizeof(client_addr);
        const int client_fd = accept(server_fd, reinterpret_cast<sockaddr*>(&client_addr), &client_len);
        if (client_fd < 0) {
            if (g_running) {
                continue;
            }
            break;
        }
        std::thread(handleClient, client_fd, std::ref(state)).detach();
    }

    close(server_fd);
    state.running = false;
    if (capture_thread.joinable()) {
        capture_thread.join();
    }
    return 0;
}
