#include "arc_detector.hpp"
#include "frame_remapper.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct TuneState {
    int scale_percent = 50;
    int max_width = 400;
    int max_height = 300;
    int use_hough = 0;
    int use_ransac = 1;
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

bool isImageFile(const fs::path& path) {
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp" || ext == ".tif" || ext == ".tiff";
}

int numericStem(const fs::path& path) {
    const std::string stem = path.stem().string();
    if (stem.empty() || !std::all_of(stem.begin(), stem.end(), [](unsigned char c) { return std::isdigit(c); })) {
        return -1;
    }
    return std::stoi(stem);
}

std::vector<fs::path> collectImages(const fs::path& input) {
    std::vector<fs::path> images;
    if (fs::is_regular_file(input) && isImageFile(input)) {
        images.push_back(input);
    } else if (fs::is_directory(input)) {
        for (const fs::directory_entry& entry : fs::directory_iterator(input)) {
            if (entry.is_regular_file() && isImageFile(entry.path())) {
                images.push_back(entry.path());
            }
        }
    }

    std::sort(images.begin(), images.end(), [](const fs::path& lhs, const fs::path& rhs) {
        const int left_num = numericStem(lhs);
        const int right_num = numericStem(rhs);
        if (left_num >= 0 && right_num >= 0) {
            return left_num < right_num;
        }
        return lhs.filename().string() < rhs.filename().string();
    });
    return images;
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

void drawDetection(cv::Mat& frame, const rcj::ArcDetection& result, double ms, const std::string& label) {
    if (result.found) {
        cv::circle(frame, result.center, static_cast<int>(std::round(result.radius)), cv::Scalar(0, 255, 255), 2);
        cv::circle(frame, result.center, 3, cv::Scalar(0, 0, 255), -1);
        cv::rectangle(frame, result.roi_rect, cv::Scalar(255, 0, 0), 2);
    }

    const std::string status = (result.found ? "FOUND " : "MISS ") +
                               label +
                               " ms=" + std::to_string(ms).substr(0, 5) +
                               " conf=" + std::to_string(result.confidence).substr(0, 5);
    cv::rectangle(frame, cv::Rect(0, 0, frame.cols, 34), cv::Scalar(0, 0, 0), -1);
    cv::putText(frame, status, cv::Point(8, 23), cv::FONT_HERSHEY_SIMPLEX, 0.62, cv::Scalar(255, 255, 255), 2);
}

void printConfig(const TuneState& state) {
    const rcj::ArcDetectorConfig config = makeConfig(state);
    std::cout << "ArcDetectorConfig{"
              << "processing_scale=" << config.processing_scale
              << ", processing_max_width=" << config.processing_max_width
              << ", processing_max_height=" << config.processing_max_height
              << ", use_hough=" << config.use_hough
              << ", use_ransac_candidates=" << config.use_ransac_candidates
              << ", min_radius=" << config.min_radius
              << ", max_radius=" << config.max_radius
              << ", ring_band=" << config.ring_band
              << ", min_arc_bins=" << config.min_arc_bins
              << ", dark_value_max=" << config.dark_value_max
              << ", dark_adaptive_offset=" << config.dark_adaptive_offset
              << ", min_confidence=" << config.min_confidence
              << ", green_hue_min=" << config.green_hue_min
              << ", green_hue_max=" << config.green_hue_max
              << ", green_sat_min=" << config.green_sat_min
              << "}\n";
}

void createTrackbars(const std::string& window, TuneState& state) {
    cv::createTrackbar("scale %", window, &state.scale_percent, 100);
    cv::createTrackbar("max width", window, &state.max_width, 800);
    cv::createTrackbar("max height", window, &state.max_height, 600);
    cv::createTrackbar("use hough", window, &state.use_hough, 1);
    cv::createTrackbar("use ransac", window, &state.use_ransac, 1);
    cv::createTrackbar("min radius", window, &state.min_radius, 800);
    cv::createTrackbar("max radius", window, &state.max_radius, 900);
    cv::createTrackbar("ring band", window, &state.ring_band, 80);
    cv::createTrackbar("min arc bins", window, &state.min_arc_bins, 72);
    cv::createTrackbar("dark max", window, &state.dark_value_max, 255);
    cv::createTrackbar("dark offset", window, &state.dark_offset, 120);
    cv::createTrackbar("min conf x100", window, &state.min_confidence, 100);
    cv::createTrackbar("green hue min", window, &state.green_hue_min, 179);
    cv::createTrackbar("green hue max", window, &state.green_hue_max, 179);
    cv::createTrackbar("green sat min", window, &state.green_sat_min, 255);
}

void printUsage(const char* argv0) {
    std::cerr << "usage: " << argv0 << " [--input pic | --camera 0] [--display :0] [--remap config/remap.xml | --no-remap]\n";
}

}  // namespace

int main(int argc, char** argv) {
    fs::path input = "pic";
    int camera = -1;
    std::string display;
    bool remap_enabled = false;
    std::string remap_path;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--input" && i + 1 < argc) {
            input = argv[++i];
            camera = -1;
        } else if (arg == "--camera" && i + 1 < argc) {
            camera = std::stoi(argv[++i]);
        } else if (arg == "--display" && i + 1 < argc) {
            display = argv[++i];
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

    if (!display.empty()) {
        setenv("DISPLAY", display.c_str(), 1);
    }

    rcj::FrameRemapper remapper;
    if (remap_enabled) {
        std::string error;
        if (!remapper.load(remap_path, &error)) {
            std::cerr << error << "\n";
            return 1;
        }
        std::cout << "remap enabled: " << remapper.path() << " size=" << remapper.mapSize().width << 'x' << remapper.mapSize().height << "\n";
    }

    const std::string window = "arc tuner";
    const std::string binary_window = "binary roi";
    TuneState state;
    cv::namedWindow(window, cv::WINDOW_NORMAL);
    cv::resizeWindow(window, 960, 720);
    cv::namedWindow(binary_window, cv::WINDOW_NORMAL);
    cv::resizeWindow(binary_window, 420, 320);
    createTrackbars(window, state);

    std::vector<fs::path> images;
    std::size_t image_index = 0;
    cv::VideoCapture cap;
    if (camera >= 0) {
        cap.open(camera);
        if (!cap.isOpened()) {
            std::cerr << "failed to open camera " << camera << "\n";
            return 1;
        }
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 800);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 600);
        cap.set(cv::CAP_PROP_FPS, 60);
    } else {
        images = collectImages(input);
        if (images.empty()) {
            std::cerr << "no images found in " << input << "\n";
            return 1;
        }
    }

    bool paused = false;
    cv::Mat frozen_frame;
    int saved_index = 0;
    while (true) {
        cv::Mat frame;
        std::string label;
        if (camera >= 0) {
            if (!paused || frozen_frame.empty()) {
                cap >> frame;
                if (frame.empty()) {
                    break;
                }
                frozen_frame = frame.clone();
            } else {
                frame = frozen_frame.clone();
            }
            label = "camera";
        } else {
            frame = cv::imread(images[image_index].string(), cv::IMREAD_COLOR);
            if (frame.empty()) {
                std::cerr << "failed to read " << images[image_index] << "\n";
                return 1;
            }
            label = images[image_index].filename().string();
        }
        if (remapper.enabled()) {
            cv::Mat remapped;
            std::string error;
            if (!remapper.remap(frame, remapped, &error)) {
                std::cerr << error << "\n";
                return 1;
            }
            frame = remapped;
            label += " remapped";
        }

        const rcj::ArcDetectorConfig config = makeConfig(state);
        rcj::ArcDetector detector(config);
        const auto start = std::chrono::steady_clock::now();
        const rcj::ArcDetection result = detector.detect(frame);
        const auto end = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(end - start).count();

        cv::Mat shown = frame.clone();
        drawDetection(shown, result, ms, label);
        cv::imshow(window, shown);
        if (!result.binary_roi.empty()) {
            cv::imshow(binary_window, result.binary_roi);
        }

        const int key = cv::waitKey(camera >= 0 && !paused ? 1 : 30) & 0xFF;
        if (key == 'q' || key == 27) {
            break;
        }
        if (key == 'p') {
            printConfig(state);
        } else if (key == 's') {
            const std::string out = "tuned_" + std::to_string(saved_index++) + ".png";
            cv::imwrite(out, shown);
            std::cout << "wrote " << out << "\n";
        } else if (key == ' ' && camera >= 0) {
            paused = !paused;
        } else if ((key == 'n' || key == 83) && !images.empty()) {
            image_index = (image_index + 1) % images.size();
        } else if ((key == 'b' || key == 81) && !images.empty()) {
            image_index = (image_index + images.size() - 1) % images.size();
        }
    }
    return 0;
}
