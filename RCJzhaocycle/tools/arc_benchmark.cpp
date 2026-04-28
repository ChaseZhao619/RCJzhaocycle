#include "arc_detector.hpp"
#include "frame_remapper.hpp"

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

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

void printUsage(const char* argv0) {
    std::cerr << "usage: " << argv0 << " [--input pic] [--repeat 1] [--binary-roi] [--remap config/remap.xml | --no-remap]\n";
}

}  // namespace

int main(int argc, char** argv) {
    fs::path input = "pic";
    int repeat = 1;
    bool binary_roi = false;
    bool remap_enabled = false;
    std::string remap_path;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--input" && i + 1 < argc) {
            input = argv[++i];
        } else if (arg == "--repeat" && i + 1 < argc) {
            repeat = std::max(1, std::stoi(argv[++i]));
        } else if (arg == "--binary-roi") {
            binary_roi = true;
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

    std::vector<fs::path> images = collectImages(input);
    if (images.empty()) {
        std::cerr << "no images found in " << input << "\n";
        return 1;
    }

    rcj::ArcDetectorConfig config;
    config.return_binary_roi = binary_roi;
    rcj::FrameRemapper remapper;
    if (remap_enabled) {
        std::string error;
        if (!remapper.load(remap_path, &error)) {
            std::cerr << error << "\n";
            return 1;
        }
        std::cerr << "remap enabled: " << remapper.path() << " size=" << remapper.mapSize().width << 'x' << remapper.mapSize().height << "\n";
    }

    std::vector<double> all_times_ms;
    int found_count = 0;
    std::cout << "image,width,height,found,center_x,center_y,radius,angle_start,angle_end,confidence,avg_ms\n";

    for (const fs::path& path : images) {
        cv::Mat image = cv::imread(path.string(), cv::IMREAD_COLOR);
        if (image.empty()) {
            std::cerr << "failed to read " << path << "\n";
            continue;
        }
        cv::Mat processed = image;
        if (remapper.enabled()) {
            std::string error;
            if (!remapper.remap(image, processed, &error)) {
                std::cerr << error << "\n";
                return 1;
            }
        }

        rcj::ArcDetection last_result;
        std::vector<double> image_times_ms;
        for (int i = 0; i < repeat; ++i) {
            rcj::ArcDetector detector(config);
            const auto start = std::chrono::steady_clock::now();
            last_result = detector.detect(processed);
            const auto end = std::chrono::steady_clock::now();
            const double ms = std::chrono::duration<double, std::milli>(end - start).count();
            image_times_ms.push_back(ms);
            all_times_ms.push_back(ms);
        }

        if (last_result.found) {
            ++found_count;
        }
        const double avg_ms = std::accumulate(image_times_ms.begin(), image_times_ms.end(), 0.0) /
                              static_cast<double>(image_times_ms.size());
        std::cout << path.filename().string() << ','
                  << processed.cols << ','
                  << processed.rows << ','
                  << (last_result.found ? 1 : 0) << ','
                  << last_result.center.x << ','
                  << last_result.center.y << ','
                  << last_result.radius << ','
                  << last_result.angle_start << ','
                  << last_result.angle_end << ','
                  << last_result.confidence << ','
                  << avg_ms << '\n';
    }

    const double avg_ms = std::accumulate(all_times_ms.begin(), all_times_ms.end(), 0.0) /
                          static_cast<double>(all_times_ms.size());
    const double max_ms = *std::max_element(all_times_ms.begin(), all_times_ms.end());
    std::cout << "summary,total_images=" << images.size()
              << ",found=" << found_count
              << ",repeat=" << repeat
              << ",avg_ms=" << avg_ms
              << ",max_ms=" << max_ms
              << ",fps_avg=" << (avg_ms > 0.0 ? 1000.0 / avg_ms : 0.0)
              << '\n';
    return found_count == static_cast<int>(images.size()) ? 0 : 3;
}
