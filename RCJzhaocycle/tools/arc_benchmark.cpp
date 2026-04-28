#include "arc_detector.hpp"
#include "frame_remapper.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct Label {
    fs::path path;
    bool expected_found = false;
    float center_x = 0.0F;
    float center_y = 0.0F;
    float radius = 0.0F;
    float tolerance_px = 25.0F;
};

struct EvalStats {
    int total = 0;
    int pass = 0;
    int positives = 0;
    int positive_pass = 0;
    int negatives = 0;
    int negative_pass = 0;
    int false_positive = 0;
    int false_negative = 0;
    double center_error_sum = 0.0;
    double radius_error_sum = 0.0;
    double max_center_error = 0.0;
    double max_radius_error = 0.0;
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

std::vector<std::string> splitCsvLine(const std::string& line) {
    std::vector<std::string> cells;
    std::string cell;
    bool quoted = false;
    for (std::size_t i = 0; i < line.size(); ++i) {
        const char ch = line[i];
        if (quoted) {
            if (ch == '"' && i + 1 < line.size() && line[i + 1] == '"') {
                cell.push_back('"');
                ++i;
            } else if (ch == '"') {
                quoted = false;
            } else {
                cell.push_back(ch);
            }
        } else if (ch == '"') {
            quoted = true;
        } else if (ch == ',') {
            cells.push_back(cell);
            cell.clear();
        } else {
            cell.push_back(ch);
        }
    }
    cells.push_back(cell);
    return cells;
}

int columnIndex(const std::vector<std::string>& header, const std::string& name) {
    for (std::size_t i = 0; i < header.size(); ++i) {
        if (header[i] == name) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

std::string cellAt(const std::vector<std::string>& row, int index) {
    if (index < 0 || index >= static_cast<int>(row.size())) {
        return {};
    }
    return row[static_cast<std::size_t>(index)];
}

float parseFloatOr(const std::string& value, float fallback) {
    if (value.empty()) {
        return fallback;
    }
    return std::stof(value);
}

std::vector<Label> loadLabels(const fs::path& testset_dir, const fs::path& labels_csv) {
    std::ifstream input(labels_csv);
    if (!input) {
        throw std::runtime_error("failed to open labels CSV: " + labels_csv.string());
    }

    std::string line;
    if (!std::getline(input, line)) {
        throw std::runtime_error("empty labels CSV: " + labels_csv.string());
    }
    const std::vector<std::string> header = splitCsvLine(line);
    const int filename_col = columnIndex(header, "filename");
    const int expected_col = columnIndex(header, "expected_found");
    const int center_x_col = columnIndex(header, "center_x");
    const int center_y_col = columnIndex(header, "center_y");
    const int radius_col = columnIndex(header, "radius");
    const int tolerance_col = columnIndex(header, "tolerance_px");
    if (filename_col < 0 || expected_col < 0) {
        throw std::runtime_error("labels CSV must contain filename and expected_found columns");
    }

    std::vector<Label> labels;
    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }
        const std::vector<std::string> row = splitCsvLine(line);
        const std::string filename = cellAt(row, filename_col);
        if (filename.empty()) {
            continue;
        }
        Label label;
        label.path = testset_dir / filename;
        label.expected_found = std::stoi(cellAt(row, expected_col)) != 0;
        label.center_x = parseFloatOr(cellAt(row, center_x_col), 0.0F);
        label.center_y = parseFloatOr(cellAt(row, center_y_col), 0.0F);
        label.radius = parseFloatOr(cellAt(row, radius_col), 0.0F);
        label.tolerance_px = parseFloatOr(cellAt(row, tolerance_col), 25.0F);
        labels.push_back(label);
    }
    return labels;
}

std::vector<Label> loadLabelsMany(const fs::path& testset_dir, const std::vector<fs::path>& labels_csvs) {
    std::vector<Label> labels;
    for (const fs::path& labels_csv : labels_csvs) {
        std::vector<Label> loaded = loadLabels(testset_dir, labels_csv);
        labels.insert(labels.end(), loaded.begin(), loaded.end());
    }
    return labels;
}

void addNegativeLabels(const fs::path& testset_dir, std::vector<Label>& labels) {
    const fs::path negatives_dir = testset_dir / "negatives";
    for (const fs::path& path : collectImages(negatives_dir)) {
        Label label;
        label.path = path;
        label.expected_found = false;
        labels.push_back(label);
    }
}

std::string displayPath(const fs::path& path, const fs::path& base) {
    std::error_code ec;
    const fs::path relative = fs::relative(path, base, ec);
    return ec ? path.string() : relative.string();
}

void applyConfigArg(rcj::ArcDetectorConfig& config, const std::string& arg, const std::string& value) {
    if (arg == "--scale-percent") {
        config.processing_scale = std::max(1, std::stoi(value)) / 100.0F;
    } else if (arg == "--max-width") {
        config.processing_max_width = std::stoi(value);
    } else if (arg == "--max-height") {
        config.processing_max_height = std::stoi(value);
    } else if (arg == "--use-hough") {
        config.use_hough = std::stoi(value) != 0;
    } else if (arg == "--min-radius") {
        config.min_radius = std::stoi(value);
    } else if (arg == "--max-radius") {
        config.max_radius = std::stoi(value);
    } else if (arg == "--ring-band") {
        config.ring_band = std::stoi(value);
    } else if (arg == "--min-arc-bins") {
        config.min_arc_bins = std::stoi(value);
    } else if (arg == "--dark-value-max") {
        config.dark_value_max = std::stoi(value);
    } else if (arg == "--dark-offset") {
        config.dark_adaptive_offset = std::stoi(value);
    } else if (arg == "--min-confidence") {
        config.min_confidence = std::max(0, std::stoi(value)) / 100.0F;
    } else if (arg == "--green-hue-min") {
        config.green_hue_min = std::stoi(value);
    } else if (arg == "--green-hue-max") {
        config.green_hue_max = std::stoi(value);
    } else if (arg == "--green-sat-min") {
        config.green_sat_min = std::stoi(value);
    }
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
                << " does not match image size " << image.cols << 'x' << image.rows;
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

void printUsage(const char* argv0) {
    std::cerr << "usage: " << argv0 << " [--input pic | --testset testset] [--labels testset/labels.csv] [--repeat 1] [--binary-roi] [--mask config/robot_mask.png] [--remap config/remap.xml | --no-remap]\n"
              << "       detector params: --scale-percent N --max-width N --max-height N --use-hough 0|1 --min-radius N --max-radius N\n"
              << "                        --ring-band N --min-arc-bins N --dark-value-max N --dark-offset N --min-confidence N\n"
              << "                        --green-hue-min N --green-hue-max N --green-sat-min N\n";
}

}  // namespace

int main(int argc, char** argv) {
    fs::path input = "pic";
    fs::path testset_dir;
    std::vector<fs::path> labels_csvs;
    fs::path mask_path;
    int repeat = 1;
    bool binary_roi = false;
    bool remap_enabled = false;
    std::string remap_path;
    rcj::ArcDetectorConfig config;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--input" && i + 1 < argc) {
            input = argv[++i];
        } else if (arg == "--testset" && i + 1 < argc) {
            testset_dir = argv[++i];
            if (labels_csvs.empty()) {
                labels_csvs.push_back(testset_dir / "labels.csv");
            }
        } else if (arg == "--labels" && i + 1 < argc) {
            labels_csvs.push_back(argv[++i]);
        } else if (arg == "--mask" && i + 1 < argc) {
            mask_path = argv[++i];
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
        } else if (i + 1 < argc && arg.rfind("--", 0) == 0) {
            applyConfigArg(config, arg, argv[++i]);
        } else {
            printUsage(argv[0]);
            return 2;
        }
    }

    config.return_binary_roi = binary_roi;

    std::vector<fs::path> images;
    std::vector<Label> labels;
    const bool labeled_mode = !labels_csvs.empty();
    if (labeled_mode) {
        if (testset_dir.empty()) {
            testset_dir = labels_csvs.front().parent_path();
        }
        try {
            labels = loadLabelsMany(testset_dir, labels_csvs);
            addNegativeLabels(testset_dir, labels);
        } catch (const std::exception& ex) {
            std::cerr << ex.what() << "\n";
            return 1;
        }
        for (const Label& label : labels) {
            images.push_back(label.path);
        }
    } else {
        images = collectImages(input);
        if (images.empty()) {
            std::cerr << "no images found in " << input << "\n";
            return 1;
        }
    }

    rcj::FrameRemapper remapper;
    if (remap_enabled) {
        std::string error;
        if (!remapper.load(remap_path, &error)) {
            std::cerr << error << "\n";
            return 1;
        }
        std::cerr << "remap enabled: " << remapper.path() << " size=" << remapper.mapSize().width << 'x' << remapper.mapSize().height << "\n";
    }
    cv::Mat keep_mask;
    if (!mask_path.empty()) {
        std::string error;
        if (!loadKeepMask(mask_path, keep_mask, error)) {
            std::cerr << error << "\n";
            return 1;
        }
        std::cerr << "mask enabled: " << mask_path << " size=" << keep_mask.cols << 'x' << keep_mask.rows << "\n";
    }

    std::vector<double> all_times_ms;
    int found_count = 0;
    EvalStats eval;
    std::cout << "image,width,height,found,center_x,center_y,radius,angle_start,angle_end,confidence,avg_ms";
    if (labeled_mode) {
        std::cout << ",expected_found,pass,center_error,radius_error";
    }
    std::cout << "\n";

    for (std::size_t image_index = 0; image_index < images.size(); ++image_index) {
        const fs::path& path = images[image_index];
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
        if (!keep_mask.empty()) {
            std::string error;
            if (!applyKeepMask(processed, keep_mask, error)) {
                std::cerr << error << " for " << path << "\n";
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
        std::cout << (labeled_mode ? displayPath(path, testset_dir) : path.filename().string()) << ','
                  << processed.cols << ','
                  << processed.rows << ','
                  << (last_result.found ? 1 : 0) << ','
                  << last_result.center.x << ','
                  << last_result.center.y << ','
                  << last_result.radius << ','
                  << last_result.angle_start << ','
                  << last_result.angle_end << ','
                  << last_result.confidence << ','
                  << avg_ms;

        if (labeled_mode) {
            const Label& label = labels[image_index];
            bool passed = false;
            double center_error = 0.0;
            double radius_error = 0.0;
            if (label.expected_found) {
                ++eval.positives;
                if (last_result.found) {
                    center_error = std::hypot(
                        static_cast<double>(last_result.center.x - label.center_x),
                        static_cast<double>(last_result.center.y - label.center_y));
                    radius_error = std::abs(static_cast<double>(last_result.radius - label.radius));
                    passed = center_error <= label.tolerance_px && radius_error <= label.tolerance_px;
                    eval.center_error_sum += center_error;
                    eval.radius_error_sum += radius_error;
                    eval.max_center_error = std::max(eval.max_center_error, center_error);
                    eval.max_radius_error = std::max(eval.max_radius_error, radius_error);
                }
                if (passed) {
                    ++eval.positive_pass;
                } else {
                    ++eval.false_negative;
                }
            } else {
                ++eval.negatives;
                passed = !last_result.found;
                if (passed) {
                    ++eval.negative_pass;
                } else {
                    ++eval.false_positive;
                }
            }
            ++eval.total;
            if (passed) {
                ++eval.pass;
            }
            std::cout << ',' << (label.expected_found ? 1 : 0)
                      << ',' << (passed ? 1 : 0)
                      << ',' << center_error
                      << ',' << radius_error;
        }
        std::cout << '\n';
    }

    const double avg_ms = std::accumulate(all_times_ms.begin(), all_times_ms.end(), 0.0) /
                          static_cast<double>(all_times_ms.size());
    const double max_ms = *std::max_element(all_times_ms.begin(), all_times_ms.end());
    std::cout << "summary,total_images=" << images.size()
              << ",found=" << found_count
              << ",repeat=" << repeat
              << ",avg_ms=" << avg_ms
              << ",max_ms=" << max_ms
              << ",fps_avg=" << (avg_ms > 0.0 ? 1000.0 / avg_ms : 0.0);
    if (labeled_mode) {
        const double avg_center_error = eval.positive_pass > 0 ? eval.center_error_sum / eval.positive_pass : 0.0;
        const double avg_radius_error = eval.positive_pass > 0 ? eval.radius_error_sum / eval.positive_pass : 0.0;
        std::cout << ",pass=" << eval.pass
                  << ",positives=" << eval.positives
                  << ",positive_pass=" << eval.positive_pass
                  << ",negatives=" << eval.negatives
                  << ",negative_pass=" << eval.negative_pass
                  << ",false_positive=" << eval.false_positive
                  << ",false_negative=" << eval.false_negative
                  << ",avg_center_error=" << avg_center_error
                  << ",avg_radius_error=" << avg_radius_error
                  << ",max_center_error=" << eval.max_center_error
                  << ",max_radius_error=" << eval.max_radius_error;
    }
    std::cout << '\n';
    if (labeled_mode) {
        return eval.pass == eval.total ? 0 : 3;
    }
    return found_count == static_cast<int>(images.size()) ? 0 : 3;
}
