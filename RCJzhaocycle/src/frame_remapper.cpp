#include "frame_remapper.hpp"

#include <opencv2/imgproc.hpp>

#include <array>
#include <sstream>

namespace rcj {
namespace {

bool readMapPair(cv::FileStorage& storage, cv::Mat& map1, cv::Mat& map2) {
    constexpr std::array<std::array<const char*, 2>, 4> key_pairs{{
        {{"map1", "map2"}},
        {{"map_x", "map_y"}},
        {{"xmap", "ymap"}},
        {{"fast_map_1", "fast_map_2"}},
    }};

    for (const auto& keys : key_pairs) {
        cv::Mat first;
        cv::Mat second;
        storage[keys[0]] >> first;
        storage[keys[1]] >> second;
        if (!first.empty()) {
            map1 = first;
            map2 = second;
            return true;
        }
    }
    return false;
}

bool validMapTypes(const cv::Mat& map1, const cv::Mat& map2) {
    if (map1.type() == CV_32FC2 && map2.empty()) {
        return true;
    }
    if (map1.type() == CV_32FC1 && map2.type() == CV_32FC1) {
        return true;
    }
    if (map1.type() == CV_16SC2 && (map2.empty() || map2.type() == CV_16UC1)) {
        return true;
    }
    return false;
}

void setError(std::string* error, const std::string& message) {
    if (error != nullptr) {
        *error = message;
    }
}

}  // namespace

bool FrameRemapper::load(const std::string& path, std::string* error) {
    cv::FileStorage storage(path, cv::FileStorage::READ);
    if (!storage.isOpened()) {
        setError(error, "failed to open remap XML: " + path);
        return false;
    }

    cv::Mat map1;
    cv::Mat map2;
    if (!readMapPair(storage, map1, map2)) {
        setError(error, "remap XML must contain map1/map2, map_x/map_y, xmap/ymap, or fast_map_1/fast_map_2: " + path);
        return false;
    }
    if (map1.empty()) {
        setError(error, "remap XML map1 is empty: " + path);
        return false;
    }
    if (!map2.empty() && map1.size() != map2.size()) {
        std::ostringstream message;
        message << "remap map sizes differ in " << path << ": map1=" << map1.cols << 'x' << map1.rows
                << " map2=" << map2.cols << 'x' << map2.rows;
        setError(error, message.str());
        return false;
    }
    if (!validMapTypes(map1, map2)) {
        std::ostringstream message;
        message << "unsupported remap map types in " << path << ": map1 type=" << map1.type()
                << " map2 type=" << (map2.empty() ? -1 : map2.type());
        setError(error, message.str());
        return false;
    }

    map1_ = map1;
    map2_ = map2;
    path_ = path;
    return true;
}

bool FrameRemapper::enabled() const {
    return !map1_.empty();
}

const std::string& FrameRemapper::path() const {
    return path_;
}

cv::Size FrameRemapper::mapSize() const {
    return map1_.size();
}

bool FrameRemapper::remap(const cv::Mat& input, cv::Mat& output, std::string* error) const {
    if (!enabled()) {
        output = input;
        return true;
    }
    if (input.empty()) {
        setError(error, "cannot remap an empty frame");
        return false;
    }
    if (input.size() != map1_.size()) {
        std::ostringstream message;
        message << "remap map size " << map1_.cols << 'x' << map1_.rows
                << " does not match frame size " << input.cols << 'x' << input.rows
                << " for " << path_;
        setError(error, message.str());
        return false;
    }

    cv::remap(input, output, map1_, map2_, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
    return true;
}

}  // namespace rcj
