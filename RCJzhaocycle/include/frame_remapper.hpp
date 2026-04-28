#pragma once

#include <opencv2/core.hpp>

#include <string>

namespace rcj {

class FrameRemapper {
public:
    bool load(const std::string& path, std::string* error = nullptr);
    bool enabled() const;
    const std::string& path() const;
    cv::Size mapSize() const;

    bool remap(const cv::Mat& input, cv::Mat& output, std::string* error = nullptr) const;

private:
    cv::Mat map1_;
    cv::Mat map2_;
    std::string path_;
};

}  // namespace rcj
