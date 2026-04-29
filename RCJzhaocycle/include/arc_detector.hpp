#pragma once

#include <opencv2/core.hpp>

#include <vector>

namespace rcj {

struct ArcDetection {
    bool found = false;
    cv::Point2f center;
    float radius = 0.0F;
    float angle_start = 0.0F;
    float angle_end = 0.0F;
    float confidence = 0.0F;
    cv::Rect roi_rect;
    cv::Mat binary_roi;
};

struct ArcDetectorConfig {
    float processing_scale = 0.5F;
    int processing_max_width = 400;
    int processing_max_height = 300;
    bool return_binary_roi = false;
    bool use_hough = false;
    bool use_ransac_candidates = true;

    int min_radius = 45;
    int max_radius = 430;
    int ring_band = 10;
    int min_arc_bins = 8;
    int angle_bins = 72;

    double hough_dp = 1.2;
    double hough_param1 = 90.0;
    double hough_param2 = 10.0;
    int hough_min_dist = 90;

    int green_hue_min = 32;
    int green_hue_max = 96;
    int green_sat_min = 28;
    int green_value_min = 35;
    int dark_value_max = 118;
    int dark_adaptive_offset = 34;

    float min_confidence = 0.18F;
    int max_candidates = 32;
    int ransac_iterations = 900;
    int ransac_max_points = 700;
    int roi_padding = 90;
};

class ArcDetector {
public:
    explicit ArcDetector(ArcDetectorConfig config = {});

    ArcDetection detect(const cv::Mat& frame);
    void reset();

private:
    struct Candidate {
        cv::Point2f center;
        float radius = 0.0F;
    };

    struct Score {
        float confidence = 0.0F;
        int support = 0;
        int interior = 0;
        int arc_bins = 0;
        float angle_start = 0.0F;
        float angle_end = 0.0F;
    };

    ArcDetection detectInRect(const cv::Mat& frame, const cv::Rect& full_rect);
    float processingScaleFor(const cv::Size& size) const;
    void buildMasks(const cv::Mat& small, cv::Mat& gray, cv::Mat& field_mask, cv::Mat& dark_mask) const;
    std::vector<Candidate> makeCandidates(const cv::Mat& gray, const cv::Mat& dark_mask, float scale) const;
    Score scoreCandidate(const Candidate& candidate, const cv::Mat& dark_mask, float scale) const;
    cv::Rect trackingRect(const cv::Size& frame_size) const;

    ArcDetectorConfig config_;
    bool has_last_ = false;
    ArcDetection last_;
};

}  // namespace rcj
