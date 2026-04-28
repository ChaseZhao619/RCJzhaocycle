#include "arc_detector.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>

namespace rcj {
namespace {

constexpr float kPi = 3.14159265358979323846F;

cv::Rect clampRect(const cv::Rect& rect, const cv::Size& size) {
    const cv::Rect bounds(0, 0, size.width, size.height);
    return rect & bounds;
}

float clampFloat(float value, float low, float high) {
    return std::max(low, std::min(value, high));
}

float normAngle(float angle) {
    while (angle < 0.0F) {
        angle += 360.0F;
    }
    while (angle >= 360.0F) {
        angle -= 360.0F;
    }
    return angle;
}

int longestGapStart(const std::vector<unsigned char>& bins) {
    int best_start = 0;
    int best_len = -1;
    const int n = static_cast<int>(bins.size());
    for (int i = 0; i < n; ++i) {
        if (bins[i] != 0) {
            continue;
        }
        int len = 0;
        while (len < n && bins[(i + len) % n] == 0) {
            ++len;
        }
        if (len > best_len) {
            best_len = len;
            best_start = i;
        }
        i += std::max(0, len - 1);
    }
    return (best_start + std::max(0, best_len)) % n;
}

}  // namespace

ArcDetector::ArcDetector(ArcDetectorConfig config) : config_(config) {
    config_.processing_scale = clampFloat(config_.processing_scale, 0.25F, 1.0F);
    config_.processing_max_width = std::max(64, config_.processing_max_width);
    config_.processing_max_height = std::max(48, config_.processing_max_height);
    config_.angle_bins = std::max(24, config_.angle_bins);
    config_.max_candidates = std::max(4, config_.max_candidates);
}

void ArcDetector::reset() {
    has_last_ = false;
    last_ = ArcDetection{};
}

ArcDetection ArcDetector::detect(const cv::Mat& frame) {
    if (frame.empty()) {
        reset();
        return {};
    }

    if (has_last_) {
        ArcDetection tracked = detectInRect(frame, trackingRect(frame.size()));
        if (tracked.found) {
            last_ = tracked;
            return tracked;
        }
        reset();
        return {};
    }

    ArcDetection detected = detectInRect(frame, cv::Rect(0, 0, frame.cols, frame.rows));
    if (detected.found) {
        last_ = detected;
    } else {
        reset();
    }
    return detected;
}

cv::Rect ArcDetector::trackingRect(const cv::Size& frame_size) const {
    const int pad = std::max(config_.roi_padding, static_cast<int>(last_.radius * 0.35F));
    const int extent = static_cast<int>(std::ceil(last_.radius + pad));
    const cv::Rect rect(
        static_cast<int>(std::floor(last_.center.x)) - extent,
        static_cast<int>(std::floor(last_.center.y)) - extent,
        extent * 2,
        extent * 2);
    return clampRect(rect, frame_size);
}

float ArcDetector::processingScaleFor(const cv::Size& size) const {
    const float by_width = static_cast<float>(config_.processing_max_width) / static_cast<float>(std::max(1, size.width));
    const float by_height = static_cast<float>(config_.processing_max_height) / static_cast<float>(std::max(1, size.height));
    return clampFloat(std::min({config_.processing_scale, by_width, by_height}), 0.08F, 1.0F);
}

ArcDetection ArcDetector::detectInRect(const cv::Mat& frame, const cv::Rect& full_rect) {
    const cv::Rect roi = clampRect(full_rect, frame.size());
    if (roi.empty()) {
        return {};
    }

    cv::Mat cropped = frame(roi);
    cv::Mat small;
    const float scale = processingScaleFor(roi.size());
    cv::resize(cropped, small, cv::Size(), scale, scale, cv::INTER_AREA);

    cv::Mat gray;
    cv::Mat field_mask;
    cv::Mat dark_mask;
    buildMasks(small, gray, field_mask, dark_mask);

    std::vector<Candidate> candidates = makeCandidates(gray, dark_mask, scale);
    Score best_score;
    Candidate best_candidate;
    for (const Candidate& candidate : candidates) {
        Score score = scoreCandidate(candidate, dark_mask, scale);
        if (score.confidence > best_score.confidence) {
            best_score = score;
            best_candidate = candidate;
        }
    }

    if (best_score.confidence < config_.min_confidence) {
        return {};
    }

    const float inv_scale = 1.0F / scale;
    ArcDetection result;
    result.found = true;
    result.center = cv::Point2f(
        roi.x + best_candidate.center.x * inv_scale,
        roi.y + best_candidate.center.y * inv_scale);
    result.radius = best_candidate.radius * inv_scale;
    result.angle_start = best_score.angle_start;
    result.angle_end = best_score.angle_end;
    result.confidence = best_score.confidence;

    const int band = std::max(config_.ring_band, static_cast<int>(result.radius * 0.06F));
    result.roi_rect = clampRect(
        cv::Rect(
            static_cast<int>(std::floor(result.center.x - result.radius - band)),
            static_cast<int>(std::floor(result.center.y - result.radius - band)),
            static_cast<int>(std::ceil((result.radius + band) * 2.0F)),
            static_cast<int>(std::ceil((result.radius + band) * 2.0F))),
        frame.size());

    if (config_.return_binary_roi) {
        const int small_band = std::max(3, static_cast<int>(std::round(config_.ring_band * scale)));
        cv::Rect small_rect(
            static_cast<int>(std::floor(best_candidate.center.x - best_candidate.radius - small_band)),
            static_cast<int>(std::floor(best_candidate.center.y - best_candidate.radius - small_band)),
            static_cast<int>(std::ceil((best_candidate.radius + small_band) * 2.0F)),
            static_cast<int>(std::ceil((best_candidate.radius + small_band) * 2.0F)));
        small_rect = clampRect(small_rect, dark_mask.size());
        if (!small_rect.empty()) {
            result.binary_roi = cv::Mat::zeros(small_rect.size(), CV_8U);
            const float inner = std::max(1.0F, best_candidate.radius - small_band);
            const float outer = best_candidate.radius + small_band;
            const float inner2 = inner * inner;
            const float outer2 = outer * outer;
            for (int y = 0; y < small_rect.height; ++y) {
                const int source_y = small_rect.y + y;
                const unsigned char* source_row = dark_mask.ptr<unsigned char>(source_y);
                unsigned char* target_row = result.binary_roi.ptr<unsigned char>(y);
                for (int x = 0; x < small_rect.width; ++x) {
                    const int source_x = small_rect.x + x;
                    const float dx = static_cast<float>(source_x) - best_candidate.center.x;
                    const float dy = static_cast<float>(source_y) - best_candidate.center.y;
                    const float d2 = dx * dx + dy * dy;
                    if (d2 >= inner2 && d2 <= outer2) {
                        target_row[x] = source_row[source_x];
                    }
                }
            }
        }
    }

    return result;
}

void ArcDetector::buildMasks(const cv::Mat& small, cv::Mat& gray, cv::Mat& field_mask, cv::Mat& dark_mask) const {
    cv::Mat bgr;
    bool grayscale_input = false;
    if (small.channels() == 1) {
        gray = small;
        cv::cvtColor(small, bgr, cv::COLOR_GRAY2BGR);
        grayscale_input = true;
    } else if (small.channels() == 3) {
        cv::cvtColor(small, gray, cv::COLOR_BGR2GRAY);
        bgr = small;
    } else if (small.channels() == 4) {
        cv::cvtColor(small, bgr, cv::COLOR_BGRA2BGR);
        cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = cv::Mat{};
        field_mask = cv::Mat{};
        dark_mask = cv::Mat{};
        return;
    }

    if (grayscale_input) {
        field_mask = cv::Mat(gray.size(), CV_8U, cv::Scalar(255));
    } else {
        cv::Mat hsv;
        cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
        cv::inRange(
            hsv,
            cv::Scalar(config_.green_hue_min, config_.green_sat_min, config_.green_value_min),
            cv::Scalar(config_.green_hue_max, 255, 255),
            field_mask);
    }

    cv::Mat kernel5 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::Mat kernel9 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(9, 9));
    cv::morphologyEx(field_mask, field_mask, cv::MORPH_CLOSE, kernel9);
    cv::dilate(field_mask, field_mask, kernel9, cv::Point(-1, -1), 1);

    const cv::Scalar mean_value = cv::mean(gray, field_mask);
    const double adaptive_dark = std::max(25.0, mean_value[0] - config_.dark_adaptive_offset);
    const double dark_threshold = std::min<double>(config_.dark_value_max, adaptive_dark);

    cv::threshold(gray, dark_mask, dark_threshold, 255, cv::THRESH_BINARY_INV);
    cv::bitwise_and(dark_mask, field_mask, dark_mask);
    cv::morphologyEx(dark_mask, dark_mask, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)));
    cv::morphologyEx(dark_mask, dark_mask, cv::MORPH_CLOSE, kernel5);
}

std::vector<ArcDetector::Candidate> ArcDetector::makeCandidates(const cv::Mat& gray, const cv::Mat& dark_mask, float scale) const {
    std::vector<Candidate> candidates;
    if (gray.empty() || dark_mask.empty()) {
        return candidates;
    }

    const int min_radius = std::max(8, static_cast<int>(std::round(config_.min_radius * scale)));
    const int max_radius = std::max(min_radius + 2, static_cast<int>(std::round(config_.max_radius * scale)));

    if (config_.use_hough) {
        cv::Mat hough_source;
        cv::GaussianBlur(dark_mask, hough_source, cv::Size(5, 5), 1.0);
        const int min_dist = std::max(20, static_cast<int>(std::round(config_.hough_min_dist * scale)));

        std::vector<cv::Vec3f> circles;
        cv::HoughCircles(
            hough_source,
            circles,
            cv::HOUGH_GRADIENT,
            config_.hough_dp,
            min_dist,
            config_.hough_param1,
            config_.hough_param2,
            min_radius,
            max_radius);

        for (const cv::Vec3f& circle : circles) {
            candidates.push_back({cv::Point2f(circle[0], circle[1]), circle[2]});
            if (static_cast<int>(candidates.size()) >= config_.max_candidates) {
                return candidates;
            }
        }
    }

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(dark_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    std::sort(contours.begin(), contours.end(), [](const auto& lhs, const auto& rhs) {
        return cv::contourArea(lhs) > cv::contourArea(rhs);
    });

    for (const std::vector<cv::Point>& contour : contours) {
        if (static_cast<int>(candidates.size()) >= config_.max_candidates) {
            break;
        }
        const double area = cv::contourArea(contour);
        if (area < 18.0 || contour.size() < 6) {
            continue;
        }

        if (contour.size() >= 12) {
            const cv::RotatedRect ellipse = cv::fitEllipse(contour);
            const float radius = (ellipse.size.width + ellipse.size.height) * 0.25F;
            if (radius >= min_radius && radius <= max_radius) {
                candidates.push_back({ellipse.center, radius});
            }
        }

        cv::Point2f center;
        float radius = 0.0F;
        cv::minEnclosingCircle(contour, center, radius);
        if (radius >= min_radius && radius <= max_radius) {
            candidates.push_back({center, radius});
        }
    }

    return candidates;
}

ArcDetector::Score ArcDetector::scoreCandidate(const Candidate& candidate, const cv::Mat& dark_mask, float scale) const {
    Score score;
    if (candidate.radius <= 1.0F) {
        return score;
    }

    const int band = std::max(3, static_cast<int>(std::round(config_.ring_band * scale)));
    const float inner = std::max(1.0F, candidate.radius - band);
    const float outer = candidate.radius + band;
    const float interior_radius = std::max(1.0F, candidate.radius - 2.7F * band);
    const float inner2 = inner * inner;
    const float outer2 = outer * outer;
    const float interior2 = interior_radius * interior_radius;

    const cv::Rect bounds = clampRect(
        cv::Rect(
            static_cast<int>(std::floor(candidate.center.x - outer)),
            static_cast<int>(std::floor(candidate.center.y - outer)),
            static_cast<int>(std::ceil(outer * 2.0F + 1.0F)),
            static_cast<int>(std::ceil(outer * 2.0F + 1.0F))),
        dark_mask.size());
    if (bounds.empty()) {
        return score;
    }

    std::vector<unsigned char> angle_bins(config_.angle_bins, 0);
    int ring_area = 0;
    int interior_area = 0;
    for (int y = bounds.y; y < bounds.y + bounds.height; ++y) {
        const unsigned char* row = dark_mask.ptr<unsigned char>(y);
        const float dy = static_cast<float>(y) - candidate.center.y;
        for (int x = bounds.x; x < bounds.x + bounds.width; ++x) {
            const float dx = static_cast<float>(x) - candidate.center.x;
            const float d2 = dx * dx + dy * dy;
            if (d2 >= inner2 && d2 <= outer2) {
                ++ring_area;
                if (row[x] != 0) {
                    ++score.support;
                    float angle = std::atan2(dy, dx) * 180.0F / kPi;
                    angle = normAngle(angle);
                    const int bin = std::min(
                        config_.angle_bins - 1,
                        static_cast<int>(angle / 360.0F * config_.angle_bins));
                    angle_bins[bin] = 1;
                }
            } else if (d2 < interior2) {
                ++interior_area;
                if (row[x] != 0) {
                    ++score.interior;
                }
            }
        }
    }

    cv::Mat bin_mat(1, config_.angle_bins, CV_8U, angle_bins.data());
    cv::dilate(bin_mat, bin_mat, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 1)));
    score.arc_bins = cv::countNonZero(bin_mat);
    if (score.arc_bins < config_.min_arc_bins || score.support < 14 || ring_area == 0) {
        return {};
    }

    const int start_bin = longestGapStart(angle_bins);
    int end_bin = start_bin;
    int steps = 0;
    while (steps < config_.angle_bins && angle_bins[end_bin] != 0) {
        end_bin = (end_bin + 1) % config_.angle_bins;
        ++steps;
    }
    score.angle_start = start_bin * 360.0F / config_.angle_bins;
    score.angle_end = end_bin * 360.0F / config_.angle_bins;

    const float support_density = static_cast<float>(score.support) / static_cast<float>(ring_area);
    const float coverage = static_cast<float>(score.arc_bins) / static_cast<float>(config_.angle_bins);
    const float interior_density = interior_area > 0 ? static_cast<float>(score.interior) / static_cast<float>(interior_area) : 0.0F;
    const float interior_penalty = 1.0F - clampFloat(interior_density * 4.0F, 0.0F, 0.82F);
    const float support_term = clampFloat(support_density * 4.0F, 0.0F, 1.0F);
    const float coverage_term = clampFloat(coverage * 2.4F, 0.0F, 1.0F);
    score.confidence = support_term * coverage_term * interior_penalty;
    return score;
}

}  // namespace rcj
