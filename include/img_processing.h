#ifndef IMG_H
#define IMG_H

#include <vector>
#include <algorithm>
#include <iostream>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"



namespace ImgProcessing{
    enum class orientation {x, y};

    struct LaneLine{
        std::vector<cv::Point> pts;
        std::vector<double> poly_cfs;
        double MSE;
        int Nx, Ny;
        int y_min, y_max;
    };

    cv::Mat get_lane_limits_mask(cv::Mat frame_in);
    cv::Mat mask_frame(cv::Mat & frame, cv::Mat & mask);
    cv::Mat threshold_between(cv::Mat & frame, int thresh_min, int threshold_max);
    cv::Mat abs_sobel_thresh(cv::Mat frame, orientation dir, 
                             int thresh_min, int threshold_max, int ksize);
    std::vector<LaneLine> fit_xy_from_mask(cv::Mat & mask, cv::Mat & frame, 
                                           int n_windows=9, int margin=30, size_t minpix=50, bool annotate=false);
}

// Compute a perspective transform M, given source and destination points
// Use a class to provide a warping method and remember the points and relevant info as attributes
class Warp2TopDown{
    private:
        // direct und inverse transform
        cv::Mat M_;  
        cv::Mat M_inv_;
        // reference points
        std::vector<cv::Point2f> pts_warpedROI_;
        std::vector<cv::Point2f> pts_ROI_;
        float frame_reduction;
        // auxiliary for diagnostic/tests
        void show_warp_area_img(cv::Mat & frame_in);
        void show_warp_area_warpedimg(cv::Mat & frame_in);

    public:
        Warp2TopDown() = default;
        Warp2TopDown(float img_reduction);
        cv::Mat warp(cv::Mat & frame_in);
        cv::Mat unwarp(cv::Mat & frame_in);
        std::vector<cv::Point> warp_path(std::vector<cv::Point> path);
        std::vector<cv::Point> unwarp_path(std::vector<cv::Point> path);
        // getters
        std::vector<cv::Point2f> xy_warpedROI();
        std::vector<cv::Point2f> xy_ROI();
};  

#endif