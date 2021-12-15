#ifndef IMG_H
#define IMG_H

#include <vector>
#include <queue>
#include <algorithm>
#include <iostream>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

#include "math.h"


namespace ImgProcessing{
    enum class orientation {x, y};

    struct LaneLine{
        std::vector<cv::Point> pts;
        std::vector<double> poly_cfs;
        double MSE;
        int Nx, Ny;
        int y_min, y_max;
    };

    cv::Mat get_lane_limits_mask(cv::Mat & frame_in);
    cv::Mat mask_frame(cv::Mat & frame, cv::Mat & mask);
    cv::Mat threshold_between(cv::Mat & frame, int thresh_min, int threshold_max);
    cv::Mat abs_sobel_thresh(cv::Mat frame, orientation dir, 
                             int thresh_min, int threshold_max, int ksize);
    std::vector<LaneLine> fit_xy_from_mask(cv::Mat & mask, cv::Mat & frame, 
                                           int n_windows=9, int margin=30, size_t minpix=50, bool annotate=true);
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
        // getters
        std::vector<cv::Point2f> xy_warpedROI();
        std::vector<cv::Point2f> xy_ROI();
};  

// store information of left and right lined in a frame, which characterizes the lane
class Lane{    
    private:
        int Nx_, Ny_;
        // polynomial coefficients of order 0-2 for left and right lanes
        double  a_[2], b_[2], c_[2], w_[2];
        // range of y for fits
        int y_left_[2]  = {0,0};
        int y_right_[2] = {0,0};

    public:
        Lane() = default;        
        void update_fit(std::vector<ImgProcessing::LaneLine> lane_fit_frame);
        std::vector<double> right_cfs();
        std::vector<double> left_cfs();
        std::vector<double> center_cfs();
        std::vector<double> best_right_cfs();
        std::vector<double> best_left_cfs();
        std::vector<cv::Point> getPolygon();
};


// integrate lane information across frames, adding stats and a buffer to stabilize against outliers
class Road : public Lane{
    private:
        int n_buffer;
        std::queue<std::vector<double>> cf_buffer_left;
        std::queue<std::vector<double>> cf_buffer_right;

        BufferStats<double> stats_left;
        BufferStats<double> stats_right;
        BufferStats<int> dx_;  // get typical values for distance between lanes
    
    public:
        Road() = default;
        Road(int n);
        void aggregate_frame_fit(Lane new_lane);
};

Road::Road(int n): n_buffer(n) {}

#endif