#ifndef LANE_H
#define LANE_H

#include <vector>
#include <queue>
#include "opencv2/core.hpp"
#include "math.h"
#include "img_processing.h"

// store information of left and right lined in a frame, which characterizes the lane
class Road;
class Lane{    
    private:
        int Nx_, Ny_;
        // polynomial coefficients of order 0-2 for left and right lanes
        double  a_[2], b_[2], c_[2], w_[2], x_near_[2];
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
        std::vector<int> yleft();
        std::vector<int> yright();
        std::vector<std::vector<cv::Point>> getPolygon();
    
    friend class Road;
};

// integrate lane information across frames, adding stats and a buffer to stabilize against outliers
class Road : public Lane{
    private:
        int n_buffer_, n_frames_;
        std::queue<std::vector<double>> cf_buffer_left_;
        std::queue<std::vector<double>> cf_buffer_right_;

        BufferStats<double> stats_left_;
        BufferStats<double> stats_right_;
        BufferStats<int> wlane_;  // get typical values for distance between lanes
    
    public:
        Road();
        Road(int n);
        void aggregate_frame_fit(Lane new_lane);
};

#endif