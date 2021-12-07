#ifndef IMG_H
#define IMG_H

#include <vector>
#include <iostream>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"


namespace ImgProcessing{
    
};


class Warp2TopDown{
    /* Compute a perspective transform M, given source and destination points
        Use a class to provide a warping method and remember the points and relevant info as attributes */
    private:
        // direct und inverse transform
        cv::Mat M_;  
        cv::Mat M_inv_;
        // reference points
        std::vector<cv::Point2f> pts_warpedROI_;
        std::vector<cv::Point2f> pts_ROI_;

    public:
        Warp2TopDown();
        cv::Mat warp(cv::Mat & frame_in);
        cv::Mat unwarp(cv::Mat & frame_in);
        // getters
        std::vector<cv::Point2f> xy_warpedROI();
        std::vector<cv::Point2f> xy_ROI();
};  

#endif