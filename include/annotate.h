#ifndef ANN_H
#define ANN_H

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "img_processing.h"
#include "lane.h"
#include "neural_network.h"

namespace annotate{
    cv::Mat add_side_panel(cv::Mat & frame_in, double r_fwd, double r_bwd);
    // annotate based on frame info
    void annotate_lanes(std::vector<ImgProcessing::LaneLine> & lanes, cv::Mat & frame, cv::Scalar color = cv::Scalar(255,255,128));
    void annotate_unwarpedlanes(std::vector<ImgProcessing::LaneLine> & lanes , Warp2TopDown & transform, cv::Mat & frame, cv::Scalar color = cv::Scalar(255,255,128));
    void annotate_unwarpedlanes(Lane & lanes, Warp2TopDown & transform, cv::Mat & frame, cv::Scalar color = cv::Scalar(255,255,128));
    // annotate based on road (buffer of frames) info
    void annotate_unwarpedlanes(Road & road, Warp2TopDown & transform, cv::Mat & frame, cv::Scalar color = cv::Scalar(255,255,128));

    void annotate_objs(cv::Mat & frame, const std::vector<Detection> & objs, cv::Scalar color = cv::Scalar(0, 0, 255));
};

#endif