#include "annotate.h"
#include "math.h"

cv::Mat annotate::add_side_panel(cv::Mat & frame_in){
    cv::Mat car = cv::imread("../data/car_top.png");
    cv::Size car_size = cv::Size(40, 90);
    // resize and zero-pad horizontally
    int panel_width = 200;
    cv::resize(car, car, car_size, 0.0, 0.0, cv::INTER_AREA);
    cv::Mat v_blank = cv::Mat::zeros(cv::Size(panel_width/2-car_size.width/2, car_size.height), frame_in.type());
    cv::hconcat(v_blank, car, car);
    cv::hconcat(car, v_blank, car);

    // zero-pad vertically
    cv::Mat h_blank =  cv::Mat::zeros(cv::Size(panel_width, frame_in.size().height/2-car_size.height/2), frame_in.type());
    cv::Mat side_bar;
    cv::vconcat(h_blank, car, side_bar);
    cv::vconcat(side_bar, h_blank, side_bar);

    // add to frame
    cv::Mat frame_out;
    cv::hconcat(frame_in, side_bar, frame_out);

    return frame_out;
}    


void annotate::annotate_lanes(std::vector<ImgProcessing::LaneLine> & lanes, cv::Mat & frame, cv::Scalar color){
    for(auto lane : lanes){
        // form polylines and plot
        std::vector<cv::Point> y{cv::Point(0,lane.y_min), cv::Point(0,(lane.y_min+lane.y_max)/2),
                                 cv::Point(0,lane.y_max)};
        std::vector<cv::Point> line = EvalFit<cv::Point>(lane.poly_cfs, y, true);  // only pts.y is used
        cv::polylines(frame, line, false, color, 2);
    }
}

void annotate::annotate_unwarpedlanes(std::vector<ImgProcessing::LaneLine> & lanes, Warp2TopDown & transform, 
                                      cv::Mat & frame, cv::Scalar color){
    for(auto lane : lanes){
        // form polylines and plot
        std::vector<cv::Point> y{cv::Point(0,lane.y_min), cv::Point(0,(lane.y_min+lane.y_max)/2),
                                 cv::Point(0,lane.y_max)};
        std::vector<cv::Point> line = EvalFit<cv::Point>(lane.poly_cfs, y, true);  // only pts.y is used
        line = transform.unwarp_path(line);
        cv::polylines(frame, line, false, color, 2);
    }
}