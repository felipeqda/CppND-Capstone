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

// overloads for convinience win using post-treated coeffs
constexpr int MIN=0, MAX=1;
void annotate::annotate_unwarpedlanes(Lane & lanes, Warp2TopDown & transform, 
                                      cv::Mat & frame, cv::Scalar color){
   
    std::vector<int> yl = lanes.yleft();
    std::vector<int> yr = lanes.yright();

    // form polylines and plot
    // 1) left lane
    std::vector<cv::Point> y{cv::Point(0,yl[MIN]), cv::Point(0,(yl[MIN]+yl[MAX])/2),
                             cv::Point(0,yl[MAX])};
    std::vector<cv::Point> line = EvalFit<cv::Point>(lanes.left_cfs(), y, true);  // only pts.y is used
    line = transform.unwarp_path(line);
    cv::polylines(frame, line, false, color, 2);
    //2 ) right lane
    y.clear();
    y.insert(y.end(), {cv::Point(0,yr[MIN]), cv::Point(0,(yr[MIN]+yr[MAX])/2), cv::Point(0,yr[MAX])});
    line = EvalFit<cv::Point>(lanes.right_cfs(), y, true);  // only pts.y is used
    line = transform.unwarp_path(line);
    cv::polylines(frame, line, false, color, 2);    
}
// version for road (more reliable coefficients)
void annotate::annotate_unwarpedlanes(Road & road, Warp2TopDown & transform, 
                                      cv::Mat & frame, cv::Scalar color){
   
    std::vector<int> yl = road.yleft();
    std::vector<int> yr = road.yright();
    int ymax = yr[MAX] > yl[MAX] ? yr[MAX] : yl[MAX];
    int ymin = yr[MIN] < yl[MIN] ? yr[MIN] : yl[MIN];

    // form polylines and plot
    // axis
    std::vector<cv::Point> y{cv::Point(0,ymin), cv::Point(0,(ymin+ymax)/2),
                             cv::Point(0,ymax)};
    // 1) left lane
    std::vector<cv::Point> line = EvalFit<cv::Point>(road.left_cfs(), y, true);  // only pts.y is used
    line = transform.unwarp_path(line);
    cv::polylines(frame, line, false, color, 2);
    //2 ) right lane
    line = EvalFit<cv::Point>(road.right_cfs(), y, true);  // only pts.y is used
    line = transform.unwarp_path(line);
    cv::polylines(frame, line, false, color, 2);    
}