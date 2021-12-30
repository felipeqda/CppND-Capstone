#include "annotate.h"
#include "math_lib.h"

cv::Mat annotate::add_side_panel(cv::Mat & frame_in, double r_fwd, double r_rear){
    cv::Mat car = cv::imread("../data/car_top.png");
    cv::Size car_size = cv::Size(40, 90);
    // resize and zero-pad horizontally
    int panel_width = 200;
    cv::resize(car, car, car_size, 0.0, 0.0, cv::INTER_AREA);
    cv::Mat v_blank = cv::Mat::zeros(cv::Size(panel_width/2-car_size.width/2, car_size.height), frame_in.type());
    cv::hconcat(v_blank, car, car);
    cv::hconcat(car, v_blank, car);

    // zero-pad vertically
    cv::Mat h_blank = cv::Mat::zeros(cv::Size(panel_width, frame_in.size().height/2-car_size.height/2), frame_in.type());
    cv::Mat side_bar;
    cv::vconcat(h_blank, car, side_bar);
    cv::vconcat(side_bar, h_blank, side_bar);

    // draw curves
    std::vector<cv::Point> c1l, c1r, c2l, c2r;
    int Ny = frame_in.size().height;
    
    float px2m = 40/1.7/2;
    
    for(int i = 0; i < 100; ++i) {
        // curve ahead (in pixels): negative curvature to the left 
        if(r_fwd > 0){
            double ang = M_PI / 2.0 * (1 + static_cast<double>(i)/99); // pi/2 to pi   
            c1l.emplace_back(cv::Point(panel_width/2 - car_size.width + px2m * r_fwd * (1 + std::cos(ang)), Ny/2 - px2m* r_fwd * std::sin(ang)));
            c1r.emplace_back(cv::Point(panel_width/2 + car_size.width + px2m * r_fwd * (1 + std::cos(ang)), Ny/2 - px2m* r_fwd * std::sin(ang)));
        } else {
            double ang = M_PI / 2.0 * (static_cast<double>(i)/99); // 0 to pi/2   
            c1l.emplace_back(cv::Point(panel_width/2 - car_size.width - px2m * r_fwd * (-1 + std::cos(ang)), Ny/2 + px2m* r_fwd * std::sin(ang)));
            c1r.emplace_back(cv::Point(panel_width/2 + car_size.width - px2m * r_fwd * (-1 + std::cos(ang)), Ny/2 + px2m* r_fwd * std::sin(ang)));
        }
        
        // curve back (in pixels): negative curvature to the left
        if(r_rear > 0){
            double ang = M_PI / 2.0 * (2+static_cast<double>(i)/99); // pi to 3*pi/2   
            c2l.emplace_back(cv::Point(panel_width/2 - car_size.width + px2m * r_rear * (1 + std::cos(ang)), Ny/2 - px2m* r_rear * std::sin(ang)));
            c2r.emplace_back(cv::Point(panel_width/2 + car_size.width + px2m * r_rear * (1 + std::cos(ang)), Ny/2 - px2m* r_rear * std::sin(ang)));
        } else {
            double ang = M_PI / 2.0 * (-1 + static_cast<double>(i)/99); // -pi/2 to 0   
            c2l.emplace_back(cv::Point(panel_width/2 - car_size.width - px2m * r_rear * (-1 + std::cos(ang)), Ny/2 + px2m* r_rear * std::sin(ang)));
            c2r.emplace_back(cv::Point(panel_width/2 + car_size.width - px2m * r_rear * (-1 + std::cos(ang)), Ny/2 + px2m* r_rear * std::sin(ang)));
        }

    }
    cv::polylines(side_bar, c1l, false, cv::Scalar(255, 255,  0), 2);
    cv::polylines(side_bar, c1r, false, cv::Scalar(255, 255,  0), 2);
    cv::polylines(side_bar, c2l, false, cv::Scalar(255, 255, 80), 2);
    cv::polylines(side_bar, c2r, false, cv::Scalar(255, 255, 80), 2);
    
    // illustrate reference system
    // std::vector<cv::Point> ox{cv::Point(0,Ny/2), cv::Point(panel_width, Ny/2)}, oy{cv::Point(panel_width/2,0), cv::Point(panel_width/2, Ny)};
    // cv::polylines(side_bar, ox, false, cv::Scalar(0, 255,   0), 2);
    // cv::polylines(side_bar, oy, false, cv::Scalar(0, 0,   255), 2);

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


void annotate::annotate_objs(cv::Mat & frame, const std::vector<Detection> & objs, cv::Scalar color){
    for (size_t idx = 0; idx < objs.size(); ++idx) {
        cv::rectangle(frame, objs[idx].box, color, 2);
    }
}