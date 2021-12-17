#include "lane.h"

// ---------------------------------
// III) Lane Class
// ---------------------------------
// implementation of lane coefficient fitting tool (across agreggate into lane parameters)
constexpr int LEFT = 0, RIGHT = 1, MIN = 0, MAX = 1;

void Lane::update_fit(std::vector<ImgProcessing::LaneLine> lane_fit_from_frame){
    a_[LEFT]  = lane_fit_from_frame[LEFT].poly_cfs[0];
    a_[RIGHT] = lane_fit_from_frame[RIGHT].poly_cfs[0];

    b_[LEFT]  = lane_fit_from_frame[LEFT].poly_cfs[1];
    b_[RIGHT] = lane_fit_from_frame[RIGHT].poly_cfs[1];

    c_[LEFT]  = lane_fit_from_frame[LEFT].poly_cfs[2]; 
    c_[RIGHT] = lane_fit_from_frame[RIGHT].poly_cfs[2];

    // score for weighted combination
    w_[LEFT] = std::pow(lane_fit_from_frame[LEFT].pts.size()  * lane_fit_from_frame[LEFT].MSE *
                       (lane_fit_from_frame[LEFT].y_max-lane_fit_from_frame[LEFT].y_min),2);
    w_[RIGHT] = std::pow(lane_fit_from_frame[RIGHT].pts.size() * lane_fit_from_frame[RIGHT].MSE * 
                        (lane_fit_from_frame[RIGHT].y_max-lane_fit_from_frame[RIGHT].y_min),2);

    // update frame size and axis, if necessary
    if (Nx_ != lane_fit_from_frame[LEFT].Nx || Ny_ !=  lane_fit_from_frame[LEFT].Ny){
        Nx_ = lane_fit_from_frame[LEFT].Nx;
        Ny_ = lane_fit_from_frame[LEFT].Ny;        
    }

    // update y range
    y_left_[MIN] = lane_fit_from_frame[LEFT].y_min;
    y_left_[MAX] = lane_fit_from_frame[LEFT].y_max;
    y_right_[MIN] = lane_fit_from_frame[RIGHT].y_min;
    y_right_[MAX] = lane_fit_from_frame[RIGHT].y_max;

    // y = 0 is further to the car, keep track of nearest x=f(max(y)), which is more reliable
    x_near_[LEFT]  = a_[LEFT]  + b_[LEFT] * Ny_ + c_[LEFT] * Ny_ * Ny_;
    x_near_[RIGHT] = a_[RIGHT] + b_[RIGHT]* Ny_ + c_[RIGHT]* Ny_ * Ny_;
    
    // std::cout << "NL(a, b, c) = (" << a_[LEFT] << "," << b_[LEFT]  << ", " << c_[LEFT] <<")\n";
    // std::cout << "NR(a, b, c) = (" << a_[RIGHT] << "," << b_[RIGHT]  << ", " << c_[RIGHT] <<")\n";
    // std::cout << "xn=["<< x_near_[LEFT] << ", " << x_near_[RIGHT] <<"]\n";

}

std::vector<int> Lane::yleft(){
    return std::move(std::vector<int>{y_left_[MIN], y_left_[MAX]});
}
std::vector<int> Lane::yright(){
    return std::move(std::vector<int>{y_right_[MIN], y_right_[MAX]});
}


std::vector<double> Lane::left_cfs(){
    std::vector<double> out{a_[LEFT], b_[LEFT], c_[LEFT]};
    return std::move(out);
}

std::vector<double> Lane::right_cfs(){
    std::vector<double> out{a_[RIGHT], b_[RIGHT], c_[RIGHT]};
    return std::move(out);
}

std::vector<double> Lane::center_cfs(){
    int idx_best = w_[LEFT] > w_[RIGHT] ? LEFT : RIGHT;
    std::vector<double> out{(a_[RIGHT]+a_[LEFT])/2, b_[idx_best], c_[idx_best]};
    return std::move(out);
}

std::vector<double> Lane::best_left_cfs(){
    int idx_best = w_[LEFT] > w_[RIGHT] ? LEFT : RIGHT;
    std::vector<double> out{a_[idx_best]-x_near_[idx_best]+x_near_[LEFT] , b_[idx_best], c_[idx_best]};
    return std::move(out);
}

std::vector<double> Lane::best_right_cfs(){
    int idx_best = w_[LEFT] > w_[RIGHT] ? LEFT : RIGHT;
    std::vector<double> out{a_[idx_best]-x_near_[idx_best]+x_near_[RIGHT] , b_[idx_best], c_[idx_best]};
    return std::move(out);
}

// auxiliar
void append_segment(std::vector<cv::Point> & line, const std::vector<cv::Point> & segment){
    line.insert(std::end(line), std::begin(segment), std::end(segment));
}

// form polygon
std::vector<std::vector<cv::Point>> Lane::getPolygon(){
    // y axis down and up in steps
    // logic: extrapolate only in reliable region
    std::vector<cv::Point> yl1 {cv::Point(0,0), cv::Point(0,y_left_[MIN]/2), cv::Point(0,y_left_[MIN])};
    std::vector<cv::Point> yl2 {cv::Point(0,y_left_[MIN]), cv::Point(0,(y_left_[MIN]+y_left_[MAX])/2), cv::Point(0,y_left_[MAX])};
    std::vector<cv::Point> yl3 {cv::Point(0,y_left_[MAX]), cv::Point(0,(y_left_[MAX]+Ny_)/2), cv::Point(0,Ny_)};
    
    std::vector<cv::Point> yr1 {cv::Point(0,Ny_), cv::Point(0,(y_right_[MAX]+Ny_)/2), cv::Point(0,y_right_[MAX])};
    std::vector<cv::Point> yr2 {cv::Point(0,y_right_[MAX]), cv::Point(0,(y_right_[MAX]+y_right_[MIN])/2), cv::Point(0,y_right_[MIN])};
    std::vector<cv::Point> yr3 {cv::Point(0,y_right_[MIN]), cv::Point(0,y_right_[MIN]/2), cv::Point(0,0)};
    
    // polyline to plot
    std::vector<cv::Point> line;
    append_segment(line, EvalFit<cv::Point>(this->best_left_cfs(), yl1, true) );
    append_segment(line, EvalFit<cv::Point>(this->left_cfs(), yl2, true) );
    append_segment(line, EvalFit<cv::Point>(this->best_left_cfs(), yl3, true) );
    append_segment(line, EvalFit<cv::Point>(this->best_right_cfs(), yr1, true) );
    append_segment(line, EvalFit<cv::Point>(this->right_cfs(), yr2, true) );
    append_segment(line, EvalFit<cv::Point>(this->best_right_cfs(), yr3, true) );
    
    std::vector<std::vector<cv::Point>> out{line};
    return std::move(out);
}

// ---------------------------------
// IV) Road Class
// ---------------------------------
// note: initializer list is necessary due to deleted BufferStats()
template class BufferStats<double>;
template class BufferStats<int>;

Road::Road(): n_buffer_(10), n_frames_(0),
              stats_left_(BufferStats<double>(10)),
              stats_right_(BufferStats<double>(10)),
              wlane_(BufferStats<int>(10))
{}

Road::Road(int n): n_buffer_(n), n_frames_(0), 
                   stats_left_(BufferStats<double>(n)),
                   stats_right_(BufferStats<double>(n)),
                   wlane_(BufferStats<int>(n))
{}

// define between-frame statistics
void Road::aggregate_frame_fit(Lane new_lane){

    // gathering first inputs
    if(n_frames_ < n_buffer_){

        if (stats_left_.has_NaN(new_lane.left_cfs()) || stats_left_.has_NaN(new_lane.right_cfs()))
            return; // do not add invalid points at start

        std::vector<int> lane_width{static_cast<int>(new_lane.x_near_[RIGHT] - new_lane.x_near_[LEFT])}; // make 1-element vector
        wlane_.add(lane_width);

        cf_buffer_left_.emplace(new_lane.left_cfs());
        cf_buffer_right_.emplace(new_lane.right_cfs());

        stats_left_.add(new_lane.left_cfs());
        stats_right_.add(new_lane.right_cfs());

        ++n_frames_;
        // update y range to the last frame
        y_left_[MIN] = new_lane.y_left_[MIN];
        y_left_[MAX] = new_lane.y_left_[MAX];
        y_right_[MIN] = new_lane.y_right_[MIN];
        y_right_[MAX] = new_lane.y_right_[MAX];

    // full buffer: assess pertinence and manage queue
    } else {
        // always gather non-outlier data, as lane width is constant
        std::vector<int> lane_width{static_cast<int>(new_lane.x_near_[RIGHT] - new_lane.x_near_[LEFT])}; // make 1-element vector
        if (!wlane_.is_outlier(lane_width)) wlane_.add(lane_width);
        
        // get more reliable estimate of local curvature by averaging inside the buffer
        // left lane marking
        if( !stats_left_.has_NaN(new_lane.left_cfs()) && !stats_left_.is_outlier(new_lane.left_cfs())){
            stats_left_.add(new_lane.left_cfs());
            stats_left_.remove(cf_buffer_left_.front());
            cf_buffer_left_.emplace(new_lane.left_cfs());
            cf_buffer_left_.pop();
            // update y range to the last fit
            y_left_[MIN] = new_lane.y_left_[MIN];
            y_left_[MAX] = new_lane.y_left_[MAX];
        }
        // right lane marking
        if(!stats_right_.has_NaN(new_lane.right_cfs()) && !stats_right_.is_outlier(new_lane.right_cfs())){
            stats_right_.add(new_lane.right_cfs());
            stats_right_.remove(cf_buffer_right_.front());
            cf_buffer_right_.emplace(new_lane.right_cfs());
            cf_buffer_right_.pop();    
            // update y range to the last fit
            y_right_[MIN] = new_lane.y_right_[MIN];
            y_right_[MAX] = new_lane.y_right_[MAX];
        }

    }

    // set own fields with treated coefficients ==> base for visualization
    std::vector<double> cf_l = stats_left_.mean();
    std::vector<double> cf_r = stats_right_.mean();

    a_[LEFT]  = cf_l[0];
    a_[RIGHT] = cf_r[0];

    b_[LEFT]  = cf_l[1];
    b_[RIGHT] = cf_r[1];

    c_[LEFT]  = cf_l[2]; 
    c_[RIGHT] = cf_r[2];

    w_[LEFT]  = new_lane.w_[LEFT];
    w_[RIGHT] = new_lane.w_[RIGHT];

    // update frame size and axis, if necessary
    if (Nx_ != new_lane.Nx_ || Ny_ !=  new_lane.Ny_){
        Nx_ = new_lane.Nx_;
        Ny_ = new_lane.Ny_;        
    }

    // y = 0 is further to the car, keep track of nearest x=f(max(y)), which is more reliable
    x_near_[LEFT]  = a_[LEFT]  + b_[LEFT] * Ny_ + c_[LEFT] * Ny_ * Ny_;
    x_near_[RIGHT] = a_[RIGHT] + b_[RIGHT]* Ny_ + c_[RIGHT]* Ny_ * Ny_;

    // std::cout << "ML(a, b, c) = (" << a_[LEFT] << "," << b_[LEFT]  << ", " << c_[LEFT] <<")\n";
    // std::cout << "MR(a, b, c) = (" << a_[RIGHT] << "," << b_[RIGHT]  << ", " << c_[RIGHT] <<")\n";
    // std::cout << "xn=["<< x_near_[LEFT] << ", " << x_near_[RIGHT] <<"]\n";

    // std::vector<double> vv1 = stats_left_.stddev();
    // if (!vv1.empty()) std::cout << "STDL(a, b, c) = (" << vv1[0] << "," << vv1[1]  << ", " << vv1[2] <<")\n";
    // std::vector<double> vv2 = stats_right_.stddev();
    // if (!vv2.empty())  std::cout << "STDR(a, b, c) = (" << vv2[0] << "," << vv2[1]  << ", " << vv2[2] <<")\n";
    
}