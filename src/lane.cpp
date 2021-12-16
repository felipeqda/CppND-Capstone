#include "lane.h"

// ---------------------------------
// III) Lane Class
// ---------------------------------
// implementation of lane coefficient fitting tool (across agreggate into lane parameters)
constexpr int LEFT = 0, RIGHT = 1, MIN = 0, MAX = 0;

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

Road::Road(): n_buffer_(10), 
              stats_left_(BufferStats<double>(10)),
              stats_right_(BufferStats<double>(10)),
              dx_(BufferStats<int>(10))
{}

Road::Road(int n): n_buffer_(n), 
                   stats_left_(BufferStats<double>(n)),
                   stats_right_(BufferStats<double>(n)),
                   dx_(BufferStats<int>(n))
{}

