#include "img_processing.h"
#include "math.h"

// auxiliary dot-product operation:
// https://stackoverflow.com/questions/27981214/opencv-how-do-i-multiply-point-and-matrix-cvmat
cv::Point2f operator*(cv::Mat M, const cv::Point2f& p){ 
    cv::Mat_<double> src(3/*rows*/,1 /* cols */); 
    src(0,0)=p.x; 
    src(1,0)=p.y; 
    src(2,0)=1.0; 
    cv::Mat_<double> dst = M*src; //USE MATRIX ALGEBRA 
    return cv::Point2f(dst(0,0),dst(1,0)); 
} 

//cf. https://docs.opencv.org/3.4/de/dd4/samples_2cpp_2warpPerspective_demo_8cpp-example.html#a26
// constructor
Warp2TopDown::Warp2TopDown(float reduction=1.0){
    // based on a Ny x Nx = 720 x 1280 image (straight_lines1/2.jpg)
    std::vector<cv::Point2f> pts_img {{191*reduction, 720*reduction}, 
                                      {601*reduction, 445*reduction}, 
                                      {678*reduction, 445*reduction}, 
                                      {1118*reduction, 720*reduction}};
    std::vector<cv::Point2f> pts_warp{{350*reduction, 720*reduction}, 
                                      {350*reduction, 0}, 
                                      {950*reduction, 0}, 
                                      {950*reduction, 720*reduction}};
    // store direct and inverse transform matrices
    M_= cv::getPerspectiveTransform(pts_img, pts_warp);
    M_inv_ = cv::getPerspectiveTransform(pts_warp, pts_img);
    frame_reduction = reduction;

    /* define ROI for spatial filtering using inverse transform
    cf. equations at
    https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#getperspectivetransform
    auxiliar 3D coordinates are used with z =1 for source and a normalization factor t for the destination
    transposition is done to facilitate matrix product
    take the warp region as a reference and expand, to get a rectangle in the top-down view representing relevant
    search area */
    std::vector<cv::Point2f> pts_warpedROI_ = {{150*reduction, 720*reduction}, 
                                               {150*reduction, 0}, 
                                               {250*reduction, 0}, 
                                               {250*reduction, 720*reduction}};
    std::vector<cv::Point2f> pts_ROI_;    
    for(cv::Point2f pt_w : pts_warpedROI_){
        pts_ROI_.emplace_back(this->M_inv_ * pt_w);
    }

    // file_matrix = (cv::Mat_<int>(3, 3) << 1, 2, 3,
    //                                   3, 4, 6,
    //                                   7, 8, 9);     
}    

// warping tools
cv::Mat Warp2TopDown::warp(cv::Mat & frame_in){
    // Warp an image using the perspective transform, M
    cv::Mat warped_frame;    
    cv::warpPerspective(frame_in, warped_frame, this->M_, frame_in.size());   
    // show_warp_area_warpedimg(warped_frame);
    return warped_frame;
}
cv::Mat Warp2TopDown::unwarp(cv::Mat & frame_in){
    //Inverse perspective transform, M_inv
    cv::Mat unwarped_frame;
    cv::warpPerspective(frame_in, unwarped_frame, this->M_inv_, frame_in.size());
    return unwarped_frame;    
}

// getters
std::vector<cv::Point2f> Warp2TopDown::xy_warpedROI(){
    return pts_warpedROI_;
}
std::vector<cv::Point2f> Warp2TopDown::xy_ROI(){
    return pts_ROI_;
}

// auxiliary functions
void Warp2TopDown::show_warp_area_img(cv::Mat & frame_in){
    float reduction = frame_reduction;
    std::vector<cv::Point> pts_img {{static_cast<int>(191*reduction), static_cast<int>(720*reduction)}, 
                                    {static_cast<int>(601*reduction), static_cast<int>(445*reduction)}, 
                                    {static_cast<int>(678*reduction), static_cast<int>(445*reduction)}, 
                                    {static_cast<int>(1118*reduction),static_cast<int>(720*reduction)}};
    std::vector<cv::Point> pts_warp{{static_cast<int>(350*reduction), static_cast<int>(720*reduction)}, 
                                    {static_cast<int>(350*reduction), 0}, 
                                    {static_cast<int>(950*reduction), 0}, 
                                    {static_cast<int>(950*reduction), static_cast<int>(720*reduction)}};
    cv::polylines(frame_in, pts_img, true, cv::Scalar(0,0,255));
}
void Warp2TopDown::show_warp_area_warpedimg(cv::Mat & frame_in){
    float reduction = frame_reduction;
    std::vector<cv::Point> pts_warp{{static_cast<int>(350*reduction), static_cast<int>(720*reduction)}, 
                                    {static_cast<int>(350*reduction), 0}, 
                                    {static_cast<int>(950*reduction), 0}, 
                                    {static_cast<int>(950*reduction), static_cast<int>(720*reduction)}};
    cv::polylines(frame_in, pts_warp, true, cv::Scalar(255,0,0));
}


cv::Mat ImgProcessing::mask_frame(cv::Mat & frame, cv::Mat & mask){
    // Mask out frame from mask
    cv::Mat frame_out;
    frame.copyTo(frame_out, mask);
    return std::move(frame_out);
}



cv::Mat ImgProcessing::threshold_between(cv::Mat & frame, int thresh_min, int threshold_max){
    cv::Mat m1, m2;
    // void threshold( src_gray, dst, threshold_value, max_binary_value, threshold_type )
    cv::threshold(frame, m1, thresh_min,    1, cv::THRESH_BINARY);     // GT X  ==> mask = 1
    cv::threshold(frame, m2, threshold_max, 1, cv::THRESH_BINARY_INV); // LT X  ==> mask = 1
    cv::bitwise_and(m1, m2, m1); // replace m1 with m1 && m2 (note nonzero is true in this case)
    return std::move(m1);
}


cv::Mat ImgProcessing::abs_sobel_thresh(cv::Mat frame, orientation dir, 
                                        int thresh_min, int threshold_max, int ksize=3){
    /* Define a function that applies Sobel x or y,
       then takes an absolute value and applies a threshold.
       cf. https://docs.opencv.org/3.4/d2/d2c/tutorial_sobel_derivatives.html */

    // Remove noise by blurring with a Gaussian filter ( kernel size = 3 )
    cv::GaussianBlur(frame, frame, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);
    
    cv::Mat grad;
    cv::Mat abs_grad;
    if(dir == orientation::x){
        cv::Sobel(frame, grad, CV_16S, 1, 0, ksize);
    } else {
        cv::Sobel(frame, grad, CV_16S, 0, 1, ksize);
    }

    // converting back to CV_8U
    cv::convertScaleAbs(grad, abs_grad);    
    // for a 2D gradient
    // cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);   
    return std::move(ImgProcessing::threshold_between(abs_grad, thresh_min, threshold_max));
}


cv::Mat ImgProcessing::get_lane_limits_mask(cv::Mat & frame_in_RGB){
    /*  Take RGB image, perform necessary color transformation /gradient calculations
        and output the detected lane pixels mask, alongside an RGB composition of the 3 sub-masks (added)
        for visualization */

    // get morphological kernel and apply pre-processing
    // https://stackoverflow.com/questions/15561863/fast-image-thresholding?rq=1
    // TODO: test and remove if not useful
    bool MORPH_ENHANCE = true;
    if(MORPH_ENHANCE){
        cv::Mat temp_img;
        cv::resize(frame_in_RGB, temp_img, cv::Size(), 0.5, 0.5, cv::INTER_AREA); 
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
        cv::morphologyEx(temp_img, temp_img, cv::MORPH_CLOSE, kernel); //, iterations=2);
        cv::resize(temp_img, temp_img, cv::Size(), 2.0, 2.0, cv::INTER_AREA);  
        frame_in_RGB = temp_img;
    }

    // convert to HLS color space
    cv::Mat frame_HLS;
    cv::cvtColor(frame_in_RGB, frame_HLS, cv::COLOR_RGB2HLS);
    // 3 output channels = (H, L, S)    
    cv::Mat HLS_chs[3], GBR_chs[3];
    cv::split(frame_HLS, HLS_chs);

    // auxiliary masks
    cv::Mat mask, grd_mask, shadow_mask;
    

    // 1) mask for high S value
    // cf. https://docs.opencv.org/3.4/d7/d1b/group__imgproc__misc.html#gaa9e58d2860d4afa658ef70a9b1115576
    // void threshold( src_gray, dst, threshold_value, max_binary_value, threshold_type );
    cv::threshold(HLS_chs[2], mask, 150, 1, cv::THRESH_BINARY);    // GT X  ==> mask = 1    

    // 2) high S x-gradient
    grd_mask = abs_sobel_thresh(HLS_chs[2], ImgProcessing::orientation::x, 20, 100);
    cv::bitwise_or(mask, grd_mask, mask);

    // 3) high x-gradient of grayscale image 
    cv::Mat frame_gray;
    cv::cvtColor(frame_in_RGB, frame_gray, cv::COLOR_RGB2GRAY);
    grd_mask = abs_sobel_thresh(frame_gray, ImgProcessing::orientation::x, 20, 100);
    cv::bitwise_or(mask, grd_mask, mask);

    // 4) shadow mask
    cv::threshold(HLS_chs[2], shadow_mask, 50, 1, cv::THRESH_BINARY);    // GT X  ==> mask = 1 
    cv::bitwise_and(mask, shadow_mask, mask); // replace m1 with m1 && m2 (note nonzero is true in this case) 

    return mask; 
}

std::vector<ImgProcessing::LaneLine> ImgProcessing::fit_xy_from_mask(cv::Mat & mask, cv::Mat & frame,
                                     int n_windows, int margin, size_t minpix, bool annotate){
    /* Take the input mask and perform a sliding window search
     Return the coordinates of the located pixels, polynomial coefficients and optionally an image showing the
       windows/detections
     **Parameters/Keywords:
     n_windows ==> Choose the number of sliding windows
     margin ==> Set the width of the windows +/- margin
     minpix ==> Set minimum number of pixels found to recenter window*/


    // treatment of mask to get more reliable region
    // cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));    
    // cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel); 

    /*
    // perform watershed detection (labels connected components with unique number
    cv::Mat labels;
    int n_labels = cv::connectedComponents(mask, labels, CV_16U);
    //  get indices which belong to each of the reliable clusters (will not be split by the margin)
    //  bins_map [j-1] contains lin/col = yx = [0,1] indices of connected region j ([2, Npts])
    //  take indices of nonzero (defined in line 267 below) so that the indices refer to the same mask
    std::vector<std::vector<cv::Point>> region_idx_map;
    for(int i = 0; i < n_labels; ++i)        
        region_idx_map.emplace_back(std::vector<int>());
    // https://stackoverflow.com/questions/25221421/c-opencv-fast-pixel-iteration/25224916
    for (int i = 0; i < labels.rows; ++i){
        uchar* pixel = labels.ptr<uchar>(i); // pointer to first pixel in row
        for (int j = 0; j < labels.cols; ++j) {
            if (pixel[j] > 0)  // values [0, n_labels-1]
                region_idx_map[px-1].emplace_back(cv::Point(i, j));
        }
    }*/

    // Take a "histogram" (x-profile) of the bottom of the image
    int Ny = mask.rows;
    int Nx = mask.cols;  
    std::cout << "Nx, Ny = " << Nx <<"," << Ny << "\n";
    std::vector<float> histogram(Nx); // inits to zero
    for (int i = Ny - 3 * Ny / n_windows; i < Ny; ++i){
        uchar* pixel = mask.ptr<uchar>(i); // pointer to first pixel in row
        for (int j = 0; j < Nx; ++j) {            
            histogram[j] += static_cast<float>(pixel[j]);
        }
    }    

    // Find the peak of the left and right halves of the histogram
    // These will be the starting point for the left and right lines
    // Consider a margin to avoid locating lanes too close to border    
    int midpoint = Nx/2, leftx_base, rightx_base;
    std::vector<float>::iterator it; 
    it  = std::max_element(histogram.begin()+Nx/10, histogram.begin()+midpoint);
    leftx_base = std::distance(histogram.begin(), it);
    std::cout << "start x: " << leftx_base << "("<< *it <<")";
    it = std::max_element(histogram.begin()+midpoint, histogram.end()-Nx/10); 
    rightx_base = std::distance(histogram.begin(), it);
    std::cout << ", "  << rightx_base << "("<< *it <<")\n";

    //Set height of windows - based on n_windows above and image shape
    int window_height = Ny / n_windows;
    // Identify the x and y positions of all nonzero pixels in the image
    std::vector<cv::Point> nonzero;
    for (int i = 0; i < Ny; ++i){
        uchar* pixel = mask.ptr<uchar>(i); // pointer to first pixel in row
        for (int j = 0; j < Nx; ++j) {
            if ( pixel[j] > 0) { 
                nonzero.emplace_back(cv::Point(j, i));
            }
        }
    }

    // Current positions to be updated later for each window in n_windows
    int leftx_current, rightx_current;
    leftx_current = leftx_base;
    rightx_current = rightx_base;

    // Create empty lists to receive left and right lane pixel indices/coordinates (first is x)
    //inds ==> refer to the image, not nonzero!
    std::vector<cv::Point> left_lane_pts, right_lane_pts;

    // keep track of the minimum y of the reliable (label-based) regions
    /*int ymin_good_left = Ny;
    int ymin_good_right = Ny; */

    std::cout << "#####################################################\n";

    // Step through the windows one by one
    for (int win = 0; win < n_windows; ++win){

        // Identify window boundaries in x and y (and right and left)
        int win_y_low = Ny - (win + 1) * window_height;
        int win_y_high = Ny - win * window_height;
        // Find the four boundaries of the window #win
        int win_xleft_low   = leftx_current - margin;
        int win_xleft_high  = leftx_current + margin;
        int win_xright_low  = rightx_current - margin;
        int win_xright_high = rightx_current + margin;

        std::cout << "win #" << win << "  y = [" << win_y_low << ", " << win_y_high << "]\n"
                  << "  x = [" << win_xleft_low << ", " << win_xleft_high << "] [" << win_xright_low << ", " << win_xright_high <<"]\n";

        // Identify the nonzero pixels in x and y within the window #win
        // take indices of the non-zero selection!
        auto within_left  = [win_xleft_low, win_xleft_high, win_y_low, win_y_high](cv::Point p){            
            return (p.x >= win_xleft_low) && (p.x <= win_xleft_high) && 
                   (p.y <= win_y_low)     && (p.y <= win_y_high) ;
        };
        auto within_right = [win_xright_low, win_xright_high, win_y_low, win_y_high](cv::Point p){
            return (p.x >= win_xright_low) && (p.x <= win_xright_high) && 
                   (p.y <= win_y_low)      && (p.y <= win_y_high) ;
        };

        // Append these coordinates to the lists: go though points only once
        for(auto p: nonzero){
            if(within_left(p)) 
                left_lane_pts.emplace_back(p);
            if(within_right(p)) 
                right_lane_pts.emplace_back(p);
        }

        // annotation to illustrate parameters
        if (annotate){
            cv::rectangle(frame, cv::Point(win_xleft_low, win_y_low), cv::Point(win_xleft_high, win_y_high), cv::Scalar(255,0,0));
            cv::rectangle(frame, cv::Point(win_xright_low, win_y_low), cv::Point(win_xright_high, win_y_high), cv::Scalar(255,255,0));
        }
        
        // Update window's x center
        // left window
        // perform a fit to predict the window tendency, if a minimum number of points is present and the y span allows
        /*const auto [imin, imax] = std::minmax_element(left_lane_pts.begin(), left_lane_pts.end(), 
                                  [](const cv::Point & p1, const cv::Point & p2){ return p1.x > p2.x; }
                                  );
        if (left_lane_pts.size() > minpix  && ((*imax).y-(*imin).y) > static_cast<int>(minpix) ){*/
        if (left_lane_pts.size() > minpix ){
            // make partial fit of order 1 or 2 and predict position at next window
            if(win < 3){
                std::vector<double>cfs = FitLine(left_lane_pts, true); // fit x = f(y)
                int y = 0.5 * (win_y_low + win_y_high)-window_height;
                leftx_current = cfs[0] + cfs[1]*y;
            } else {
                std::vector<double>cfs = FitParabola(left_lane_pts, true); // fit x = f(y)
                int y = 0.5 * (win_y_low + win_y_high)-window_height;
                leftx_current = cfs[0] + cfs[1]*y + cfs[2]*y*y;
            }
        }

        // right window
        // perform a fit to predict the window tendency, if a minimum number of points is present and the y span allows
        const auto [imin2, imax2] = std::minmax_element(right_lane_pts.begin(), right_lane_pts.end(), 
                                    [](const cv::Point & p1, const cv::Point & p2){ return p1.x > p2.x; }
                                    );
        if (right_lane_pts.size() > minpix  && ((*imax2).y-(*imin2).y) > static_cast<int>(minpix) ){
            // make partial fit of order 1 or 2 and predict position at next window
            if(win < 3){
                std::vector<double>cfs = FitLine(right_lane_pts, true); // fit x = f(y)
                int y = 0.5 * (win_y_low + win_y_high)-window_height;
                rightx_current = static_cast<int>(cfs[0] + cfs[1]*y);
            } else {
                std::vector<double>cfs = FitParabola(right_lane_pts, true); // fit x = f(y)
                int y = 0.5 * (win_y_low + win_y_high)-window_height;
                rightx_current = static_cast<int>(cfs[0] + cfs[1]*y + cfs[2]*y*y);
            }
        }
    } // for each window


    // Fit a second order polynomial to each lane, assuming x = f(y) #
    // for info on residuals ==> https: // stackoverflow.com / questions / 5477359 / chi - square - numpy - polyfit - numpy
    std::vector<double>polycf_left = FitParabola(left_lane_pts, true); // fit x = f(y)
    double sqr_error_left = chi_squared(polycf_left, left_lane_pts, true);
    double MSE_left = std::sqrt(sqr_error_left / left_lane_pts.size());

    std::vector<double>polycf_right = FitParabola(right_lane_pts, true); // fit x = f(y)
    double sqr_error_right = chi_squared(polycf_right, right_lane_pts, true);
    double MSE_right = std::sqrt(sqr_error_right / right_lane_pts.size());

    
    /*# Lane annotation (to be warped and shown in pipeline)
    lane_annotation = np.zeros([Ny, Nx, 3], dtype=np.uint8)
    lane_annotation[lefty, leftx] = [255, 0, 0]
    lane_annotation[righty, rightx] = [0, 0, 255]*/


    // wrap data into LaneLines structs
    ImgProcessing::LaneLine leftlane  = LaneLine{std::move(left_lane_pts),  std::move(polycf_left),  MSE_left};
    ImgProcessing::LaneLine rightlane = LaneLine{std::move(right_lane_pts), std::move(polycf_right), MSE_right};
    std::vector<LaneLine> out_lanes {leftlane, rightlane};
    return std::move(out_lanes);

}

/*
# functions to process the mask by finding lane pixels and fitting
# (based on quizzes)
# ---------------------------------------------------------------------
class LaneLine:
    """store the coordinates of the pixels and the polynomial coefficients for each lane
     also store where the line reaches the bottom of the image """

    def __init__(self, x_coord, y_coord, poly_coef, MSE, y_min_reliable):
        self.Npix = np.size(x_coord) #size of fitted region
        self.x_pix = x_coord
        self.y_pix = y_coord
        self.cf = poly_coef
        if np.size(y_coord) > 0:
            self.x_bottom = np.polyval(poly_coef, np.max(y_coord))
        else:
            self.x_bottom = None
        # MSE of fit (compare "goodness of fit")
        self.MSE = MSE
        if np.isnan(y_min_reliable):
            self.y_min_reliable = None #neutral indexing value
        else:
            self.y_min_reliable = np.int32(y_min_reliable)
# ---------------------------------------------------------------------
def weight_fit_cfs(left, right):
    """ judge fit quality, providing weights and a weighted average of the coefficients
        inputs are LaneLine objects """
    cfs = np.vstack((left.cf, right.cf))
    cf_MSE = np.vstack((left.MSE, right.MSE))
    # average a/b coefficients with inverse-MSE weights
    w1 = np.sum(cf_MSE) / cf_MSE
    # consider number of points as well
    w2 = np.reshape(np.array([left.Npix, right.Npix]) / (left.Npix + right.Npix), [2, 1])
    # aggregate weights
    w = w1 * w2
    cf_avg = np.mean(w * cfs, axis=0) / np.mean(w, axis=0)
    return w, cf_avg
# ---------------------------------------------------------------------

*/
