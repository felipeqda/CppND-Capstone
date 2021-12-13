#include "img_processing.h"


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


cv::Mat ImgProcessing::mask_lane(cv::Mat & frame_in_RGB){
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

    // Output frame
    cv::Mat frame_out;
    frame_in_RGB.copyTo(frame_out, mask);  
    return frame_out;    
}

cv::Mat ImgProcessing::fit_xy_from_mask(cv::Mat & frame_in, int n_windoes=9, int margin = 100, int minpix = 50){
    /* Take the input mask and perform a sliding window search
     Return the coordinates of the located pixels, polynomial coefficients and optionally an image showing the
       windows/detections
     **Parameters/Keywords:
     nwindows ==> Choose the number of sliding windows
     margin ==> Set the width of the windows +/- margin
     minpix ==> Set minimum number of pixels found to recenter window*/


}


    // treatment of mask to get more reliable region
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::morphologyEx(frame_in, frame_in, cv::MORPH_OPEN, kernel); ;

    // perform watershed detection (labels connected components with unique number
    cv::Mat labels;
    int n_labels = cv::connectedComponents(frame_in, labels, CV_16U);
    /*  get indices which belong to each of the reliable clusters (will not be split by the margin)
        bins_map [j-1] contains lin/col = yx = [0,1] indices of connected region j ([2, Npts])
        take indices of nonzero (defined in line 267 below) so that the indices refer to the same mask */
    std::vector<std::vector<cv::Point>> region_idx_map;
    CV_16U px;
    for(int i = 0; i < n_labels; ++i)        
        region_idx_map.emplace_back(std::vector<int>());
    // https://stackoverflow.com/questions/25221421/c-opencv-fast-pixel-iteration/25224916
    for (int i = 0; i < labels.rows; ++i){
        cv::Vec3b* pixel = labels.ptr<cv::Vec3b>(i); // pointer to first pixel in row
        for (int j = 0; j < img.cols; ++j) {
            px = pixel[j][0]; // values (0 - n_labels-1)
            if (px > 0)
                region_idx_map[px-1].emplace_back(cv::Point(i, j));
        }
    }
        std::vector
    region_idx_map = [(labels == j).nonzero() for j in range(1, np.max(labels) + 1)]

    # Take a "histogram" (x-profile) of the bottom half of the image
    histogram = np.sum(mask_input[mask_input.shape[0] // 2:, :], axis=0)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    # Consider a margin to avoid locating lanes too close to border
    Ny, Nx = mask_input.shape[0:2]
    midpoint = np.int(Nx // 2)
    leftx_base = np.argmax(histogram[Nx // 10:midpoint]) + Nx // 10
    rightx_base = np.argmax(histogram[midpoint:Nx - Nx // 10]) + midpoint

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(Ny // nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = mask_input.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices/coordinates (first is x)
    # inds ==> refer to the image, not nonzero!
    left_lane_inds  = [[], []]
    right_lane_inds = [[], []]

    # Create an output image to draw on and visualize the result
    if NO_IMG == False:
        out_img = np.dstack((mask_input, mask_input, mask_input))
    else:
        out_img = None  # no image returned

    # keep track of the minimumy of the reliable (label-based regions)
    ymin_good_left = np.nan
    ymin_good_right = np.nan

    # Step through the windows one by one
    for window in range(nwindows):

        # Identify window boundaries in x and y (and right and left)
        win_y_low = Ny - (window + 1) * window_height
        win_y_high = Ny - window * window_height
        # Find the four boundaries of the window #
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image
        if NO_IMG == False:
            cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                          (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low),
                          (win_xright_high, win_y_high), (0, 255, 0), 2)

        # Identify the nonzero pixels in x and y within the window #
        # take indices of the non-zero selection!
        good_left_inds = np.where((nonzerox >= win_xleft_low) & (nonzerox <= win_xleft_high) &
                                  (nonzeroy >= win_y_low) & (nonzeroy <= win_y_high))[0]
        good_right_inds = np.where((nonzerox >= win_xright_low) & (nonzerox <= win_xright_high) &
                                   (nonzeroy >= win_y_low) & (nonzeroy <= win_y_high))[0]

        # Append these coordinates to the lists
        # left lane
        left_lane_inds[0].append(nonzerox[good_left_inds])
        left_lane_inds[1].append(nonzeroy[good_left_inds])
        # right lane
        right_lane_inds[0].append(nonzerox[good_right_inds])
        right_lane_inds[1].append(nonzeroy[good_right_inds])

        # check the connected regions inside the selection
        # if points are found, add the whole region to the selection of good indices
        # kept a separate list as this refers to the image indices (and not nonzero, as the labeling
        # requires the full image and not only the non zero points of the mask)
        labels_in_left = np.unique(labels[nonzeroy[good_left_inds], nonzerox[good_left_inds]])
        labels_in_left = labels_in_left[labels_in_left > 0]  # 0 = background
        if np.size(labels_in_left) > 0:
            for k in labels_in_left:
                # y indices of the whole region ==> get portion of region within y-window
                # value of label k maps to index k-1!
                yreg_left, xreg_left = region_idx_map[k - 1][0], region_idx_map[k - 1][1]
                reg_good_idx = np.where((yreg_left >= win_y_low) & (yreg_left <= win_y_high))[0]
                # store x and y coordinates in the appropriate lists
                left_lane_inds[0].append(xreg_left[reg_good_idx])
                left_lane_inds[1].append(yreg_left[reg_good_idx])
                # keep track of minimum value
                ymin_good_left = np.nanmin(np.concatenate(([ymin_good_left], yreg_left[reg_good_idx])))

        # same for right
        labels_in_right = np.unique(labels[nonzeroy[good_right_inds], nonzerox[good_right_inds]])
        labels_in_right = labels_in_right[labels_in_right > 0]  # 0 = background
        if np.size(labels_in_right) > 0:
            for k in labels_in_right:
                # y indices of the whole region ==> get portion of region within y-window
                # value of label k maps to index k-1!
                yreg_right, xreg_right = region_idx_map[k - 1][0], region_idx_map[k - 1][1]
                reg_good_idx = np.where((yreg_right >= win_y_low) & (yreg_right <= win_y_high))[0]
                # store x and y coordinates in the appropriate lists
                right_lane_inds[0].append(xreg_right[reg_good_idx])
                right_lane_inds[1].append(yreg_right[reg_good_idx])
                # keep track of minimum value
                ymin_good_right = np.nanmin(np.concatenate(([ymin_good_right], yreg_right[reg_good_idx])))


        # Update window's x center
        # left window
        # perform a fit to predict the window tendency, if a minimum number of points is present and the y span allows
        if (np.size(np.concatenate(left_lane_inds[1])) >= minpix) and \
                (np.max(np.concatenate(left_lane_inds[1]))- np.min(np.concatenate(left_lane_inds[1]))) > minpix:
            # make partial fit of order 1 or 2
            order = 1*(window<3) + 2*(window>=3)
            polycf_left = np.polyfit(np.concatenate(left_lane_inds[1]),
                                     np.concatenate(left_lane_inds[0]), order)
            # predict position at next window
            leftx_current = np.int(np.round(np.polyval(polycf_left, 0.5 * (win_y_low + win_y_high)-window_height)))

        # right window
        # perform a fit to predict the window tendency, if a minimum number of points is present and the y span allows
        if (np.size(np.concatenate(right_lane_inds[1])) >= minpix) and \
                (np.max(np.concatenate(right_lane_inds[1]))- np.min(np.concatenate(right_lane_inds[1]))) > minpix:
            # make partial fit of order 1 or 2
            order = 1*(window < 3) + 2*(window>=3)
            polycf_right = np.polyfit(np.concatenate(right_lane_inds[1]),
                                      np.concatenate(right_lane_inds[0]), order)
            # predict position at next window
            rightx_current = np.int(np.round(np.polyval(polycf_right, 0.5 * (win_y_low + win_y_high) - window_height)))

        # keep this plot ==> cool to explain procedure!
        PLOT_METHOD = False
        if PLOT_METHOD and (window == 4):
            stop()
            plt.figure(num=1)
            plt.clf()
            plt.imshow(out_img)
            y = np.arange(win_y_low-window_height, win_y_high)
            plt.plot(np.polyval(polycf_left, y), y, 'r--')
            plt.plot(np.repeat(leftx_current, 2), np.repeat(0.5 * (win_y_low + win_y_high) - window_height,2), 'b+')
            plt.plot(np.polyval(polycf_right, y), y, 'r--')
            plt.plot(np.repeat(rightx_current, 2), np.repeat(0.5 * (win_y_low + win_y_high) - window_height,2), 'b+')
            plt.pause(20)
            stop()


    # for each window

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    # Extract left and right line pixel positions
    # all indices refer to the whole mask!
    leftx = np.concatenate(left_lane_inds[0])
    lefty = np.concatenate(left_lane_inds[1])
    rightx = np.concatenate(right_lane_inds[0])
    righty = np.concatenate(right_lane_inds[1])

    # Fit a second order polynomial to each using `np.polyfit`, assuming x = f(y) #
    # for info on residuals ==> https: // stackoverflow.com / questions / 5477359 / chi - square - numpy - polyfit - numpy
    # use weights based on y, so the bottom pixels are weighted more
    polycf_left, sqr_error_left, _, _, _ = np.polyfit(lefty, leftx, 2, full = True, w=lefty / Ny)
    MSE_left = np.sqrt(sqr_error_left[0] / np.size(lefty))

    polycf_right, sqr_error_right, _, _, _ = np.polyfit(righty, rightx, 2, full = True, w=righty / Ny)
    MSE_right = np.sqrt(sqr_error_right[0] / np.size(righty))

    # Lane annotation (to be warped and shown in pipeline)
    lane_annotation = np.zeros([Ny, Nx, 3], dtype=np.uint8)
    lane_annotation[lefty, leftx] = [255, 0, 0]
    lane_annotation[righty, rightx] = [0, 0, 255]

    # Optional Visualization Steps
    if NO_IMG == False:
        # Set colors in the left and right lane regions
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]

        # Generate x and y values for plotting
        ploty = np.linspace(0, Ny - 1, Ny)

        left_fitx = np.polyval(polycf_left, ploty)
        right_fitx = np.polyval(polycf_right, ploty)

        # Plots the left and right polynomials on the lane lines (only works in non-interactive backend!)
        # otherwise ends up in currently open figure
        if plt.get_backend() == 'Agg':
            plt.plot(left_fitx, ploty, color='yellow')
            plt.plot(right_fitx, ploty, color='yellow')
    # Optional Visualization Steps

    # wrap data into LaneLines objects
    leftlane = LaneLine(leftx, lefty, polycf_left, MSE_left, ymin_good_left)
    rightlane = LaneLine(rightx, righty, polycf_right, MSE_right, ymin_good_right)

    return leftlane, rightlane, lane_annotation, out_img

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