#include "img_processing.h"


// auxiliary:
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
Warp2TopDown::Warp2TopDown(){
    // based on a Ny x Nx = 720 x 1280 image (straight_lines1/2.jpg)
    std::vector<cv::Point2f> pts_img{{190 + 1, 720}, {600 + 1, 445}, {680 - 2, 445}, {1120 - 2, 720}};
    std::vector<cv::Point2f> pts_warp{{350, 720}, {350, 0}, {50, 0}, {950, 720}};
    // store direct and inverse transform matrices
    M_= cv::getPerspectiveTransform(pts_img, pts_warp);
    M_inv_ = cv::getPerspectiveTransform(pts_warp, pts_img);

    /* define ROI for spatial filtering using inverse transform
    cf. equations at
    https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#getperspectivetransform
    auxiliar 3D coordinates are used with z =1 for source and a normalization factor t for the destination
    transposition is done to facilitate matrix product
    take the warp region as a reference and expand, to get a rectangle in the top-down view representing relevant
    search area */
    std::vector<cv::Point2f> pts_warpedROI_ = {{150, 720}, {150, 0}, {250, 0}, {250, 720}};
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
    cv::warpPerspective(std::move(frame_in), warped_frame, this->M_, frame_in.size());
    return warped_frame;
}
cv::Mat Warp2TopDown::unwarp(cv::Mat & frame_in){
    //Inverse perspective transform, M_inv
    cv::Mat unwarped_frame;
    cv::warpPerspective(std::move(frame_in), unwarped_frame, this->M_inv_, frame_in.size());
    return unwarped_frame;    
}

// getters
std::vector<cv::Point2f> Warp2TopDown::xy_warpedROI(){
    return pts_warpedROI_;
}
std::vector<cv::Point2f> Warp2TopDown::xy_ROI(){
    return pts_ROI_;
}