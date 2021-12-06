#ifndef CAL_H
#define CAL_H

#include <opencv2/core.hpp>     // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>

// container for output
struct calParams {
  cv::Mat cam_matrix;
  cv::Mat dist_coeff;
};

// forward declaration of calibration utilities
calParams get_calibration_params(bool force_redo=false);
void save_calibration_params(const cv::Mat & Camera_Matrix, const cv::Mat & Distortion_Coefficients);
void read_calibration_params(cv::Mat& Camera_Matrix, cv::Mat& Distortion_Coefficients);

#endif