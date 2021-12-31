
// io to xml:
// https://stackoverflow.com/questions/34382832/how-to-read-vectorvectorpoint3f-from-xml-file-with-filestorage

#include "calibration.h"
#include <vector>
#include <iostream>

// requires a higher version of g++
# if __has_include(<filesystem>)
    #include <filesystem>  
	#ifdef __APPLE__
		namespace fs = std::__fs::filesystem;
	#else
		namespace fs = std::filesystem;
	#endif
#else
    // works for virtual machine version ==> requires target_link_libraries(... stdc++fs) in CMakeLists.txt
    #include <experimental/filesystem>
    namespace fs = std::experimental::filesystem;
# endif

namespace xml_io{

// https://stackoverflow.com/questions/16312904/how-to-write-a-float-mat-to-a-file-in-opencv/16314013
void write_matrix2xml(const cv::Mat & matrix, std::string filename) {
    // Write data to xml file
    cv::FileStorage file_handle(filename, cv::FileStorage::WRITE);
    file_handle << "matrix" << matrix;
    file_handle.release();
}

// https://docs.opencv.org/4.x/dd/d74/tutorial_file_input_output_with_xml_yml.html
cv::Mat read_xml2matrix(std::string filename){
    // Read data from xml file
    cv::FileStorage file_handle(filename, cv::FileStorage::READ);
    cv::Mat out_data;
    if (file_handle.isOpened()) file_handle["matrix"] >> out_data;
  	file_handle.release();
    return std::move(out_data);
}

}; // end of namespace xml_io

namespace paths{
	std::string camera_mat_file = std::string("../data/camera_cal/cal_cameramatrix.xml");
  	std::string distortion_coeffs_file = std::string("../data/camera_cal/cal_distcoefs.xml");
};

void save_calibration_params(const cv::Mat &  Camera_Matrix, const cv::Mat & Distortion_Coefficients){
    xml_io::write_matrix2xml(Camera_Matrix, paths::camera_mat_file);
    xml_io::write_matrix2xml(Distortion_Coefficients, paths::distortion_coeffs_file);
}

void read_calibration_params(cv::Mat & Camera_Matrix, cv::Mat & Distortion_Coefficients){
    Camera_Matrix = xml_io::read_xml2matrix(paths::camera_mat_file);
    Distortion_Coefficients = xml_io::read_xml2matrix(paths::distortion_coeffs_file);
}

calParams get_calibration_params(bool force_redo){
  
  	// declare main outputs
	cv::Mat Camera_Matrix, Distortion_Coefficients;  
  
  	// check for existing files
  	bool found = fs::exists( fs::path(paths::camera_mat_file) ) &&
				 fs::exists( fs::path(paths::distortion_coeffs_file) );
  
	if(!found || force_redo){  
      // I) Perform computation of calibration parameters and save
      cv::Size patternsize(9, 6); //interior number of corners
      std::vector<cv::Point3f> corner_pattern; // expected corner patterns in a given image
      for(int j=0;j<6;++j){
          for (int i=0;i<9;++i){
              corner_pattern.emplace_back(cv::Point3f{static_cast<float>(i), static_cast<float>(j), 0.0});
          }
      }

      // lists over all images
      std::vector<std::vector<cv::Point3f>> nominal_corners;
      std::vector<std::vector<cv::Point2f>> actual_corners;

      std::string cal_img_path = "../data/camera_cal";
      cv::Size img_size;
      for (const auto & img : fs::directory_iterator(cal_img_path)){
          // cf. https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga93efa9b0aa890de240ca32b11253dd4a
          std::vector<cv::Point2f> corners; //this will be filled by the detected corners

          cv::Mat image_gray = cv::imread(img.path(), cv::IMREAD_GRAYSCALE);
          img_size = image_gray.size();
          // grayscl = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

          //CALIB_CB_FAST_CHECK saves a lot of time on images
          //that do not contain any chessboard corners
          bool patternfound = cv::findChessboardCorners(image_gray, patternsize, corners,
                              cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE + cv::CALIB_CB_FAST_CHECK);
          if(patternfound){
              // add to vector with nominal and actual corners
              nominal_corners.emplace_back(corner_pattern);
              actual_corners.emplace_back(corners);
          }

      } //for each image

      //cf. https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga3207604e4b1a1758aa66acb6ed5aa65d
      // Compute intrinsic camera parameters.
      std::vector<cv::Mat> rvecs, tvecs;
      cv::calibrateCamera(nominal_corners, actual_corners, img_size, Camera_Matrix, Distortion_Coefficients, rvecs, tvecs);

      // save outputs
      save_calibration_params(Camera_Matrix, Distortion_Coefficients);
  
  	// B) Restore  previous computation
	} else {
		read_calibration_params(Camera_Matrix, Distortion_Coefficients);      
    }
  
  	// prepare outputs
  	calParams output{Camera_Matrix, Distortion_Coefficients};
  	return output;
}
