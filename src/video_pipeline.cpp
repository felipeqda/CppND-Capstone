#include "video_pipeline.h"
#include "keyboard_interface.h"
#include "calibration.h"
#include "img_processing.h"
#include "annotate.h"

// constructor
VideoPipeline::VideoPipeline(std::string input_file, 
                             std::string win_label="Input Video",
                             float frame_reduction = 1.0){
    // open file for reading
    filename = input_file;
    valid_video = this->read_video(filename);
    frame_reduction_ = frame_reduction;

    if (valid_video){
        // Create Window
        window_name = std::move(win_label); // window label==> used as cv2 win-handle
        std::cout << "window label: " << window_name << std::endl;
        cv::namedWindow(window_name.c_str(), cv::WINDOW_AUTOSIZE);
        cv::moveWindow( window_name.c_str(), 400, 0);  

        // interface
        kb_delay = default_delay; // delay for key capture (0 will stop the execution)
        // progress bar
        pbar.set_theme_basic();
        pbar.set_title(std::string("video:"));
        pbar.set_funit(std::string("fps"));
        pbar.set_ETAformat(std::string(""));  // format must be printfcompatible!      
    }

    // Prepare transform object
    topdown_transform = Warp2TopDown(frame_reduction);

}

// open video for reading, return success flag
bool VideoPipeline::read_video(std::string path){
    input_video = cv::VideoCapture{path};
    if (!input_video.isOpened()){
        std::cout  << "Could not open input file " << path << std::endl;
        return false;
    }
    // get file parameters
    frame_size_input = cv::Size(static_cast<int>(input_video.get(cv::CAP_PROP_FRAME_WIDTH)),
                                static_cast<int>(input_video.get(cv::CAP_PROP_FRAME_HEIGHT)));
    n_frames = static_cast<int>(input_video.get(cv::CAP_PROP_FRAME_COUNT));
    idx_frame = -1;          // Counter for read frames
    return true;
}

// open videos for writting, return success flag
// cf. https://docs.opencv.org/3.4/d7/d9e/tutorial_video_write.html
bool VideoPipeline::write_video(std::string path, cv::Size output_size){
    cv::VideoWriter output_video;
    int ex = static_cast<int>(input_video.get(cv::CAP_PROP_FOURCC));     // Get Codec Type- Int form
    output_video.open(path, ex, input_video.get(cv::CAP_PROP_FPS), output_size, true);
    if (!output_video.isOpened()){
        std::cout  << "Could not open output file " << path << std::endl;
        return false;
    }
    n_written_frames = 0;          // Counter for written frames
    return true;
}

cv::Mat VideoPipeline::get_frame(){
    cv::Mat frame;
    if(valid_video) {
        // read frame and update count
        this->input_video >> frame;
        pbar.progress(++idx_frame, n_frames);
    }
    if (this->reduction() != 1.0  && !frame.empty()){
        cv::resize(frame, frame, cv::Size(), this->reduction(), this->reduction(), cv::INTER_AREA);  // shrink input frame
    }

    return std::move(frame);
}

void VideoPipeline::save_frame(cv::Mat frame){    
    output_video << frame;
    ++n_written_frames;
}

void VideoPipeline::display(cv::Mat frame){
    cv::imshow(window_name.c_str(), frame);
}

// interface signal to quit
bool VideoPipeline::quit_loop(bool verbose=false){
    return keyboard_interface(kb_delay, default_delay, verbose);
}    

// main pipeline image processing 
cv::Mat VideoPipeline::apply_processing(cv::Mat frame_in){
	cv::Mat frame_out;  // pass by value ==> copy
    
  	// I) Calibration: Undistort image based on calibration parameters
    if(!cal_available){      
  		cal = get_calibration_params();  
      	cal_available=true;
    }
   	cv::undistort(std::move(frame_in), frame_out, cal.cam_matrix, cal.dist_coeff);

    // II) Warp image to bird's eye view
    frame_out = topdown_transform.warp(frame_out);

    // III) color transformations and gradients
    frame_out = ImgProcessing::mask_lane(frame_out);

    // N) add side frame    
    // frame_out = annotate::add_side_panel(frame_out);
  	return std::move(frame_out);
}