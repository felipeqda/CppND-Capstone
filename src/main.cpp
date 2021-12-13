// example from:
//https://docs.opencv.org/4.5.4/d5/dc4/tutorial_video_input_psnr_ssim.html

#include <iostream> // for standard I/O
#include <string>   // for strings
#include <iomanip>  // for controlling float print precision
#include <sstream>  // string to number conversion


#include "tqdm.h"               // progress bar (external module)
#include "keyboard_interface.h"
#include "video_pipeline.h"
#include "calibration.h"

static void help(){
    std::cout
        << "------------------------------------------------------------------------------" << std::endl
        << "This program shows a video"                                                     << std::endl
        << "Usage:"                                                                         << std::endl
        << "./[executable] <referenceVideo>"                                                << std::endl
        << "--------------------------------------------------------------------------"     << std::endl
        << std::endl;
}


int main(int argc, char *argv[]){
    // TODO: allow input of file
    help();

    std::string input_file = std::string("/home/workspace/CarND-Advanced-Lane-Lines/project_video.mp4");
    float img_reduction = 0.5;

    VideoPipeline pipeline = VideoPipeline(input_file,
                                           std::string("Input Video: (ESPACE = pause/unpause, ENTER = step, ESC = quit)"),
                                           img_reduction) ;
    std::cout << "Input video resolution: Width=" << pipeline.width() << "  Height=" << pipeline.height()
              << " with n_frames = " << pipeline.frame_count() << std::endl;

    cv::Mat frame_in, frame_out; // containers to read and output a frame

    while(true){ //Show the image captured in the window and repeat 

        frame_in = pipeline.get_frame();
        if (frame_in.empty()){
            std::cout << "Input video is over!" << std::endl;
            break;
        }

        // call image processing pipeline on frame
      	// frame_out = frame_in;
        frame_out = pipeline.apply_processing(frame_in);

        // display
        // cv::resize(frame_out, frame_out, cv::Size(), 0.5, 0.5, cv::INTER_AREA);  // shrink input frame to display
        pipeline.display(frame_out);

        // allow user input for quitting/stopping playback at a frame
        bool quit_loop = pipeline.quit_loop(true);
        if (quit_loop) break;

    }
    return 0;
}