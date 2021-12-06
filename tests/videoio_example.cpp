// example from:
//https://docs.opencv.org/4.5.4/d5/dc4/tutorial_video_input_psnr_ssim.html

#include <iostream> // for standard I/O
#include <string>   // for strings
#include <iomanip>  // for controlling float print precision
#include <sstream>  // string to number conversion
#include <opencv2/core.hpp>     // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/imgproc.hpp>  // Gaussian Blur
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>  // OpenCV window I/O
#include "tqdm.h"               // progress bar (external module)

// key code aliases (cf. https://www.c-sharpcorner.com/blogs/ascii-key-code-value1)
namespace keys{
    constexpr int esc    = 27;
    constexpr int espace = 32;
    constexpr int enter  = 13;
}

// what to do on key-press ==> return true on quit loop
bool keyboard_interface(int & delay_control, const int default_delay=5, bool verbose=false){
    int key_code = cv::waitKey(delay_control);  // integer waiting time in ms
    if(key_code != -1 && verbose) std::cout << "you pressed: " << key_code << std::endl;
    
    switch (key_code) {
    case keys::esc:
        // esc press will quit program
        if(verbose) std::cout << "ESC pressed: Quitting program!" << std::endl;            
        return true;
        break;

    case keys::espace:
        // toggle zero delay, which will block until next key press (pressing espace will pause/carry on)  
        delay_control = (delay_control==default_delay) ? 0: default_delay;
        return false;
        break;

    case keys::enter:
        // set zero delay, which will block until next key press (pressing enter will step through frames) 
        delay_control = 0;
        return false;
        break;

    default:
        // other key presses will resume execution
        delay_control = default_delay;
        return false;
        break;
    }
}

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
    help();
    std::string videoPath = std::string("/home/workspace/CarND-Advanced-Lane-Lines/project_video.mp4");
    int frameNum = -1;          // Frame counter
    cv::VideoCapture inputVideo(videoPath);
    if (!inputVideo.isOpened()){
        std::cout  << "Could not open reference " << videoPath << std::endl;
        return -1;
    }
    cv::Size frameSize = cv::Size((int) inputVideo.get(cv::CAP_PROP_FRAME_WIDTH),
                                  (int) inputVideo.get(cv::CAP_PROP_FRAME_HEIGHT));
    int nFrames{static_cast<int>(inputVideo.get(cv::CAP_PROP_FRAME_COUNT))};
    const char* WIN_IN = "Input Video: (ESPACE to pause/unpause, ENTER to step frame, ESC to quit)"; // window label

    // Create Window
    cv::namedWindow(WIN_IN, cv::WINDOW_AUTOSIZE);
    cv::moveWindow( WIN_IN, 400       , 0);         //750,  2 (bernat =0)
    std::cout << "Input video resolution: Width=" << frameSize.width << "  Height=" << frameSize.height
              << " with n_frames = " << nFrames << std::endl;

    cv::Mat frameIn, frameOut; // containers to read and output a frame

    // interface
    int delay_ms = 5; // delay for key capture (0 will stop the execution)
    // progress bar
    tqdm pbar;
    pbar.set_theme_basic();
    pbar.set_title(std::string("video:"));
    pbar.set_funit(std::string("fps"));
    // pbar.set_ETAformat(std::string("%.0f s (elap.) << %.0f s (rem.)"));  // format must be printfcompatible!
    pbar.set_ETAformat(std::string(""));  // format must be printfcompatible!


    while(true){ //Show the image captured in the window and repeat    
        inputVideo >> frameIn;
        if (frameIn.empty() || false ){
            pbar.finish();
            std::cout << "Input video is over!" << std::endl;
            break;
        }
        ++frameNum;
        pbar.progress(frameNum, nFrames);

        // display
        cv::resize(frameIn, frameOut, cv::Size(), 0.5, 0.5, cv::INTER_AREA);  // shrink input frame to display
        cv::imshow(WIN_IN, frameOut);

        // allow user input for quitting/chaging frame
        bool quit_loop = keyboard_interface(delay_ms);
        if (quit_loop) break;

    }
    return 0;
}