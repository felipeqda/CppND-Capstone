// example from:
//https://docs.opencv.org/4.5.4/d5/dc4/tutorial_video_input_psnr_ssim.html

#include <iostream> // for standard I/O
#include <string>   // for strings
#include <iomanip>  // for controlling float print precision
#include <sstream>  // string to number conversion
#include <thread>
#include <future>
#include <functional>
#include <algorithm>

#include "tqdm.h"               // progress bar (external module)
#include "keyboard_interface.h"
#include "video_pipeline.h"
#include "calibration.h"

// requires a higher version of g++
# if __has_include(<filesystem>)
    #include <filesystem>  
    namespace fs = std::filesystem;  clear
#else
    // works for virtual machine version ==> requires target_link_libraries(... stdc++fs) in CMakeLists.txt
    #include <experimental/filesystem>
    namespace fs = std::experimental::filesystem;
# endif


static void help(){
    std::cout
        << "--------------------------------------------------------------------------------------------" << std::endl
        << "This program shows a video with a lane-identification pipeline and optional object detection" << std::endl
        << "Usage:"                                                                                       << std::endl
        << "./videopipe -f <referenceVideo.mp4>  [--cnn]                                                " << std::endl
        << "--------------------------------------------------------------------------------------------" << std::endl
        << std::endl;
}

int main(int argc, char *argv[]){
    // show usage instructions
    help();

    std::string input_file = std::string("../data/project_video.mp4");
    float img_reduction = 0.5;
    bool use_cnn{true}, default_video{true};

    // command line interface
    if( argc > 1 ) {
      	// allow specification of the file path after "-f" argument
        for( int i = 1; i < argc; ++i ){
            // -f <video_file>
            if( std::string_view{argv[i]} == "-f" && (i+1) < argc ){
                input_file = argv[++i];
                default_video = false;
                std::cout << "selected input video: " << input_file << std::endl;
                if (!fs::exists( fs::path(input_file) ) ) throw std::invalid_argument("input video does not exist!");
            }                
            // --cnn ==> activate cnn
            if( std::string_view{argv[i]} == "--cnn"){
                use_cnn = true;
                std::cout << "using object-detection neural network!" << std::endl;
            }
        }
    } // arguments specified

    if(default_video) {
      	// inform user and take default value vehicles
        std::cout << "------------------------------------------------------------ " << std::endl;
        std::cout << "Using default input in " << input_file                         << std::endl;
        if(!use_cnn) std::cout << "set --cnn to detect objects in the video!"        << std::endl;
        std::cout << "------------------------------------------------------------ " << std::endl;
    } 


    VideoPipeline pipeline = VideoPipeline(input_file,
                                           std::string("Input Video: (ESPACE = pause/unpause, ENTER = step, ESC = quit)"),
                                           img_reduction) ;
    std::cout << "Input video resolution: Width=" << pipeline.width() << "  Height=" << pipeline.height()
              << " with n_frames = " << pipeline.frame_count() << std::endl;

    cv::Mat frame_in, frame_out; // containers to read and output a frame

    // thread-related: must survive between loop calls
    std::vector<Detection> objs;  // vector with detections to be filled by the convolutional neural network 
    std::vector<std::future<void>> thd_futures; // auxiliar to call threads

    while(true){ //Show the image captured in the window and repeat 

        frame_in = pipeline.get_frame();
        if (frame_in.empty()){
            std::cout << "Input video is over!" << std::endl;
            break;
        }
        // apply calibration (undistortion)
        frame_in = pipeline.calibrate(frame_in);

        // detect objects (every n frames to speed up, no CUDA!)
        if(use_cnn && pipeline.frame_idx() % 20 == 0){
            // note std::ref required for pass-by-ref in this context
            thd_futures.push_back(std::async(std::launch::async, &VideoPipeline::apply_cnn_thread, &pipeline, 
                                             std::ref(frame_in), std::ref(objs), std::move(cv::Scalar(-128, -128, -128)), false, 0.3, 0.1) );
            // objs = pipeline.apply_cnn(frame_in, cv::Scalar(-128, -128, -128), false, 0.3, 0.1);  // non-parallel call 
        }     

        // call image processing pipeline on frame (masking and frame annotation)
      	// frame_out = frame_in;
        frame_out = pipeline.apply_processing(frame_in);

        // display
        if(use_cnn)
            pipeline.annotate_objs(frame_out, objs);
        // cv::resize(frame_out, frame_out, cv::Size(), 0.5, 0.5, cv::INTER_AREA);  // shrink input frame to display
        pipeline.display(frame_out);

        // allow user input for quitting/stopping playback at a frame
        bool quit_loop = pipeline.quit_loop(true);
        if (quit_loop) break;
    }

    // wait for launched threads
    if (use_cnn){
        std::for_each(thd_futures.begin(), thd_futures.end(), 
            [](std::future<void> &ftr) { ftr.wait();  }
        );
    }

    return 0;
}