#ifndef VIDEOPIPE_H
#define VIDEOPIPE_H

#include <string>
#include <memory>
#include <opencv2/core.hpp>     // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/imgproc.hpp>  // Image processing operations
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>  // OpenCV window I/O
#include <queue>
#include <mutex>
#include <future>

#include "tqdm.h"
#include "keyboard_interface.h"
#include "calibration.h"
#include "img_processing.h"
#include "lane.h"
#include "neural_network.h"

class VideoPipeline{
    private:
        // input video parameters
        std::string filename;
        bool valid_video{false};
        cv::VideoCapture input_video;
        cv::Size frame_size_input;
        int n_frames{0}, idx_frame{-1}, n_buffer{0};
        
        float frame_reduction_; // shrink factor for frame
        int kb_delay, default_delay{5}; // delay in [ms], 0 means wait indefinitely
        tqdm pbar;  // progress bar

        // output video/window
        cv::VideoWriter output_video;
        cv::Size frame_size_output;
        std::string window_name;
        int n_written_frames;

        // tools
        bool cal_available{false};
        calParams cal;  // struct with cv::Mats needed for calibration
        Warp2TopDown topdown_transform;
        Lane local_lane_fit;
        Road road_fit;
        std::queue<double> radius_m;
        NeuralNetwork cnn;
        std::mutex mtx;  // allow paralelization

    public:
        //constructor
        VideoPipeline(std::string file_in, std::string win_label = "Input Video", float frame_reduction = 1.0, int n_buffer = 20);

        bool read_video(std::string path);
        bool write_video(std::string path, cv::Size output_size);

        cv::Mat get_frame();
        void save_frame(cv::Mat frame);
        void display(cv::Mat frame);

        cv::Mat apply_processing(cv::Mat frame);
        cv::Mat calibrate(const cv::Mat & frame_in);
        // object-detection related
        std::vector<Detection> apply_cnn(const cv::Mat& frame, const cv::Scalar& mean, bool swap_RB, double conf_thresh, double nms_thresh);
        void apply_cnn_thread(const cv::Mat& frame, std::vector<Detection> & objs, const cv::Scalar& mean, bool swap_RB, double conf_thresh, double nms_thresh);
        void annotate_objs(cv::Mat & frame, const std::vector<Detection> & objs, cv::Scalar color = cv::Scalar(0, 0, 255));

        // interface logic
        bool quit_loop(bool verbose);

        // getters for input video frame info
        int width(){       
            return valid_video ? frame_size_input.width: 0; 
        }
        int height(){      
            return valid_video ? frame_size_input.height: 0; 
        }
        int frame_count(){ 
            return n_frames; 
        }
        int frame_idx(){
            return idx_frame;
        }
        float reduction(){
            return frame_reduction_;
        }

        // setters
        void set_delay(int delay_ms){
            kb_delay = delay_ms;        // current delay in the interface logic (will change to key press to stop)
            default_delay = delay_ms;   // default delay: value the delay returns to when resuming video playback
        }

};

#endif