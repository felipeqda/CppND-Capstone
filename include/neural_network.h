#ifndef NN_H
#define NN_H

// apply deep neural network for object detection and classification using opencv/dnn framework
// https://docs.opencv.org/3.4/d5/de7/tutorial_dnn_googlenet.html


// YOLO pre-trained 
// https://pjreddie.com/darknet/yolo/

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <sstream>
#include <fstream>
#include <queue>

namespace dnn = cv::dnn;

struct Detection{
    cv::Rect box;
    double score;
    int class_id;
};


class NeuralNetwork{
    private:
        dnn::Net net_;
        int backend_, target_;
        std::vector<cv::String> out_layer_names_;
        std::vector<std::string> class_names_;
        float scale_;
        cv::Size input_size_;

    public:
        NeuralNetwork() = default;
        NeuralNetwork(const cv::String & model_path, const cv::String & config_path, const cv::String & classes_list, 
                      const cv::String & framework, const cv::Size input_size, float scale, int backend=0, int target=0);
        void load(const cv::String & model_path, const cv::String & config_path = "", const cv::String & framework = "",
                  int backend=0, int target =0);

        std::vector<Detection> run(const cv::Mat& frame, const cv::Scalar& mean = cv::Scalar(-128, -128, -128), bool swap_RB=true,
                                   float conf_thresh=0.5, float nms_thresh=0.4, int backend=-1);
        void run(const cv::Mat& frame, std::vector<Detection> & detections, std::mutex & mtx, const cv::Scalar& mean = cv::Scalar(-128, -128, -128),
                 bool swap_RB=true, float conf_thresh=0.5, float nms_thresh=0.4, int backend=-1);
        void preprocess(const cv::Mat& frame, cv::Size inpSize, float scale, const cv::Scalar& mean, bool swap_RB);
        std::vector<Detection> postprocess(const cv::Mat& frame, const std::vector<cv::Mat>& out, int backend=0, 
                                           float conf_thresh=0.5, float nms_thresh=0.4);
        void draw_box(int class_id, float conf, int left, int top, int right, int bottom, cv::Mat& frame);
        void draw(std::vector<Detection> detections, cv::Mat& frame);

        double exectime();

};

#endif