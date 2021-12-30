# include "neural_network.h"
// https://docs.opencv.org/3.4/d5/de7/tutorial_dnn_googlenet.html
// backends (int)
// 0: automatically (by default)
// 1: Halide language (http://halide-lang.org/)
// 2: Intel's Deep Learning Inference Engine (https://software.intel.com/openvino-toolkit)
// 3: OpenCV implementation
// 4: VKCOM,
// 5: CUDA

// target (int)
// 0: CPU target (by default),
// 1: OpenCL
// 2: OpenCL fp16 (half-float precision)
// 3: VPU
// 4: Vulkan
// 6: CUDA
// 7: CUDA fp16 (half-float preprocess)

NeuralNetwork::NeuralNetwork(const cv::String & model_path, const cv::String & config_path, const cv::String & classes_list, 
                             const cv::String & framework, const cv::Size input_size, float scale,
                             int backend, int target): target_(target), backend_(backend){                            
    this->load(model_path, config_path, framework);
    // store inputs to be used across calls
    input_size_ = input_size;
    scale_ = scale;
    // open file with classes names.
    std::ifstream ifs(classes_list.c_str());
    if (!ifs.is_open())
        CV_Error(cv::Error::StsError, "File " + classes_list + " not found");
    std::string line;
    while (std::getline(ifs, line)){
        class_names_.emplace_back(line);
    }
}

void NeuralNetwork::load(const cv::String & model_path, const cv::String & config_path, const cv::String & framework,
                         int backend, int target){
    // Load and pre-configure a model.
    net_ = dnn::readNet(model_path, config_path, framework);
    net_.setPreferableBackend(backend);
    net_.setPreferableTarget(target);
    out_layer_names_ = net_.getUnconnectedOutLayersNames();
}


void NeuralNetwork::preprocess(const cv::Mat& frame, cv::Size input_size, float scale,
                               const cv::Scalar& mean, bool swap_RB){
    static cv::Mat blob;
    // Create a 4D blob from a frame.
    if (input_size.width <= 0) input_size.width = frame.cols;
    if (input_size.height <= 0) input_size.height = frame.rows;
    dnn::blobFromImage(frame, blob, 1.0, input_size, cv::Scalar(), swap_RB, false, CV_8U);

    // Set-up inptus to run a model.
    net_.setInput(blob, "", scale, mean);
    if (net_.getLayer(0)->outputNameToIndex("im_info") != -1) { // Faster-RCNN or R-FCN
        cv::resize(frame, frame, input_size);
        cv::Mat im_info = (cv::Mat_<float>(1, 3) << input_size.height, input_size.width, 1.6f);
        net_.setInput(im_info, "im_info");
    }
}

std::vector<Detection> NeuralNetwork::postprocess(const cv::Mat& frame, const std::vector<cv::Mat>& cnn_output, int backend, float conf_thresh, float nms_thresh) {
    static std::vector<int> out_layers = net_.getUnconnectedOutLayers();
    static std::string out_layer_type = net_.getLayer(out_layers[0])->type;
    std::vector<Detection> detect_list;
    if (out_layer_type == "DetectionOutput")    {
        // Network produces output blob with a shape 1x1xNx7 where N is a number of
        // detections and an every detection is a vector of values
        // [batchId, classId, confidence, left, top, right, bottom]
        CV_Assert(cnn_output.size() > 0);
        for (size_t k = 0; k < cnn_output.size(); k++){
            float* data = (float*)cnn_output[k].data;
            for (size_t i = 0; i < cnn_output[k].total(); i += 7){
                float confidence = data[i + 2];
                if (confidence > conf_thresh){
                    int left   = (int)data[i + 3];
                    int top    = (int)data[i + 4];
                    int right  = (int)data[i + 5];
                    int bottom = (int)data[i + 6];
                    int width  = right - left + 1;
                    int height = bottom - top + 1;
                    if (width <= 2 || height <= 2){
                        left   = (int)(data[i + 3] * frame.cols);
                        top    = (int)(data[i + 4] * frame.rows);
                        right  = (int)(data[i + 5] * frame.cols);
                        bottom = (int)(data[i + 6] * frame.rows);
                        width  = right - left + 1;
                        height = bottom - top + 1;
                    }                    
                    int id = static_cast<int>(data[i + 1] - 1);  // Skip 0th background class id
                    Detection obj{cv::Rect(left, top, width, height), confidence, id};
                    detect_list.emplace_back(obj);
                }
            }
        }
    } else if (out_layer_type == "Region") {
        for (size_t i = 0; i < cnn_output.size(); ++i) {
            // Network produces output blob with a shape NxC where N is a number of
            // detected objects and C is a number of classes + 4 where the first 4
            // numbers are [center_x, center_y, width, height]
            float* data = (float*) cnn_output[i].data;
            for (int j = 0; j < cnn_output[i].rows; ++j, data += cnn_output[i].cols) {
                cv::Mat scores = cnn_output[i].row(j).colRange(5, cnn_output[i].cols);
                cv::Point class_id_pt;
                double confidence;
                //locate minimum
                std::vector<double> score;
                for(size_t i =0; i<detect_list.size();++i) score.emplace_back(detect_list[i].score);
                cv::minMaxLoc(scores, 0, &confidence, 0, &class_id_pt);
                if (confidence > conf_thresh) {
                    int centerX = (int)(data[0] * frame.cols);
                    int centerY = (int)(data[1] * frame.rows);
                    int width = (int)(data[2] * frame.cols);
                    int height = (int)(data[3] * frame.rows);
                    int left = centerX - width / 2;
                    int top = centerY - height / 2;

                    int id = static_cast<int>(class_id_pt.x);  // Skip 0th background class id
                    Detection obj{cv::Rect(left, top, width, height), confidence, id};
                    detect_list.emplace_back(obj);
                }
            }
        }
    } else
        CV_Error(cv::Error::StsNotImplemented, "Unknown output layer type: " + out_layer_type);

    // NMS is used inside Region layer only on DNN_BACKEND_OPENCV for another backends we need NMS in sample
    // or NMS is required if number of outputs > 1
    if (out_layers.size() > 1 || (out_layer_type == "Region" && backend != dnn::DNN_BACKEND_OPENCV)) {
        std::map<int, std::vector<size_t> > class2idxs;
        for (size_t i = 0; i < detect_list.size(); ++i){
            if (detect_list[i].score >= conf_thresh) {
                class2idxs[detect_list[i].class_id].emplace_back(i);
            }
        }
        
        std::vector<Detection> nms_outputs;
        for (std::map<int, std::vector<size_t> >::iterator it = class2idxs.begin(); it != class2idxs.end(); ++it) {
            std::vector<cv::Rect> local_boxes;
            std::vector<float> local_confidences;
            std::vector<size_t> class_idxs = it->second;
            for (size_t i = 0; i < class_idxs.size(); ++i) {
                local_boxes.push_back(detect_list[class_idxs[i]].box);
                local_confidences.push_back(detect_list[class_idxs[i]].score);
            }
            std::vector<int> nms_idxs;
            dnn::NMSBoxes(local_boxes, local_confidences, conf_thresh, nms_thresh, nms_idxs);
            for (size_t i = 0; i < nms_idxs.size(); ++i) {
                size_t idx = nms_idxs[i];
                Detection obj{local_boxes[idx], local_confidences[idx], it->first};                  
                nms_outputs.emplace_back(obj);
            }
        }
        detect_list = nms_outputs;
    }

    return detect_list;
}


void NeuralNetwork::draw_box(int class_id, float conf, int left, int top, int right, int bottom, cv::Mat& frame){
    cv::rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 255, 0));
    std::string label = cv::format("%.2f", conf);
    if (!class_names_.empty()) {
        CV_Assert(class_id < (int) class_names_.size());
        label = class_names_[class_id] + ": " + label;
    }

    int baseLine;
    cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

    top = std::max(top, labelSize.height);
    cv::rectangle(frame, cv::Point(left, top - labelSize.height), cv::Point(left + labelSize.width, top + baseLine), 
                  cv::Scalar::all(255), cv::FILLED);
    cv::putText(frame, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar());
}


double NeuralNetwork::exectime(){
    // Put efficiency information.
    std::vector<double> layersTimes;
    double freq = cv::getTickFrequency() / 1000;
    double t = net_.getPerfProfile(layersTimes) / freq;
    return t;
}

void NeuralNetwork::draw(std::vector<Detection> objs, cv::Mat & frame ){
    for (size_t idx = 0; idx < objs.size(); ++idx) {
        cv::Rect box = objs[idx].box;
        this->draw_box(objs[idx].class_id, objs[idx].score, box.x, box.y,
                       box.x + box.width, box.y + box.height, frame);
    }
    std::string label = cv::format("Inference time: %.2f ms", this->exectime());
    cv::putText(frame, label, cv::Point(0, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0));

}

// run, return vector of detected objects
std::vector<Detection> NeuralNetwork::run(const cv::Mat& frame, const cv::Scalar& mean, bool swap_RB, float conf_thresh, float nms_thresh, int backend){ 
    this->preprocess(frame, input_size_, scale_, mean, swap_RB);
    std::vector<cv::Mat> cnn_output;
    net_.forward(cnn_output, out_layer_names_);
    backend = backend == -1 ? backend_ : backend; // default uses value set on init
    std::vector<Detection> detections = this->postprocess(frame, cnn_output, backend, conf_thresh, nms_thresh);
    return detections;
}
// run, updating vector of detected objects in parallel
void NeuralNetwork::run(const cv::Mat& frame,  std::vector<Detection> & detections, std::mutex & mtx, 
                        const cv::Scalar& mean, bool swap_RB, float conf_thresh, float nms_thresh, int backend){ 
    this->preprocess(frame, input_size_, scale_, mean, swap_RB);
    std::vector<cv::Mat> cnn_output;
    net_.forward(cnn_output, out_layer_names_);
    backend = backend == -1 ? backend_ : backend; // default uses value set on init    
    std::vector<Detection> new_detections = this->postprocess(frame, cnn_output, backend, conf_thresh, nms_thresh);

    // lock, only to update vector (released on end of scope)
    std::lock_guard<std::mutex> lck(mtx);
    detections = std::move(new_detections);
    return;
}