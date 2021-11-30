// TensorFlow C++ API References
// https://medium.com/@reachraktim/using-the-new-tensorflow-2-x-c-api-for-object-detection-inference-ad4b7fd5fecc
// https://medium.com/analytics-vidhya/inference-tensorflow2-model-in-c-aa73a6af41cf


#include <stdlib.h>
#include <stdio.h>
#include <string>
// tensorflow headers for loading model
#include <tensorflow/cc/saved_model/loader.h>
#include <tensorflow/cc/saved_model/tag_constants.h>
// #include "tensorflow/cc/client/client_session.h"

// #include "tensorflow/cc/ops/const_op.h"
// #include "tensorflow/cc/ops/image_ops.h"
// #include "tensorflow/cc/ops/standard_ops.h"
// #include "tensorflow/core/framework/graph.pb.h"
// #include "tensorflow/core/framework/tensor.h"
// #include "tensorflow/core/graph/default_device.h"
// #include "tensorflow/core/graph/graph_def_builder.h"
// #include "tensorflow/core/lib/core/errors.h"
// #include "tensorflow/core/lib/core/stringpiece.h"
// #include "tensorflow/core/lib/core/threadpool.h"
// #include "tensorflow/core/lib/io/path.h"
// #include "tensorflow/core/lib/strings/stringprintf.h"
// #include "tensorflow/core/platform/init_main.h"
// #include "tensorflow/core/platform/logging.h"
// #include "tensorflow/core/platform/types.h"
// #include "tensorflow/core/public/session.h"
// #include "tensorflow/core/util/command_line_flags.h"
// #include "tensorflow/core/framework/tensor_slice.h"

// opencv headers
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using tf = tensorflow;


// gather the output of the convolutional neural network (boxes containing objects, scores and labels)
class ClassifierOutput{
    public:
        ClassifierOutput(std::vector<float> tf_box, float tf_score, int tf_label);
		// acessors
		std::vector<float> box  { return _box;  }
		std::vector<float> score{ return score_;}
		std::vector<float> label{ return label_;}
    private:
        std::vector<int> box_;
        float score_;
        int label_;
		// index to access the output nodes according to model
		int idx_box_, idx_score_, idx_label_; 
};

ClassifierOutput::ClassifierOutput(std::vector<float> tf_box, float tf_score, int tf_label){
	for(auto coord_tf_box: tf_box){
		box_.emplace_back(coord);
	}		
	score_ = tf_score;
	label_ = tf_label;
}

// interface with Tensorflow-saved model (SSD CCN in this case)
class TFModel{
	private:
		tf::SavedModelBundle bundle_;
		tf::SessionOptions session_options_;
		tf::RunOptions run_options_;
        tf::Status status_:
		// behaviour
		void load_model_(std::string path2model);
		std::vector<tf::Tensor> run_model_(tf::Tensor const & input_tensor);
	public:
		TFModel(std::string path2model);  // constructor loading the saved model from file
		std::vector<ClassifierOutput> processFrame(cv::Mat const & framein, int COI, float threshold);  // run the model on input image
};

// constructor
TFModel::TFModel(std::string path2model){		
	session_options_ = tf::SessionOptions();
	run_options_ = tf::RunOptions();
	this->load_model_(path2model);	
}

// load model from disk
void TFModel::load_model_(std::string path2model){		
	tf::session_options.config.mutable_gpu_options()->set_allow_growth(true);	
	status_ = tf::LoadSavedModel(session_options, run_options, path2model, {"serve"}, &bundle);	
	if !(status_.ok()) std::cout << "Error in loading TF model: " << path2model << std::endl);
}

// run the model (IO in form of Tensors)
std::vector<tf::Tensor> TFModel::run_model_(tf::Tensor const & input_tensor){
	std::vector<tf::Tensor> tf_outputs;
	std::string input_layer = "serving_default_input_tensor:0";
	std::vector<std::pair<string, tf::Tensor>> tf_inputs  = {{input_layer, input_tensor}};
	std::vector<string> output_layer = {
		{"StatefulPartitionedCall:0", 	// detection_anchor_indices
		 "StatefulPartitionedCall:1",	// detection_boxes
		 "StatefulPartitionedCall:2", 	// detection_classes
		 "StatefulPartitionedCall:3",	// detection_multiclass_scores
		 "StatefulPartitionedCall:4", 	// detection_scores                
		 "StatefulPartitionedCall:5"}}; // num_detections
	// set indices for retrieval according to the layer order description
	idx_box_ = 1;
	idx_label_ = 2;
	idx_score_ = 4
	idx_n_ = 5;

	status_ = this->bundle_.GetSession()->Run(tf_inputs, output_layer, {}, &tf_outputs);
	if !(status_.ok()) std::cout << "Error in running TF model!" << std::endl);
	return tf_outputs;
}

// Convert cv::MAt to tf::Tensor
// https://stackoverflow.com/questions/39379747/import-opencv-mat-into-c-tensorflow-without-copying
tf::Tensor cvMat2tensor (cv::Mat const & frame_mat){
	// allocate a Tensor (output)
	tf::Tensor frame_tensor(tf::DT_UINT8, tf::TensorShape({1,frame_mat.rows,frame_mat.cols,3}));
	// get pointer to memory for that Tensor
	uint8_t *ptr_aux = frame_tensor.flat<tensorflow::uint8>().data();
	// create a "fake" cv::Mat from it 
	cv::Mat cameraImg(frame_mat.rows, frame_mat.cols, cv::CV_32FC3, ptr_aux);
	// use it here as a destination
	frame_mat.convertTo(cameraImg, cv::CV_32FC3);  //cv::CV_8UC3 ?
	return frame_tensor;
}

// get frame as cv::Mat and return vector of object with outputs
std::vector<ClassifierOutput> TFModel::processFrame(cv::Mat const & frame_in, int COI=-1, float threshold=0.0){
	cv::Size frame_size = frame_in.size();
	int frame_height = frame_size.height;
	int frame_width = frame_size.width;

	tf::Tensor tf_model_input = cvMat2tensor(frame_in);
	std::vector<tf::Tensor> cnn_detections = self->run_model(tf_model_input);
	
	// decode the model output (vector of tensors) ==> indices match node description
	nboxes = cnn_detection[idx_n_];
	auto cnn_boxes =  cnn_detection[idx_box_].tensor<float, 3>();
	auto cnn_scores = cnn_detection[idx_score_].tensor<float, 2>();
	auto cnn_labels = cnn_detection[idx_label_].tensor<float, 2>();

	std::vector<ClassifierOutput> detections; // output container
	for (int i =0; i<nboxes; ++i){
		// restrict to class of interest?
		if( (COI != -1) && (ccn_labels(0,i) != COI) ) continue;
		
		// retrict to above threshold
		if (cnn_score(0,i) < threshold) continue;

		// result ok, pack results into object...
		// opencv-rectangle compatible pixel coordinates
		int ymin = static_cast<int> (ccn_boxes(0,i,0) * frame_height);
		int xmin = static_cast<int> (ccn_boxes(0,i,1) * frame_width);
		int h = static_cast<int> (ccn_boxes(0,i,2) * frame_height) - ymin;
		int w = static_cast<int> (ccn_boxes(0,i,3) * frame_width) - xmin;
		std::vector<int> px_coords{xmin, ymin, w, h};
		detections.emplace_back(ClassifierOutput(px_coords, cnn_score(0,i), ccn_labels(0,i)));
	}
	return detections;
}



// Test the CCN application to a single frame and display

int main(int argc, char* argv[]){
	
	std::string test_image_file{"../data/test_images/test1.jpg"};
	std::string model_path{"../data/efficientdet_d0_coco17_tpu-32/saved_model/saved_model.pb"};

	// read test image
	cv::Mat img = cv::imread(test_image_file, cv::IMREAD_COLOR);
	if (img.empty()){
		std::cout <<" Failed to read image!" << std::endl;
	}	

	// Load the saved_model
	TFModel ccn{model_path};  // constructor loading the saved model from file

	// Run on the input image
	std::vector<ClassifierOutput> out = ccn.processFrame(img);  // run the model on input image

    // Create Window
	const char* WIN_OUT = "Output Window"; // window label
    cv::namedWindow(WIN_OUT, cv::WINDOW_AUTOSIZE);
    cv::moveWindow( WIN_OUT, 400       , 0);         
	cv::Size frame_size = img.size();
    std::cout << "Input frame size: Width=" << frameSize.width << "  Height=" << frameSize.height << std::endl;

	std::cout << "Boxes found: " << boxes.size() << std::endl;

	for (int i=0; i < boxes.size(); ++i){
		cv::Rect rect = cv::Rect(boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]);
		cv::rectangle(img, rect, cv::Scalar(0, 0, 255), 2);
	}
    
	// display
    // cv::resize(frameIn, frameOut, cv::Size(), 0.5, 0.5, cv::INTER_AREA);  // shrink input frame to display
    cv::imshow(WIN_OUT, img);

}