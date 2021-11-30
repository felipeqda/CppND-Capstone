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


using tf = tensorflow;


// Load
tf::SavedModelBundelLite model_bundle;
tf::SessionOptions session_options = SessionOptions();
tf::RunOptions run_options = RunOptions();
tf::Status status = LoadSavedModel(session_options, run_options, export_dir, {kSavedModelTagServe}, &model_bundle);

// Inference
tf::outputLayer = {"StatefulPartitionedCall:0", 
    	           "StatefulPartitionedCall:4", 
        	       "StatefulPartitionedCall:1", 
            	   "StatefulPartitionedCall:5",
               	   "StatefulPartitionedCall:3"};
tf::Status runStatus = model_bundle.GetSession()->Run({{"serving_default_input_tensor:0", input_tensor}}, outputLayer, {}, &outputs);



class TFModel{
	private:
		tf::SavedModelBundle bundle_;
		tf::SessionOptions session_options_;
		tf::RunOptions run_options_;
        tf::Status status_:
		template<typename T>
		tf::Status run_model_(T &);
		// void make_prediction(std::vector<tf::Tensor> &image_output, ClassifierOutput &pred);
	public:
		TFModel(std::string path2model);  // constructor loading the saved model from file
		void run(string filename, Prediction &out_pred);  // run the model on input
};

TFModel::TFModel(std::string path2model){		
	tf::session_options.config.mutable_gpu_options()->set_allow_growth(true);	
	status_ = tf::LoadSavedModel(session_options, run_options, path2model, {"serve"}, &bundle);
	if !(status.ok()) std::cout << "Error in loading TF model: " << path2model << std::endl);
}

tf::Status TFModel::run_model_(std::string filename, Prediction &out_pred){
	std::vector<tf::Tensor> image_output;
	tf::Status read_status = tf::ReadImageFile(filename, &image_output);
	make_prediction(image_output, out_pred);
}

void ModelLoader::make_prediction(std::vector<Tensor> &image_output, Prediction &out_pred){
	const string input_node = "serving_default_input_tensor:0";
	std::vector<std::pair<string, Tensor>> inputs_data  = {{input_node, image_output[0]}};
	std::vector<string> output_nodes = {{"StatefulPartitionedCall:0", //detection_anchor_indices
				"StatefulPartitionedCall:1", //detection_boxes
				"StatefulPartitionedCall:2", //detection_classes
				"StatefulPartitionedCall:3",//detection_multiclass_scores
				"StatefulPartitionedCall:4", //detection_scores                
				"StatefulPartitionedCall:5"}}; //num_detections

	
	std::vector<Tensor> predictions;
	this->bundle.GetSession()->Run(inputs_data, output_nodes, {}, &predictions);


	auto predicted_boxes = predictions[1].tensor<float, 3>();
	auto predicted_scores = predictions[4].tensor<float, 2>();
	auto predicted_labels = predictions[2].tensor<float, 2>();
	
	//inflate with predictions
	for (int i=0; i < 100; i++){
		std::vector<float> coords;
		for (int j=0; j <4 ; j++){
			coords.push_back( predicted_boxes(0, i, j));
		}
		(*out_pred.boxes).push_back(coords);
		(*out_pred.scores).push_back(predicted_scores(0, i));
		(*out_pred.labels).push_back(predicted_labels(0, i));
	}
}

// opencv headers
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

// using tensorflow::int32;
// using tensorflow::Status;
// using tensorflow::string;
// using tensorflow::Tensor;
// using tensorflow::tstring;
// using tensorflow::SavedModelBundle;
// using tensorflow::SessionOptions;
// using tensorflow::RunOptions;
// using tensorflow::Scope;
// using tensorflow::ClientSession;


// gather the output of the neural network (boxes containing objects, scores and labels)
class ClassifierOutput{
    public:
        ClassifierOutput(){}
    private:
        std::unique_ptr<std::vector<std::vector<float>>> boxes_;
        std::unique_ptr<std::vector<float>> scores_;
        std::unique_ptr<std::vector<int>> labels_;
};


Status ReadImageFile(const string &filename, std::vector<Tensor>* out_tensors){

	//@TODO: Check if filename is valid

	using namespace ::tensorflow::ops;
	Scope root = Scope::NewRootScope();
	auto output = tensorflow::ops::ReadFile(root.WithOpName("file_reader"), filename);

	tensorflow::Output image_reader;
	const int wanted_channels = 3;
	image_reader = tensorflow::ops::DecodeJpeg(root.WithOpName("file_decoder"), output, DecodeJpeg::Channels(wanted_channels));

	auto image_unit8 = Cast(root.WithOpName("uint8_caster"), image_reader, tensorflow::DT_UINT8);
	auto image_expanded = ExpandDims(root.WithOpName("expand_dims"), image_unit8, 0);

	tensorflow::GraphDef graph;
	auto s = (root.ToGraphDef(&graph));

	if (!s.ok()){
		printf("Error in loading image from file\n");
	}
	else{
		printf("Loaded correctly!\n");
	}

	ClientSession session(root);

	auto run_status = session.Run({image_expanded}, out_tensors);
	if (!run_status.ok()){
		printf("Error in running session \n");
	}
	return Status::OK();

}





// Test the CCN application to a single frame and display

#define THRESHOLD 0.8


int main(int argc, char* argv[]){
	if (argc != 4){
		std::cout << "Error! Usage: <path/to_saved_model> <path/to_input/image.jpg> <path/to/output/image.jpg>" << std::endl;
		return 1;
	}

	// Make a Prediction instance
	Prediction out_pred;
	out_pred.boxes = std::unique_ptr<std::vector<std::vector<float>>>(new std::vector<std::vector<float>>());
	out_pred.scores = std::unique_ptr<std::vector<float>>(new std::vector<float>());
	out_pred.labels = std::unique_ptr<std::vector<int>>(new std::vector<int>());

	const string model_path = argv[1]; 
	const string test_image_file  = argv[2];
	const string test_prediction_image = argv[3];

	// Load the saved_model
	ModelLoader model(model_path);

	//Predict on the input image
	model.predict(test_image_file, out_pred);

	cv::Mat img = cv::imread(test_image_file, cv::IMREAD_COLOR);

	cv::Size size = img.size();
	int height = size.height;
	int width = size.width;

	auto boxes = (*out_pred.boxes);
	auto scores = (*out_pred.scores);

	for (int i=0; i < boxes.size(); i++){
	    auto box = boxes[i];
	    auto score = scores[i];
	    if (score < THRESHOLD){
	        continue;
	    }
		int ymin = (int) (box[0] * height);
		int xmin = (int) (box[1] * width);
		int h = (int) (box[2] * height) - ymin;
		int w = (int) (box[3] * width) - xmin;
		cv::Rect rect = cv::Rect(xmin, ymin, w, h);
		cv::rectangle(img, rect, cv::Scalar(0, 0, 255), 2);
	}

	if (img.empty()){
		std::cout <<" Failed to read image" << std::endl;
	}

	imwrite(test_prediction_image, img);
}