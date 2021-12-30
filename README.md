# CPPND: C++ Capstone Project

This is the repository for the Capstone project in the [Udacity C++ Nanodegree Program](https://www.udacity.com/course/c-plus-plus-nanodegree--nd213), which is meant to fulfill the requirements in [the rubric](https://review.udacity.com/#!/rubrics/2533/view).

The application created here, using a wide range of C++ features, is based on a [video pipeline for lane finding in ADAS](https://github.com/felipeqda/CarND-Advanced-Lane-Lines/blob/77e83945fa359e60651ca760d5a56ae0ac79bfc0/writeup.md) initially developed in Python for the [Udacity Self-Driving Car Nanodegree](https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd0013). The C++ version process a video from a car camera on-the-fly and outputs lane markings (and as an optional feature object detection using the pre-trained [tiny-YOLO](https://pjreddie.com/darknet/yolo/) convolutional neural network, integrated with [opencv](https://docs.opencv.org/3.4/da/d9d/tutorial_dnn_yolo.html)) with around 21 fps in the Udacity project server. The python version runs offline, with a processing time several times larger the video duration on the same machine, but the C++ version fully unleashes the application's potential!

To keep the description short, focus is turned to the C++ code implementation features. The [Python repository](https://github.com/felipeqda/CarND-Advanced-Lane-Lines/blob/77e83945fa359e60651ca760d5a56ae0ac79bfc0/writeup.md) contains a more through description of the algorithms and processing steps. The [opencv/dnn](https://docs.opencv.org/3.4/da/d9d/tutorial_dnn_yolo.html) feature is new to the C++ version.

## Dependencies for Running Locally (tested in Ubuntu 16.04 in the Udacity server)
* cmake >= 3.11.3
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)
* opencv 4.1
  * Linux [installation instructions](https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html)
  * Windows [installation instructions](https://docs.opencv.org/4.x/d3/d52/tutorial_windows_install.html)
* other libraries (cf. *CmakeLists.txt*)
  * stdc++fs, libthread

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory in the top level directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./videopipe [--cnn]` (the default video path is contained within the /data/ directory).


## File Structure

  The repository has the following folder/file structure:

    .
    ├── build                   # Compiled files (use `cmake` and `make`)
    ├── cmake                   # Auxiliary build instruction files (created automatically by `cmake`)
    ├── data                    # Input files
    │   ├── camera_cal          # Images for fish-eye camera calibration (`cv:::undistort`)
    │   ├── darknet             # Repository files for YOLO* convolutional neural network for object detection.
    │   ├── test_images         # Auxiliary images for calibration (originally provided in the Advaced Lane Lines Project Repository**)
    │   ├── car_top.png         # Image for annotation
    │   └── project_video.mp4   # Main input video (provided as input in the Advanced Lane Lines Project Repository)
    ├── include                 # Cpp headers (`*.h`)
    │   └── ...                 # (annotate.h, calibration.h, img_processing.h, keyboard_interface.h, lane.h, math_lib.h, neural_network.h, tqdm.h, video_pipeline.h)
    ├── include                 # Cpp headers (`*.h`)
    ├── src                     # Source files (`*.cpp*`)
    │   └── ..                  # (annotate.cpp, calibration.cpp, img_processing.cpp, keyboard_interface.cpp, lane.cpp, math_lib.cpp, neural_network.cpp, tqdm.cpp, video_pipeline.cpp)
    ├── CMakeLists.txt          # Cmake instructions for build
    ├── LICENSE.md              # Udacity license information
    └── README.md               # This file

References:
*[YOLO](https://pjreddie.com/darknet/yolo/)
**[Advanced Lane Lines Project Repository](https://github.com/felipeqda/CarND-Advanced-Lane-Lines)

## Code and Class Structure

  The main function which handles user input arguments is `main` defined in _main.cpp_. It instantiates an object of the class `VideoPipeline` (defined in *video_pipeline.cpp*) which runs an infinite loop handling frame input, processing and display; and quits when the video is over. The methods of `VideoPipeline` handle the following functionality:
  * interface with the video file (`cv::VideoCapture` and `cv::VideoWriter`)
  * calibration (intermediate steps to apply `cv::undistort` using tools defined in _calibration.cpp_ )
  * application of computer vision to locate lanes (using `Lane` object defined in _lane.cpp_, cf. [video pipeline for lane finding in ADAS](https://github.com/felipeqda/CarND-Advanced-Lane-Lines/blob/77e83945fa359e60651ca760d5a56ae0ac79bfc0/writeup.md) for a full description of the algorithms and processing chain)
    * warping of the camera image to top-down perspective
    * color space transformations and masking of the frame to find the lanes
    * fitting of the lanes (using fit tools in *math_lib.cpp*)
  * tracking/statistics of the lanes to get road parameters (using `Road` object defined in _lane.cpp_ and statistical tools in *math_lib.cpp*)
  * parallel execution of the neural network (using `NeuralNetwork` object defined in *neural_network.cpp*)
  * annotation of the detected lanes onto the frame (using namespace `annotation` defined in *annotation.cpp*)
  * annotation of the detected objects onto the frame, when available (vector of `cv::Rect` is populated by parallel task employing mutexes and parallel execution tools within the `VideoPipeline` and `NeuralNetwork` objects' methods, final marking is done using namespace `annotation` defined in *annotation.cpp*)


## Rubric Points

The goal of this section is to provide an overview to highlight the fulfillment of the rubric points, rather than an exaustive discussion. The examples and comments are provided in table format, separated by topics following the [rubric](https://review.udacity.com/#!/rubrics/2533/view) structure.

#### Compiling and testing (REQUIRED)

Criteria                              | Comment
------------------------------------- | -------------
The submission must compile and run.  | Compilation is done using `cmake .. && make` in the build directory. 


#### Loops, Functions, IO

Criteria                              | Specification / Comment
------------------------------------- | -------------
The project demonstrates an understanding of C++ functions and control structures.  | A variety of control structures are used in the project: e.g. main loop for frame processing. The project code is clearly organized into functions, including namespaces (e.g. *annotate.cpp* and member functions of various clases).
The project reads data from a file and process the data, or the program writes data to a file. | The input video file (*data/project_video.mp4*) is processed and frames output to a `opencv::window` object.
The project accepts user input and processes the input. | The commmand-line interface allows using a different video file and turning the neural network processing on/off.


#### Object-Oriented Programming

Criteria                              | Comment
------------------------------------- | -------------
The project uses Object Oriented Programming techniques. | Several classes are defined, e.g. VideoPipeline, Lane or Road.
Classes use appropriate access specifiers for class members. | Classes show private member variables and public interfaces.
Class constructors utilize member initialization lists.  | E.g. `VideoPipeline` and `NeuralNetwork` constructors use initializer lists
Classes follow an appropriate inheritance hierarchy. | E.g. `Road` inherits from `Lane`, as it refers to the aggregation of the information across frames
Overloaded functions allow the same function to operate on different parameters. | E.g. `BufferStats::stddev` is overloaded for no inputs or _float_ input
Templates generalize functions in the project. | E.g. the class `BufferStats` in *math_lib.cpp* uses templates to deal with different input argument types (float, int)


#### Memory Management

Criteria                              | Comment
------------------------------------- | -------------
The project makes use of references in function declarations.  | const & is preferred to pass by copy in several instances of the code. 
The project uses move semantics to move data, instead of copying it, where possible. | When using std containers, `.emplace_back()` is preferred, `std::move` is used in several instances.

#### Concurrency

Criteria                              | Comment
------------------------------------- | -------------
The project uses multithreading.        | `std::async` is used to launch a task in parallel to run the neural network on the frame and populate the vector with detected objects.
A mutex or lock is used in the project. | `std::lock_guard` and `std::unique_lock` are used for sharing the vector of detected objects between the main thread and the task running the neural network.