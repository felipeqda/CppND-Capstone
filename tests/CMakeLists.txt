cmake_minimum_required(VERSION 3.11.3)

project(Capstone)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -pthread")

# add OpenCv
find_package(OpenCV 4.1 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
add_definitions(${OpenCV_DEFINITIONS})
MESSAGE( [Main] " OpenCV_INCLUDE_DIRS = ${OpenCV_INCLUDE_DIRS}") 


# add TensorFlow
# add path of TensorFlowConfig.cmake (set up paths of library)
# cf. https://github.com/cjweeks/tensorflow-cmake
list(APPEND CMAKE_PREFIX_PATH /home/workspace/CppND-Capstone/cmake)
find_package(TensorFlow REQUIRED)
include_directories(${TensorFlow_INCLUDE_DIRS})
link_directories(${TensorFlow_LIBRARY_DIRS})
add_definitions(${TensorFlow_DEFINITIONS})

# Eigen
find_package (Eigen3 REQUIRED)
include_directories(${Eigen3_INCLUDE_DIRS})
MESSAGE( [Main] " EIGEN3_INCLUDE_DIRS = ${EIGEN3_INCLUDE_DIRS}") 

# Protobuf
find_package(Protobuf REQUIRED)
include_directories(${Protobuf_INCLUDE_DIRS})
MESSAGE( [Main] " PROTOBUF_INCLUDE_DIRS = ${Protobuf_INCLUDE_DIRS}") 

# set variables for external dependencies
set(EXTERNAL_DIR "${PROJECT_SOURCE_DIR}/external" CACHE PATH "Location where external dependencies will installed")
set(DOWNLOAD_LOCATION "${EXTERNAL_DIR}/src" CACHE PATH "Location where external projects will be downloaded")
mark_as_advanced(EXTERNAL_DIR DOWNLOAD_LOCATION)
# include(Protobuf)
# include(Eigen3)
set(PROJECT_INCLUDE_DIRS ${TensorFlow_INCLUDE_DIRS} ${EXTERNAL_DIR}/include)
set(PROJECT_LIBRARIES ${TensorFlow_LIBRARIES})
set(PROJECT_DEPENDENCIES Eigen Protobuf)

# Find all executables
# file(GLOB project_SRCS src/*.cpp) #src/*.h
# run only the example
# file(GLOB project_SRCS src/videoio_example.cpp)  
file(GLOB project_SRCS src/tensorflow_example.cpp)  


include_directories(${PROJECT_INCLUDE_DIRS})


# Add project executable
add_executable(videopipe ${project_SRCS})
target_link_libraries(videopipe ${OpenCV_LIBRARIES} ${PROJECT_LIBRARIES})
add_dependencies(videopipe ${PROJECT_DEPENDENCIES})

target_compile_options(videopipe PRIVATE -Wall -Wextra)

# Allow debugging
set(CMAKE_BUILD_TYPE Debug)