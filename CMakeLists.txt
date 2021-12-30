cmake_minimum_required(VERSION 3.11.3)

project(Capstone)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -pthread")

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# add OpenCv
find_package(OpenCV 4.1 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
add_definitions(${OpenCV_DEFINITIONS})

# Find all executables, including headers
file(GLOB project_SRCS src/*.cpp include/*.h) 

# Add project executable
add_executable(videopipe ${project_SRCS})
target_link_libraries(videopipe PRIVATE ${OpenCV_LIBRARIES} stdc++fs)
target_link_libraries(videopipe PRIVATE Threads::Threads)
target_include_directories(videopipe PRIVATE include)
target_compile_options(videopipe PRIVATE -Wall -Wextra)


# Allow debugging
set(CMAKE_BUILD_TYPE Debug)