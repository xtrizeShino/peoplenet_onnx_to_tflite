cmake_minimum_required(VERSION 2.8)
project(PeopleNetInfer)

# Create Main project
add_executable(PeopleNetInfer
	peoplenet_main.cpp
)

# For OpenCV
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
	target_include_directories(PeopleNetInfer PUBLIC ${OpenCV_INCLUDE_DIRS})
	target_link_libraries(PeopleNetInfer ${OpenCV_LIBS})
endif()

# For Tensorflow Lite
target_link_libraries(PeopleNetInfer /home/shino/tensorflow/bazel-bin/tensorflow/lite/libtensorflowlite.so)
target_include_directories(PeopleNetInfer PUBLIC /home/shino/tensorflow)
target_include_directories(PeopleNetInfer PUBLIC /home/shino/tensorflow/tensorflow)
target_include_directories(PeopleNetInfer PUBLIC /home/shino/tensorflow/tensorflow/lite)
target_include_directories(PeopleNetInfer PUBLIC  /home/shino/flatbuffers/include)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -std=c++17 -lstdc++")

# Copy resouce
file(COPY ${CMAKE_SOURCE_DIR}/resource/ DESTINATION ${PROJECT_BINARY_DIR}/resource/)
add_definitions(-DRESOURCE_DIR="${PROJECT_BINARY_DIR}/resource/")