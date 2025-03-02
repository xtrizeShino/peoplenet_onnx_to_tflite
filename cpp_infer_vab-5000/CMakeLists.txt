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

### Avseil.io (absl)
add_subdirectory(abseil-cpp)
set(protobuf_ABSL_USED_TARGETS
absl::absl_check
absl::absl_log
absl::algorithm
absl::base
absl::bind_front
absl::bits
absl::btree
absl::cleanup
absl::cord
absl::core_headers
absl::debugging
absl::die_if_null
absl::dynamic_annotations
absl::flags
absl::flat_hash_map
absl::flat_hash_set
absl::function_ref
absl::hash
absl::layout
absl::log_initialize
absl::log_severity
absl::memory
absl::node_hash_map
absl::node_hash_set
absl::optional
absl::span
absl::status
absl::statusor
absl::strings
absl::synchronization
absl::time
absl::type_traits
absl::utility
absl::variant
)
target_link_libraries(PeopleNetInfer ${protobuf_ABSL_USED_TARGETS})

# flatbuffers
#add_subdirectory(flatbuffers)
#target_link_libraries(PeopleNetInfer flatbuffers)

# For Tensorflow Lite
target_link_libraries(PeopleNetInfer /home/debian/cross-tf/libtensorflowlite.so)
target_link_libraries(PeopleNetInfer /home/debian/cross-tf/libdelegate_loader.so)
target_link_libraries(PeopleNetInfer /home/debian/cross-tf/libtflite_settings_json_parser.so)
target_link_libraries(PeopleNetInfer /home/debian/cross-tf/libandroid_info.so)
target_link_libraries(PeopleNetInfer /home/debian/peoplenet_onnx_to_tflite/cpp_infer_vab-5000/flatbuffers/build/libflatbuffers.a)
#target_link_libraries(PeopleNetInfer /home/debian/peoplenet_onnx_to_tflite/cpp_infer_vab-5000/flatbuffers/build/flatc)
target_include_directories(PeopleNetInfer PUBLIC /home/debian/tensorflow)
target_include_directories(PeopleNetInfer PUBLIC /home/debian/tensorflow/tensorflow)
target_include_directories(PeopleNetInfer PUBLIC /home/debian/tensorflow/tensorflow/lite)
target_include_directories(PeopleNetInfer PUBLIC /home/debian/peoplenet_onnx_to_tflite/cpp_infer_vab-5000/flatbuffers/include)
#target_include_directories(PeopleNetInfer PUBLIC  /home/debian/flatbuffers/include)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -std=c++17 -lstdc++")

# Copy resouce
file(COPY ${CMAKE_SOURCE_DIR}/resource/ DESTINATION ${PROJECT_BINARY_DIR}/resource/)
add_definitions(-DRESOURCE_DIR="${PROJECT_BINARY_DIR}/resource/")