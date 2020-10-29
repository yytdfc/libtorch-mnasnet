cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(libtorch-mnasnet)
set(CMAKE_CXX_STANDARD 14)

find_package(Torch REQUIRED)
set(OpenCV_DIR /usr/local/share/opencv4)
find_package(OpenCV REQUIRED)

include_directories(/usr/local/include/opencv4)

aux_source_directory(src SRC)

add_executable(mnasnet-test src/test.cc)
target_link_libraries(mnasnet-test "${TORCH_LIBRARIES}" opencv_core opencv_highgui)

# add_executable(mnist src/mnist.cc)
# target_link_libraries(mnist "${TORCH_LIBRARIES}")
