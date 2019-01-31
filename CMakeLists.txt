cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(libtorch-mnasnet)
set(CMAKE_CXX_STANDARD 14)

find_package(Torch REQUIRED)

aux_source_directory(src SRC)

add_executable(mnasnet-test src/test.cc)

target_link_libraries(mnasnet-test "${TORCH_LIBRARIES}")
