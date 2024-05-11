cmake_minimum_required(VERSION 3.10)

project(matrix)

aux_source_directory(./src DIR_SRCS)

include_directories(./include)

add_executable(matrix ${DIR_SRCS})

set(CMAKE_BUILD_TYPE Debug)

add_compile_options(-std=c++11)