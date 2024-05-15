cmake_minimum_required(VERSION 3.10)
project(matrix)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Debug)

aux_source_directory(./src DIR_SRCS)
include_directories(./include)

add_executable(matrix ${DIR_SRCS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma")