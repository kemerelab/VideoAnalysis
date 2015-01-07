cmake_minimum_required(VERSION 2.8.12)

project(vidanalysis)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(vidanalysis vidanalysis.cpp)
target_link_libraries(vidanalysis ${OpenCV_LIBS})
