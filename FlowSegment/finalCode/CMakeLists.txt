# cmake for OpenCV Cookbook 3rd edition chapter 12
# your opencv/build directory should be in your system PATH

# set minimum required version for cmake
cmake_minimum_required(VERSION 3.0)

# define the project name
set(project_name "Chapter 12")

# set the project name
project("${project_name}")

set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_STANDARD 17)

# add opencv package to the project
find_package(OpenCV REQUIRED)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})

# add executable
add_executable( videoprocessing videoprocessing.cpp)
add_executable( foreground foreground.cpp basic.cpp optimalDFTSize.cpp ImgSeg.cpp mat_save.cpp)

# link libraries
target_link_libraries( videoprocessing ${OpenCV_LIBS})
target_link_libraries( foreground ${OpenCV_LIBS})

# copy required images to every directory with executable
SET (IMAGES ${CMAKE_SOURCE_DIR}/images/bike.avi)
FILE(COPY ${IMAGES} DESTINATION .)
FILE(COPY ${IMAGES} DESTINATION "Debug")
FILE(COPY ${IMAGES} DESTINATION "Release")
