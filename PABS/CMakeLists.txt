# cmake for OpenCV Cookbook 3rd edition chapter 12
# your opencv/build directory should be in your system PATH

# set minimum required version for cmake
cmake_minimum_required(VERSION 3.0)

# define the project name
set(project_name "BGS")

# set the project name
project("${project_name}")

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)

# add opencv package to the project
find_package(OpenCV REQUIRED)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})

file(GLOB foreground foreground.cpp)

file(GLOB_RECURSE tools_src ./src/tools/*.cpp ./src/tools/*.c)
file(GLOB_RECURSE tools_inc ./src/tools/*.h ./src/tools/*.hpp)

file(GLOB_RECURSE utils_src ./src/utils/*.cpp ./src/utils/*.c)
file(GLOB_RECURSE utils_inc ./src/utils/*.h ./src/utils/*.hpp)

file(GLOB_RECURSE bgs_src ./src/algorithms/*.cpp ./src/algorithms/*.c)
file(GLOB_RECURSE bgs_inc ./src/algorithms/*.h ./src/algorithms/*.hpp)

add_library(bgslibrary_core STATIC ${bgs_src} ${tools_src} ${utils_src} ${bgs_inc} ${tools_inc} ${utils_inc})
# generates the export header bgslibrary_core_EXPORTS.h automatically
include(GenerateExportHeader)
GENERATE_EXPORT_HEADER(bgslibrary_core
    BASE_NAME bgslibrary_core
    EXPORT_MACRO_NAME bgslibrary_core_EXPORTS
    EXPORT_FILE_NAME bgslibrary_core_EXPORTS.h
    STATIC_DEFINE BGSLIBRARY_CORE_EXPORTS_BUILT_AS_STATIC)
target_link_libraries(bgslibrary_core ${OpenCV_LIBS})
set_property(TARGET bgslibrary_core PROPERTY PUBLIC_HEADER ${bgs_inc} ${tools_inc} ${utils_inc})


# add executable
add_executable( bgs_foreground ${foreground})

# link libraries
target_link_libraries( bgs_foreground ${OpenCV_LIBS} bgslibrary_core)

