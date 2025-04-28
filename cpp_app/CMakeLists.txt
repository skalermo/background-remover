cmake_minimum_required(VERSION 3.10)
project(cpp_app)

# Source files
set(SOURCES
    src/main.cpp
)

# Include local headers
include_directories(${PROJECT_SOURCE_DIR})

# Find OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Find libyuv
find_library(YUV_LIBRARY NAMES yuv PATHS /usr/lib /usr/local/lib)

# Find Torch
find_package(Torch REQUIRED)

# Create the executable
add_executable(cpp_app ${SOURCES})

# Link libraries
target_link_libraries(cpp_app ${YUV_LIBRARY} ${OpenCV_LIBS} ${TORCH_LIBRARIES})
set_property(TARGET cpp_app PROPERTY CXX_STANDARD 17)
