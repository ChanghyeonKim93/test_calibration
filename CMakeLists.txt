cmake_minimum_required(VERSION 3.9)
project(test_calibration)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O2")
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR})

set(INCLUDE_DIR ${INCLUDE_DIR} 
  ${CMAKE_CURRENT_BINARY_DIR})

find_package(Eigen3 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)

include_directories(${INCLUDE_DIR})
include_directories(${Eigen3_INCLUDE_DIRS})
include_directories(${CERES_INCLUDE_DIRS})

set(SRC ${SRC}
)

add_executable(example_camera_calibration src/example_camera_calibration.cc
 ${SRC})
target_link_libraries(example_camera_calibration 
${Eigen3_LIBRARIES} ${CERES_LIBRARIES})
