cmake_minimum_required(VERSION 2.8)
project( EDcircle )

SET( CMAKE_BUILD_TYPE Debug )

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

include_directories(
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/src
)

add_library(${PROJECT_NAME} SHARED
  src/EDPF.cpp
  src/EDcircle.cpp
)

add_executable( main main.cpp )
target_link_libraries( main 
${PROJECT_NAME}
${OpenCV_LIBS} )