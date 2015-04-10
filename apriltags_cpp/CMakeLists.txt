cmake_minimum_required(VERSION 2.8)
SET(CMAKE_CXX_FILES "-g -Wall")

set(PROJECT_NAME apriltags_cpp)
PROJECT(${PROJECT_NAME})

# Find OpenCV package -> required
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Find eigen3 package -> required
find_package(Eigen3 REQUIRED)
include_directories(${Eigen3_INCLUDE_DIRS})

# Find OpenMP, if available add -fopenmp flags to compiler flags
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()
message(${CMAKE_CXX_FLAGS})


set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

set(INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${INCLUDE})

set(APRIL_SRC src/Edge.cc
	src/GLineSegment2D.cc  
	src/Quad.cc          
	src/TagFamily.cc
	src/FloatImage.cc  
	src/GrayModel.cc       
	src/Segment.cc       
	src/UnionFindSimple.cc
	src/Gaussian.cc    
	src/Homography33.cc    
	src/TagDetection.cc
	src/GLine2D.cc     
	src/MathUtil.cc        
	src/TagDetector.cc)

add_library(${PROJECT_NAME} ${APRIL_SRC})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBRARIES} ${Eigen3_LIBRARIES})

add_executable(apriltags_demo apriltags_demo.cpp)
target_link_libraries(apriltags_demo ${PROJECT_NAME})
