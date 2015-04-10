cmake_minimum_required(VERSION 2.8)

set(PROJECT_NAME apriltags_c)
PROJECT(${PROJECT_NAME})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=gnu99 -Wall -Wno-unused-parameter -Wno-unused-function -pthread -I. -Icommon -O4")

set(APRIL_INCLUDE ${PROJECT_SOURCE_DIR}
		${PROJECT_SOURCE_DIR}/common)
include_directories(${APRIL_INCLUDE})

set(APRIL_SRC apriltag.c
	apriltag_quad_thresh.c
	g2d.c
	tag16h5.c
	tag25h7.c
	tag25h9.c
	tag36artoolkit.c
	tag36h10.c
	tag36h11.c
	common/getopt.c
	common/homography.c
	common/image_f32.c
	common/image_u32.c
	common/image_u8.c
	common/matd.c
	common/pnm.c
	common/postscript_utils.h
	common/string_util.c
	common/svd22.c
	common/time_util.c
	common/unionfind.c
	common/workerpool.c
	common/zarray.c
	common/zhash.c
	common/zmaxheap.c)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

add_library(${PROJECT_NAME} ${APRIL_SRC})
target_link_libraries(${PROJECT_NAME} -lpthread -lm)

add_executable(apriltags_demo apriltag_demo.c)
target_link_libraries(apriltags_demo ${PROJECT_NAME})
