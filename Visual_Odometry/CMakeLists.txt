cmake_minimum_required(VERSION 3.15)

set(CMAKE_BUILD_TYPE Debug)

# set the project name
project(Visual_Odometry)

get_filename_component(PARENT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/.. ABSOLUTE)
# guard against in-source builds
if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
    message(FATAL_ERROR "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there. You may need to remove CMakeCache.txt. ")
endif()

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
# find_package(Gnuplot 5.4 COMPONENTS ALL)
find_package(OpenCV REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)
find_package(PythonLibs 3.0 REQUIRED)
include_directories(${PYTHON3_INCLUDE_DIRS} ${NumPy_INCLUDE_DIRS})


set(OpenCV_INCLUDE_DIRS /Users/NathanDurocher/opencv/include)

include_directories(${EIGEN_INCLUDE_DIRS}
					${BOOST_INCLUDE_DIRS}
					${OpenCV_INCLUDE_DIRS}
					include/
					${PARENT_DIR}/library/matplotlib-cpp/
)

link_directories(${PARENT_DIR}/library/matplotlib-cpp/
)

# add the executable
add_executable(${PROJECT_NAME}
				main.cpp
				include/VO.h
				VO.cpp
        		include/feature_detector.h
        		feature_detector.cpp
				include/image_loader.h
				image_loader.cpp
) 

target_link_libraries (${PROJECT_NAME}
						Eigen3::Eigen
						${Gnuplot_LIBRARIES}
						${OpenCV_LIBS}
        				${PYTHON_LIBRARIES}
        				Python3::NumPy
)