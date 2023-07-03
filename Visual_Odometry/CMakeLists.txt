cmake_minimum_required(VERSION 3.15)

set(CMAKE_BUILD_TYPE Debug)

# set the project name
project(Visual_Odometry)

get_filename_component(PARENT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/.. ABSOLUTE)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Boost 1.78 COMPONENTS ALL)
find_package(Gnuplot 5.4 COMPONENTS ALL)
find_package(OpenCV REQUIRED)

# specify the C++ standard
set(CMAKE_CXX_FLAGS "${CMAKE__CXX_FLAGS} -std=c++17")
set(OpenCV_INCLUDE_DIRS /Users/NathanDurocher/opencv/include)

include_directories(${EIGEN_INCLUDE_DIRS}
					${BOOST_INCLUDE_DIRS}
					${OpenCV_INCLUDE_DIRS}
					include/
					library/
)

link_directories(${PARENT_DIR}/gnuplot-iostream
)

# add the executable
add_executable(${PROJECT_NAME}
				main.cpp
				include/VO.h
				VO.cpp
) 

target_link_libraries (${PROJECT_NAME}
						Eigen3::Eigen
						${Boost_LIBRARIES}
						${Gnuplot_LIBRARIES}
						${OpenCV_LIBS}
						${Gnuplot_LIBRARIES}

)