cmake_minimum_required(VERSION 3.10)

set(CMAKE_BUILD_TYPE Debug)

# specify the C++ standard
set(CMAKE_CXX_FLAGS "${CMAKE__CXX_FLAGS} -std=c++17")

# set the project name
project(GPS_EKF)

get_filename_component(PARENT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/.. ABSOLUTE)

find_package (Eigen3 3.3 REQUIRED NO_MODULE
)
find_package (Boost 1.78 COMPONENTS ALL
)
find_package (Gnuplot 5.4 COMPONENTS ALL
)

include_directories(${EIGEN_INCLUDE_DIRS}
					${BOOST_INCLUDE_DIRS}
					${PARENT_DIR}/gnuplot-iostream
)

link_directories(${PARENT_DIR}/gnuplot-iostream
)

# add the executable
add_executable(${PROJECT_NAME}
				src/main.cpp
				include/${PROJECT_NAME}/robot.cpp
				include/${PROJECT_NAME}/robot.h
) 

target_link_libraries (${PROJECT_NAME}
						Eigen3::Eigen
						${Boost_LIBRARIES}
						${Gnuplot_LIBRARIES}
)