cmake_minimum_required(VERSION 3.10)

# set the project name
project(RRT)

get_filename_component(PARENT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/.. ABSOLUTE)

if(POLICY CMP0167)
	cmake_policy(SET CMP0167 NEW)
endif()

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

include_directories(${EIGEN_INCLUDE_DIRS}
					${PARENT_DIR}/gnuplot-iostream
					include/
)

# add the executable
add_executable(${PROJECT_NAME}
		main.cpp
		rrt.cpp
		include/rrt.h
) 

target_link_libraries (${PROJECT_NAME}
		Eigen3::Eigen
		ImGuiImPlotLib
)