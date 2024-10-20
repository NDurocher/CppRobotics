cmake_minimum_required(VERSION 3.15)

set(CMAKE_BUILD_TYPE Debug)

# set the project name
project(particle_filter)

get_filename_component(PARENT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/.. ABSOLUTE)

find_package(Eigen3 3.3 REQUIRED NO_MODULE
)
find_package(Boost 1.78 COMPONENTS ALL
)
find_package(OpenCV REQUIRED)

set(OpenCV_INCLUDE_DIRS /Users/NathanDurocher/opencv/include)

include_directories(${EIGEN_INCLUDE_DIRS}
	${BOOST_INCLUDE_DIRS}
	${OpenCV_INCLUDE_DIRS}
	include/
)

# add the executable
add_executable(${PROJECT_NAME}
	main.cpp
	particle_filter.cpp
	include/particle_filter.h
)

target_link_libraries(${PROJECT_NAME} PUBLIC
	Eigen3::Eigen
	${Boost_LIBRARIES}
	${OpenCV_LIBS}
	CppRob::utils
)