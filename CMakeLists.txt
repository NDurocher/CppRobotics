cmake_minimum_required(VERSION 3.15)

project(CppRobotics VERSION 0.1.0
		DESCRIPTION "A Cpp project of common robotic algorithms"
		LANGUAGES C CXX)

# specify the C++ standard
set(CMAKE_CXX_FLAGS "${CMAKE__CXX_FLAGS} -std=c++20")

set(OpenCV_INCLUDE_DIRS /Users/NathanDurocher/opencv/include)

# include user libraries
add_subdirectory(library)

# executables
add_subdirectory(EKFSLAM)
add_subdirectory(GPS_EKF)
add_subdirectory(RRT)
add_subdirectory(Visual_Odometry)
add_subdirectory(particle_filter)