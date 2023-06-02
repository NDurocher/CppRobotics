cmake_minimum_required(VERSION 3.15)

project(CppRobotics VERSION 0.1.0
		DESCRIPTION "A Cpp library of common robotic alogrithms"
		LANGUAGES C CXX)

# specify the C++ standard
set(CMAKE_CXX_FLAGS "${CMAKE__CXX_FLAGS} -std=c++20")



# executables
#add_subdirectory(EKFSLAM)
add_subdirectory(GPS_EKF)
#add_subdirectory(RRT)
#add_subdirectory(Visual_Odometry)


# inlcude user libraries
add_subdirectory(library)