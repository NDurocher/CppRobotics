cmake_minimum_required(VERSION 3.10)

set(CMAKE_BUILD_TYPE Debug)

# specify the C++ standard
set(CMAKE_CXX_FLAGS "${CMAKE__CXX_FLAGS} -std=c++17")

# set the project name
project(EKF_SLAM)


find_package (Eigen3 3.3 REQUIRED NO_MODULE
)
find_package (Boost 1.78 COMPONENTS ALL
)
find_package (Gnuplot 5.4 COMPONENTS ALL
)

include_directories(${EIGEN_INCLUDE_DIR}
					${BOOST_INCLUDE_DIRS}
					${GNUPLOT_INCLUDE_DIRS}
					/Users/NathanDurocher/cppyourself/CppRobotics/Localization gnuplot-iostream
)

# add the executable
add_executable("${PROJECT_NAME}" 
	SLAM_EKF.cpp
	SLAM_EKF.h
	EKF.cpp
	robot.cpp
	robot.h
) 

target_link_libraries ("${PROJECT_NAME}" 
						Eigen3::Eigen
						${Boost_LIBRARIES}
						${Gnuplot_LIBRARIES}
)