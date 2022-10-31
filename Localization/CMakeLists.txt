cmake_minimum_required(VERSION 3.10)

set(CMAKE_BUILD_TYPE Debug)

# specify the C++ standard
set(CMAKE_CXX_FLAGS "${CMAKE__CXX_FLAGS} -std=c++17")

# set the project name
project(EKF)


find_package (Eigen3 3.3 REQUIRED NO_MODULE
)
find_package (Boost 1.78 COMPONENTS ALL
)

include_directories(${EIGEN_INCLUDE_DIRS}
					${BOOST_INCLUDE_DIRS}
)

# add the executable
add_executable("${PROJECT_NAME}" 
	EKF.cpp
#	EKF.h
	robot.cpp
	robot.h
) 

target_link_libraries ("${PROJECT_NAME}" 
						Eigen3::Eigen
						${Boost_LIBRARIES}
)

# add_library("${PROJECT_NAME}_lib" SHARED EKF.cpp)