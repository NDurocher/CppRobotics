cmake_minimum_required(VERSION 3.15)

set(CMAKE_BUILD_TYPE Debug)

# set the project name
project(EKFSLAM)

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
					include/
)

link_directories(${PARENT_DIR}/gnuplot-iostream
)

# add the executable
add_executable(${PROJECT_NAME} 
				main.cpp
				ekf_slam.cpp
				include/ekf_slam.h
) 

target_link_libraries(${PROJECT_NAME} PUBLIC
						Eigen3::Eigen
						${Boost_LIBRARIES}
						${Gnuplot_LIBRARIES}
						CppRob::utils
)