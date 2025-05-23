cmake_minimum_required(VERSION 3.15)

# set the project name
project(EKFSLAM)

get_filename_component(PARENT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/.. ABSOLUTE)

if(POLICY CMP0167)
	cmake_policy(SET CMP0167 NEW)
endif()

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package (Boost REQUIRED COMPONENTS iostreams system filesystem)
find_package (Gnuplot 5.4 COMPONENTS ALL)

include_directories(${EIGEN_INCLUDE_DIRS}
		${PARENT_DIR}/gnuplot-iostream
		include/
)

link_directories(${PARENT_DIR}/gnuplot-iostream)

# add the executable
add_executable(${PROJECT_NAME} 
		main.cpp
		ekf_slam.cpp
		include/ekf_slam.h
) 

target_link_libraries(${PROJECT_NAME} PUBLIC
		Eigen3::Eigen
		Boost::iostreams
		Boost::system
		Boost::filesystem
		${Gnuplot_LIBRARIES}
		CppRob::utils
)

add_library(lib_${PROJECT_NAME} 
		ekf_slam.cpp
		include/ekf_slam.h
)

target_link_libraries(lib_${PROJECT_NAME} PUBLIC
		Eigen3::Eigen
		Boost::iostreams
		Boost::system
		Boost::filesystem
		${Gnuplot_LIBRARIES}
		CppRob::utils
)