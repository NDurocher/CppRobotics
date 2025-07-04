cmake_minimum_required(VERSION 3.15)

# set the project name
project(Visual_Odometry)

# guard against in-source builds
if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
    message(FATAL_ERROR "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there. You may need to remove CMakeCache.txt. ")
endif()

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenCV REQUIRED)

include_directories(${EIGEN_INCLUDE_DIRS}
					${OpenCV_INCLUDE_DIRS}
					include/
)

# add library
add_library(${PROJECT_NAME}_lib
			include/utils.h
			include/VO.h
			VO.cpp
			include/feature_detector.h
			feature_detector.cpp
			include/image_loader.h
			image_loader.cpp
			include/pose_loader.h
			pose_loader.cpp
)

target_link_libraries(${PROJECT_NAME}_lib PUBLIC
						Eigen3::Eigen
						${OpenCV_LIBS}
						CppRob::utils
)

# create alias
add_library(CppRob::${PROJECT_NAME}_lib ALIAS ${PROJECT_NAME}_lib)

# add the executable
add_executable(${PROJECT_NAME}
		main.cpp
)

target_link_libraries (${PROJECT_NAME} PRIVATE
		${PROJECT_NAME}_lib
)