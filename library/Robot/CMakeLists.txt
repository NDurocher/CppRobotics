cmake_minimum_required(VERSION 3.12)
project(robot VERSION 0.1.0
        DESCRIPTION "A base differential drive robot library"
        LANGUAGES C CXX)

# Set the library sources
set(LIBRARY_SOURCES
    robot.cpp
)

# Set the library headers
set(LIBRARY_HEADERS
    include/robot.h
)

find_package (Eigen3 3.3 REQUIRED NO_MODULE
)
include_directories(${EIGEN_INCLUDE_DIRS})

# Create the library target
add_library(${PROJECT_NAME} ${LIBRARY_SOURCES} ${LIBRARY_HEADERS})

target_link_libraries (${PROJECT_NAME} PUBLIC
                        Eigen3::Eigen
)

# Set the include directories for the library
target_include_directories(${PROJECT_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# create alias
add_library(CppRob::${PROJECT_NAME} ALIAS ${PROJECT_NAME})