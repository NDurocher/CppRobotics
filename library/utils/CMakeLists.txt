cmake_minimum_required(VERSION 3.12)
project(utils VERSION 0.1.0
        DESCRIPTION "A utility library"
        LANGUAGES C CXX)

# Set the library sources
set(LIBRARY_SOURCES
        my_time.cpp
        robot.cpp
)

# Set the library headers
set(LIBRARY_HEADERS
        include/my_time.h
        include/robot.h
)

find_package(Eigen3 3.3 REQUIRED NO_MODULE
)

# Create the library target
add_library(${PROJECT_NAME} ${LIBRARY_SOURCES} ${LIBRARY_HEADERS})

include_directories(${EIGEN_INCLUDE_DIRS}
)

# Set the include directories for the library
target_include_directories(${PROJECT_NAME} PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(${PROJECT_NAME} PUBLIC
        Eigen3::Eigen
)

# create alias
add_library(CppRob::${PROJECT_NAME} ALIAS ${PROJECT_NAME})