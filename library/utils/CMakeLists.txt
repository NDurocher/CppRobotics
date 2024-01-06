cmake_minimum_required(VERSION 3.12)
project(utils VERSION 0.1.0
        DESCRIPTION "A utility library"
        LANGUAGES C CXX)

# Set the library sources
set(LIBRARY_SOURCES
        my_time.cpp
)

# Set the library headers
set(LIBRARY_HEADERS
        include/my_time.h
)

# Create the library target
add_library(${PROJECT_NAME} ${LIBRARY_SOURCES} ${LIBRARY_HEADERS})

# Set the include directories for the library
target_include_directories(${PROJECT_NAME} PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# create alias
add_library(CppRob::${PROJECT_NAME} ALIAS ${PROJECT_NAME})