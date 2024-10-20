cmake_minimum_required(VERSION 3.12)

project(utils VERSION 0.1.0
        DESCRIPTION "A utility library"
        LANGUAGES C CXX)

find_package(Eigen3 3.3 REQUIRED NO_MODULE
)

# Create the library target
add_library(${PROJECT_NAME} STATIC
        include/utils/robot.h
        robot.cpp
        include/utils/my_time.h
        my_time.cpp
        include/utils/obstacle.h
        obstacle.cpp)

set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)

# # Set the library sources
# target_sources(${PROJECT_NAME}
# )

# Set the include directories for the library
target_include_directories(${PROJECT_NAME} PUBLIC include)

target_link_libraries(${PROJECT_NAME} PUBLIC
        Eigen3::Eigen
)

# create alias
add_library(CppRob::${PROJECT_NAME} ALIAS ${PROJECT_NAME})