cmake_minimum_required(VERSION 3.15)
project(BayesEstimator)

# Set the library sources
set(LIBRARY_SOURCES
    BayesEstimator.cpp
)

# Set the library headers
set(LIBRARY_HEADERS
    include/BayesEstimator.h
)

# Create the library target
add_library(${PROJECT_NAME} ${LIBRARY_SOURCES} ${LIBRARY_HEADERS})

# Set the include directories for the library
target_include_directories(${PROJECT_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Set the library version
set_target_properties(${PROJECT_NAME} PROPERTIES
    VERSION 1.0.0
    SOVERSION 1
)

# create alias
add_library(CppRob::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

# add the executable
#add_executable(${PROJECT_NAME}
#				src/main.cpp
#				include/${PROJECT_NAME}/BayesEstimator.cpp
#				include/${PROJECT_NAME}/BayesEstimator.h
#) 


