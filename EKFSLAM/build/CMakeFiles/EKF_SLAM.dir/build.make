# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.24

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.24.0/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.24.0/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/NathanDurocher/cppyourself/CppRobotics/EKFSLAM

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/NathanDurocher/cppyourself/CppRobotics/EKFSLAM/build

# Include any dependencies generated for this target.
include CMakeFiles/EKF_SLAM.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/EKF_SLAM.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/EKF_SLAM.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/EKF_SLAM.dir/flags.make

CMakeFiles/EKF_SLAM.dir/SLAM_EKF.cpp.o: CMakeFiles/EKF_SLAM.dir/flags.make
CMakeFiles/EKF_SLAM.dir/SLAM_EKF.cpp.o: /Users/NathanDurocher/cppyourself/CppRobotics/EKFSLAM/SLAM_EKF.cpp
CMakeFiles/EKF_SLAM.dir/SLAM_EKF.cpp.o: CMakeFiles/EKF_SLAM.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/NathanDurocher/cppyourself/CppRobotics/EKFSLAM/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/EKF_SLAM.dir/SLAM_EKF.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/EKF_SLAM.dir/SLAM_EKF.cpp.o -MF CMakeFiles/EKF_SLAM.dir/SLAM_EKF.cpp.o.d -o CMakeFiles/EKF_SLAM.dir/SLAM_EKF.cpp.o -c /Users/NathanDurocher/cppyourself/CppRobotics/EKFSLAM/SLAM_EKF.cpp

CMakeFiles/EKF_SLAM.dir/SLAM_EKF.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/EKF_SLAM.dir/SLAM_EKF.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/NathanDurocher/cppyourself/CppRobotics/EKFSLAM/SLAM_EKF.cpp > CMakeFiles/EKF_SLAM.dir/SLAM_EKF.cpp.i

CMakeFiles/EKF_SLAM.dir/SLAM_EKF.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/EKF_SLAM.dir/SLAM_EKF.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/NathanDurocher/cppyourself/CppRobotics/EKFSLAM/SLAM_EKF.cpp -o CMakeFiles/EKF_SLAM.dir/SLAM_EKF.cpp.s

CMakeFiles/EKF_SLAM.dir/EKF.cpp.o: CMakeFiles/EKF_SLAM.dir/flags.make
CMakeFiles/EKF_SLAM.dir/EKF.cpp.o: /Users/NathanDurocher/cppyourself/CppRobotics/EKFSLAM/EKF.cpp
CMakeFiles/EKF_SLAM.dir/EKF.cpp.o: CMakeFiles/EKF_SLAM.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/NathanDurocher/cppyourself/CppRobotics/EKFSLAM/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/EKF_SLAM.dir/EKF.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/EKF_SLAM.dir/EKF.cpp.o -MF CMakeFiles/EKF_SLAM.dir/EKF.cpp.o.d -o CMakeFiles/EKF_SLAM.dir/EKF.cpp.o -c /Users/NathanDurocher/cppyourself/CppRobotics/EKFSLAM/EKF.cpp

CMakeFiles/EKF_SLAM.dir/EKF.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/EKF_SLAM.dir/EKF.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/NathanDurocher/cppyourself/CppRobotics/EKFSLAM/EKF.cpp > CMakeFiles/EKF_SLAM.dir/EKF.cpp.i

CMakeFiles/EKF_SLAM.dir/EKF.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/EKF_SLAM.dir/EKF.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/NathanDurocher/cppyourself/CppRobotics/EKFSLAM/EKF.cpp -o CMakeFiles/EKF_SLAM.dir/EKF.cpp.s

CMakeFiles/EKF_SLAM.dir/robot.cpp.o: CMakeFiles/EKF_SLAM.dir/flags.make
CMakeFiles/EKF_SLAM.dir/robot.cpp.o: /Users/NathanDurocher/cppyourself/CppRobotics/EKFSLAM/robot.cpp
CMakeFiles/EKF_SLAM.dir/robot.cpp.o: CMakeFiles/EKF_SLAM.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/NathanDurocher/cppyourself/CppRobotics/EKFSLAM/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/EKF_SLAM.dir/robot.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/EKF_SLAM.dir/robot.cpp.o -MF CMakeFiles/EKF_SLAM.dir/robot.cpp.o.d -o CMakeFiles/EKF_SLAM.dir/robot.cpp.o -c /Users/NathanDurocher/cppyourself/CppRobotics/EKFSLAM/robot.cpp

CMakeFiles/EKF_SLAM.dir/robot.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/EKF_SLAM.dir/robot.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/NathanDurocher/cppyourself/CppRobotics/EKFSLAM/robot.cpp > CMakeFiles/EKF_SLAM.dir/robot.cpp.i

CMakeFiles/EKF_SLAM.dir/robot.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/EKF_SLAM.dir/robot.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/NathanDurocher/cppyourself/CppRobotics/EKFSLAM/robot.cpp -o CMakeFiles/EKF_SLAM.dir/robot.cpp.s

# Object files for target EKF_SLAM
EKF_SLAM_OBJECTS = \
"CMakeFiles/EKF_SLAM.dir/SLAM_EKF.cpp.o" \
"CMakeFiles/EKF_SLAM.dir/EKF.cpp.o" \
"CMakeFiles/EKF_SLAM.dir/robot.cpp.o"

# External object files for target EKF_SLAM
EKF_SLAM_EXTERNAL_OBJECTS =

EKF_SLAM: CMakeFiles/EKF_SLAM.dir/SLAM_EKF.cpp.o
EKF_SLAM: CMakeFiles/EKF_SLAM.dir/EKF.cpp.o
EKF_SLAM: CMakeFiles/EKF_SLAM.dir/robot.cpp.o
EKF_SLAM: CMakeFiles/EKF_SLAM.dir/build.make
EKF_SLAM: /usr/local/lib/libboost_contract-mt.dylib
EKF_SLAM: /usr/local/lib/libboost_coroutine-mt.dylib
EKF_SLAM: /usr/local/lib/libboost_date_time-mt.dylib
EKF_SLAM: /usr/local/lib/libboost_fiber-mt.dylib
EKF_SLAM: /usr/local/lib/libboost_graph-mt.dylib
EKF_SLAM: /usr/local/lib/libboost_iostreams-mt.dylib
EKF_SLAM: /usr/local/lib/libboost_json-mt.dylib
EKF_SLAM: /usr/local/lib/libboost_locale-mt.dylib
EKF_SLAM: /usr/local/lib/libboost_log_setup-mt.dylib
EKF_SLAM: /usr/local/lib/libboost_math_c99-mt.dylib
EKF_SLAM: /usr/local/lib/libboost_math_c99f-mt.dylib
EKF_SLAM: /usr/local/lib/libboost_math_c99l-mt.dylib
EKF_SLAM: /usr/local/lib/libboost_math_tr1-mt.dylib
EKF_SLAM: /usr/local/lib/libboost_math_tr1f-mt.dylib
EKF_SLAM: /usr/local/lib/libboost_math_tr1l-mt.dylib
EKF_SLAM: /usr/local/lib/libboost_nowide-mt.dylib
EKF_SLAM: /usr/local/lib/libboost_prg_exec_monitor-mt.dylib
EKF_SLAM: /usr/local/lib/libboost_program_options-mt.dylib
EKF_SLAM: /usr/local/lib/libboost_random-mt.dylib
EKF_SLAM: /usr/local/lib/libboost_stacktrace_addr2line-mt.dylib
EKF_SLAM: /usr/local/lib/libboost_stacktrace_basic-mt.dylib
EKF_SLAM: /usr/local/lib/libboost_stacktrace_noop-mt.dylib
EKF_SLAM: /usr/local/lib/libboost_system-mt.dylib
EKF_SLAM: /usr/local/lib/libboost_test_exec_monitor-mt.a
EKF_SLAM: /usr/local/lib/libboost_timer-mt.dylib
EKF_SLAM: /usr/local/lib/libboost_type_erasure-mt.dylib
EKF_SLAM: /usr/local/lib/libboost_unit_test_framework-mt.dylib
EKF_SLAM: /usr/local/lib/libboost_wave-mt.dylib
EKF_SLAM: /usr/local/lib/libboost_wserialization-mt.dylib
EKF_SLAM: /usr/local/lib/libboost_context-mt.dylib
EKF_SLAM: /usr/local/lib/libboost_container-mt.dylib
EKF_SLAM: /usr/local/lib/libboost_log-mt.dylib
EKF_SLAM: /usr/local/lib/libboost_regex-mt.dylib
EKF_SLAM: /usr/local/lib/libboost_chrono-mt.dylib
EKF_SLAM: /usr/local/lib/libboost_filesystem-mt.dylib
EKF_SLAM: /usr/local/lib/libboost_atomic-mt.dylib
EKF_SLAM: /usr/local/lib/libboost_thread-mt.dylib
EKF_SLAM: /usr/local/lib/libboost_serialization-mt.dylib
EKF_SLAM: CMakeFiles/EKF_SLAM.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/NathanDurocher/cppyourself/CppRobotics/EKFSLAM/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable EKF_SLAM"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/EKF_SLAM.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/EKF_SLAM.dir/build: EKF_SLAM
.PHONY : CMakeFiles/EKF_SLAM.dir/build

CMakeFiles/EKF_SLAM.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/EKF_SLAM.dir/cmake_clean.cmake
.PHONY : CMakeFiles/EKF_SLAM.dir/clean

CMakeFiles/EKF_SLAM.dir/depend:
	cd /Users/NathanDurocher/cppyourself/CppRobotics/EKFSLAM/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/NathanDurocher/cppyourself/CppRobotics/EKFSLAM /Users/NathanDurocher/cppyourself/CppRobotics/EKFSLAM /Users/NathanDurocher/cppyourself/CppRobotics/EKFSLAM/build /Users/NathanDurocher/cppyourself/CppRobotics/EKFSLAM/build /Users/NathanDurocher/cppyourself/CppRobotics/EKFSLAM/build/CMakeFiles/EKF_SLAM.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/EKF_SLAM.dir/depend

