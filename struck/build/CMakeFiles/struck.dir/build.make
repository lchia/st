# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/kog/lchpro/app/screening_tracking/struck

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/kog/lchpro/app/screening_tracking/struck/build

# Include any dependencies generated for this target.
include CMakeFiles/struck.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/struck.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/struck.dir/flags.make

CMakeFiles/struck.dir/src/ImageRep.cpp.o: CMakeFiles/struck.dir/flags.make
CMakeFiles/struck.dir/src/ImageRep.cpp.o: ../src/ImageRep.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/kog/lchpro/app/screening_tracking/struck/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/struck.dir/src/ImageRep.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/struck.dir/src/ImageRep.cpp.o -c /home/kog/lchpro/app/screening_tracking/struck/src/ImageRep.cpp

CMakeFiles/struck.dir/src/ImageRep.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/struck.dir/src/ImageRep.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/kog/lchpro/app/screening_tracking/struck/src/ImageRep.cpp > CMakeFiles/struck.dir/src/ImageRep.cpp.i

CMakeFiles/struck.dir/src/ImageRep.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/struck.dir/src/ImageRep.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/kog/lchpro/app/screening_tracking/struck/src/ImageRep.cpp -o CMakeFiles/struck.dir/src/ImageRep.cpp.s

CMakeFiles/struck.dir/src/ImageRep.cpp.o.requires:

.PHONY : CMakeFiles/struck.dir/src/ImageRep.cpp.o.requires

CMakeFiles/struck.dir/src/ImageRep.cpp.o.provides: CMakeFiles/struck.dir/src/ImageRep.cpp.o.requires
	$(MAKE) -f CMakeFiles/struck.dir/build.make CMakeFiles/struck.dir/src/ImageRep.cpp.o.provides.build
.PHONY : CMakeFiles/struck.dir/src/ImageRep.cpp.o.provides

CMakeFiles/struck.dir/src/ImageRep.cpp.o.provides.build: CMakeFiles/struck.dir/src/ImageRep.cpp.o


CMakeFiles/struck.dir/src/HaarFeature.cpp.o: CMakeFiles/struck.dir/flags.make
CMakeFiles/struck.dir/src/HaarFeature.cpp.o: ../src/HaarFeature.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/kog/lchpro/app/screening_tracking/struck/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/struck.dir/src/HaarFeature.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/struck.dir/src/HaarFeature.cpp.o -c /home/kog/lchpro/app/screening_tracking/struck/src/HaarFeature.cpp

CMakeFiles/struck.dir/src/HaarFeature.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/struck.dir/src/HaarFeature.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/kog/lchpro/app/screening_tracking/struck/src/HaarFeature.cpp > CMakeFiles/struck.dir/src/HaarFeature.cpp.i

CMakeFiles/struck.dir/src/HaarFeature.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/struck.dir/src/HaarFeature.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/kog/lchpro/app/screening_tracking/struck/src/HaarFeature.cpp -o CMakeFiles/struck.dir/src/HaarFeature.cpp.s

CMakeFiles/struck.dir/src/HaarFeature.cpp.o.requires:

.PHONY : CMakeFiles/struck.dir/src/HaarFeature.cpp.o.requires

CMakeFiles/struck.dir/src/HaarFeature.cpp.o.provides: CMakeFiles/struck.dir/src/HaarFeature.cpp.o.requires
	$(MAKE) -f CMakeFiles/struck.dir/build.make CMakeFiles/struck.dir/src/HaarFeature.cpp.o.provides.build
.PHONY : CMakeFiles/struck.dir/src/HaarFeature.cpp.o.provides

CMakeFiles/struck.dir/src/HaarFeature.cpp.o.provides.build: CMakeFiles/struck.dir/src/HaarFeature.cpp.o


CMakeFiles/struck.dir/src/MultiFeatures.cpp.o: CMakeFiles/struck.dir/flags.make
CMakeFiles/struck.dir/src/MultiFeatures.cpp.o: ../src/MultiFeatures.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/kog/lchpro/app/screening_tracking/struck/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/struck.dir/src/MultiFeatures.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/struck.dir/src/MultiFeatures.cpp.o -c /home/kog/lchpro/app/screening_tracking/struck/src/MultiFeatures.cpp

CMakeFiles/struck.dir/src/MultiFeatures.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/struck.dir/src/MultiFeatures.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/kog/lchpro/app/screening_tracking/struck/src/MultiFeatures.cpp > CMakeFiles/struck.dir/src/MultiFeatures.cpp.i

CMakeFiles/struck.dir/src/MultiFeatures.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/struck.dir/src/MultiFeatures.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/kog/lchpro/app/screening_tracking/struck/src/MultiFeatures.cpp -o CMakeFiles/struck.dir/src/MultiFeatures.cpp.s

CMakeFiles/struck.dir/src/MultiFeatures.cpp.o.requires:

.PHONY : CMakeFiles/struck.dir/src/MultiFeatures.cpp.o.requires

CMakeFiles/struck.dir/src/MultiFeatures.cpp.o.provides: CMakeFiles/struck.dir/src/MultiFeatures.cpp.o.requires
	$(MAKE) -f CMakeFiles/struck.dir/build.make CMakeFiles/struck.dir/src/MultiFeatures.cpp.o.provides.build
.PHONY : CMakeFiles/struck.dir/src/MultiFeatures.cpp.o.provides

CMakeFiles/struck.dir/src/MultiFeatures.cpp.o.provides.build: CMakeFiles/struck.dir/src/MultiFeatures.cpp.o


CMakeFiles/struck.dir/src/Features.cpp.o: CMakeFiles/struck.dir/flags.make
CMakeFiles/struck.dir/src/Features.cpp.o: ../src/Features.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/kog/lchpro/app/screening_tracking/struck/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/struck.dir/src/Features.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/struck.dir/src/Features.cpp.o -c /home/kog/lchpro/app/screening_tracking/struck/src/Features.cpp

CMakeFiles/struck.dir/src/Features.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/struck.dir/src/Features.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/kog/lchpro/app/screening_tracking/struck/src/Features.cpp > CMakeFiles/struck.dir/src/Features.cpp.i

CMakeFiles/struck.dir/src/Features.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/struck.dir/src/Features.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/kog/lchpro/app/screening_tracking/struck/src/Features.cpp -o CMakeFiles/struck.dir/src/Features.cpp.s

CMakeFiles/struck.dir/src/Features.cpp.o.requires:

.PHONY : CMakeFiles/struck.dir/src/Features.cpp.o.requires

CMakeFiles/struck.dir/src/Features.cpp.o.provides: CMakeFiles/struck.dir/src/Features.cpp.o.requires
	$(MAKE) -f CMakeFiles/struck.dir/build.make CMakeFiles/struck.dir/src/Features.cpp.o.provides.build
.PHONY : CMakeFiles/struck.dir/src/Features.cpp.o.provides

CMakeFiles/struck.dir/src/Features.cpp.o.provides.build: CMakeFiles/struck.dir/src/Features.cpp.o


CMakeFiles/struck.dir/src/Config.cpp.o: CMakeFiles/struck.dir/flags.make
CMakeFiles/struck.dir/src/Config.cpp.o: ../src/Config.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/kog/lchpro/app/screening_tracking/struck/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/struck.dir/src/Config.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/struck.dir/src/Config.cpp.o -c /home/kog/lchpro/app/screening_tracking/struck/src/Config.cpp

CMakeFiles/struck.dir/src/Config.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/struck.dir/src/Config.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/kog/lchpro/app/screening_tracking/struck/src/Config.cpp > CMakeFiles/struck.dir/src/Config.cpp.i

CMakeFiles/struck.dir/src/Config.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/struck.dir/src/Config.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/kog/lchpro/app/screening_tracking/struck/src/Config.cpp -o CMakeFiles/struck.dir/src/Config.cpp.s

CMakeFiles/struck.dir/src/Config.cpp.o.requires:

.PHONY : CMakeFiles/struck.dir/src/Config.cpp.o.requires

CMakeFiles/struck.dir/src/Config.cpp.o.provides: CMakeFiles/struck.dir/src/Config.cpp.o.requires
	$(MAKE) -f CMakeFiles/struck.dir/build.make CMakeFiles/struck.dir/src/Config.cpp.o.provides.build
.PHONY : CMakeFiles/struck.dir/src/Config.cpp.o.provides

CMakeFiles/struck.dir/src/Config.cpp.o.provides.build: CMakeFiles/struck.dir/src/Config.cpp.o


CMakeFiles/struck.dir/src/Sampler.cpp.o: CMakeFiles/struck.dir/flags.make
CMakeFiles/struck.dir/src/Sampler.cpp.o: ../src/Sampler.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/kog/lchpro/app/screening_tracking/struck/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/struck.dir/src/Sampler.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/struck.dir/src/Sampler.cpp.o -c /home/kog/lchpro/app/screening_tracking/struck/src/Sampler.cpp

CMakeFiles/struck.dir/src/Sampler.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/struck.dir/src/Sampler.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/kog/lchpro/app/screening_tracking/struck/src/Sampler.cpp > CMakeFiles/struck.dir/src/Sampler.cpp.i

CMakeFiles/struck.dir/src/Sampler.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/struck.dir/src/Sampler.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/kog/lchpro/app/screening_tracking/struck/src/Sampler.cpp -o CMakeFiles/struck.dir/src/Sampler.cpp.s

CMakeFiles/struck.dir/src/Sampler.cpp.o.requires:

.PHONY : CMakeFiles/struck.dir/src/Sampler.cpp.o.requires

CMakeFiles/struck.dir/src/Sampler.cpp.o.provides: CMakeFiles/struck.dir/src/Sampler.cpp.o.requires
	$(MAKE) -f CMakeFiles/struck.dir/build.make CMakeFiles/struck.dir/src/Sampler.cpp.o.provides.build
.PHONY : CMakeFiles/struck.dir/src/Sampler.cpp.o.provides

CMakeFiles/struck.dir/src/Sampler.cpp.o.provides.build: CMakeFiles/struck.dir/src/Sampler.cpp.o


CMakeFiles/struck.dir/src/HaarFeatures.cpp.o: CMakeFiles/struck.dir/flags.make
CMakeFiles/struck.dir/src/HaarFeatures.cpp.o: ../src/HaarFeatures.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/kog/lchpro/app/screening_tracking/struck/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object CMakeFiles/struck.dir/src/HaarFeatures.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/struck.dir/src/HaarFeatures.cpp.o -c /home/kog/lchpro/app/screening_tracking/struck/src/HaarFeatures.cpp

CMakeFiles/struck.dir/src/HaarFeatures.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/struck.dir/src/HaarFeatures.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/kog/lchpro/app/screening_tracking/struck/src/HaarFeatures.cpp > CMakeFiles/struck.dir/src/HaarFeatures.cpp.i

CMakeFiles/struck.dir/src/HaarFeatures.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/struck.dir/src/HaarFeatures.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/kog/lchpro/app/screening_tracking/struck/src/HaarFeatures.cpp -o CMakeFiles/struck.dir/src/HaarFeatures.cpp.s

CMakeFiles/struck.dir/src/HaarFeatures.cpp.o.requires:

.PHONY : CMakeFiles/struck.dir/src/HaarFeatures.cpp.o.requires

CMakeFiles/struck.dir/src/HaarFeatures.cpp.o.provides: CMakeFiles/struck.dir/src/HaarFeatures.cpp.o.requires
	$(MAKE) -f CMakeFiles/struck.dir/build.make CMakeFiles/struck.dir/src/HaarFeatures.cpp.o.provides.build
.PHONY : CMakeFiles/struck.dir/src/HaarFeatures.cpp.o.provides

CMakeFiles/struck.dir/src/HaarFeatures.cpp.o.provides.build: CMakeFiles/struck.dir/src/HaarFeatures.cpp.o


CMakeFiles/struck.dir/src/GraphUtils/GraphUtils.cpp.o: CMakeFiles/struck.dir/flags.make
CMakeFiles/struck.dir/src/GraphUtils/GraphUtils.cpp.o: ../src/GraphUtils/GraphUtils.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/kog/lchpro/app/screening_tracking/struck/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object CMakeFiles/struck.dir/src/GraphUtils/GraphUtils.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/struck.dir/src/GraphUtils/GraphUtils.cpp.o -c /home/kog/lchpro/app/screening_tracking/struck/src/GraphUtils/GraphUtils.cpp

CMakeFiles/struck.dir/src/GraphUtils/GraphUtils.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/struck.dir/src/GraphUtils/GraphUtils.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/kog/lchpro/app/screening_tracking/struck/src/GraphUtils/GraphUtils.cpp > CMakeFiles/struck.dir/src/GraphUtils/GraphUtils.cpp.i

CMakeFiles/struck.dir/src/GraphUtils/GraphUtils.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/struck.dir/src/GraphUtils/GraphUtils.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/kog/lchpro/app/screening_tracking/struck/src/GraphUtils/GraphUtils.cpp -o CMakeFiles/struck.dir/src/GraphUtils/GraphUtils.cpp.s

CMakeFiles/struck.dir/src/GraphUtils/GraphUtils.cpp.o.requires:

.PHONY : CMakeFiles/struck.dir/src/GraphUtils/GraphUtils.cpp.o.requires

CMakeFiles/struck.dir/src/GraphUtils/GraphUtils.cpp.o.provides: CMakeFiles/struck.dir/src/GraphUtils/GraphUtils.cpp.o.requires
	$(MAKE) -f CMakeFiles/struck.dir/build.make CMakeFiles/struck.dir/src/GraphUtils/GraphUtils.cpp.o.provides.build
.PHONY : CMakeFiles/struck.dir/src/GraphUtils/GraphUtils.cpp.o.provides

CMakeFiles/struck.dir/src/GraphUtils/GraphUtils.cpp.o.provides.build: CMakeFiles/struck.dir/src/GraphUtils/GraphUtils.cpp.o


CMakeFiles/struck.dir/src/RawFeatures.cpp.o: CMakeFiles/struck.dir/flags.make
CMakeFiles/struck.dir/src/RawFeatures.cpp.o: ../src/RawFeatures.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/kog/lchpro/app/screening_tracking/struck/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building CXX object CMakeFiles/struck.dir/src/RawFeatures.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/struck.dir/src/RawFeatures.cpp.o -c /home/kog/lchpro/app/screening_tracking/struck/src/RawFeatures.cpp

CMakeFiles/struck.dir/src/RawFeatures.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/struck.dir/src/RawFeatures.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/kog/lchpro/app/screening_tracking/struck/src/RawFeatures.cpp > CMakeFiles/struck.dir/src/RawFeatures.cpp.i

CMakeFiles/struck.dir/src/RawFeatures.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/struck.dir/src/RawFeatures.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/kog/lchpro/app/screening_tracking/struck/src/RawFeatures.cpp -o CMakeFiles/struck.dir/src/RawFeatures.cpp.s

CMakeFiles/struck.dir/src/RawFeatures.cpp.o.requires:

.PHONY : CMakeFiles/struck.dir/src/RawFeatures.cpp.o.requires

CMakeFiles/struck.dir/src/RawFeatures.cpp.o.provides: CMakeFiles/struck.dir/src/RawFeatures.cpp.o.requires
	$(MAKE) -f CMakeFiles/struck.dir/build.make CMakeFiles/struck.dir/src/RawFeatures.cpp.o.provides.build
.PHONY : CMakeFiles/struck.dir/src/RawFeatures.cpp.o.provides

CMakeFiles/struck.dir/src/RawFeatures.cpp.o.provides.build: CMakeFiles/struck.dir/src/RawFeatures.cpp.o


CMakeFiles/struck.dir/src/main.cpp.o: CMakeFiles/struck.dir/flags.make
CMakeFiles/struck.dir/src/main.cpp.o: ../src/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/kog/lchpro/app/screening_tracking/struck/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Building CXX object CMakeFiles/struck.dir/src/main.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/struck.dir/src/main.cpp.o -c /home/kog/lchpro/app/screening_tracking/struck/src/main.cpp

CMakeFiles/struck.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/struck.dir/src/main.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/kog/lchpro/app/screening_tracking/struck/src/main.cpp > CMakeFiles/struck.dir/src/main.cpp.i

CMakeFiles/struck.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/struck.dir/src/main.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/kog/lchpro/app/screening_tracking/struck/src/main.cpp -o CMakeFiles/struck.dir/src/main.cpp.s

CMakeFiles/struck.dir/src/main.cpp.o.requires:

.PHONY : CMakeFiles/struck.dir/src/main.cpp.o.requires

CMakeFiles/struck.dir/src/main.cpp.o.provides: CMakeFiles/struck.dir/src/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/struck.dir/build.make CMakeFiles/struck.dir/src/main.cpp.o.provides.build
.PHONY : CMakeFiles/struck.dir/src/main.cpp.o.provides

CMakeFiles/struck.dir/src/main.cpp.o.provides.build: CMakeFiles/struck.dir/src/main.cpp.o


CMakeFiles/struck.dir/src/HistogramFeatures.cpp.o: CMakeFiles/struck.dir/flags.make
CMakeFiles/struck.dir/src/HistogramFeatures.cpp.o: ../src/HistogramFeatures.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/kog/lchpro/app/screening_tracking/struck/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_11) "Building CXX object CMakeFiles/struck.dir/src/HistogramFeatures.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/struck.dir/src/HistogramFeatures.cpp.o -c /home/kog/lchpro/app/screening_tracking/struck/src/HistogramFeatures.cpp

CMakeFiles/struck.dir/src/HistogramFeatures.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/struck.dir/src/HistogramFeatures.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/kog/lchpro/app/screening_tracking/struck/src/HistogramFeatures.cpp > CMakeFiles/struck.dir/src/HistogramFeatures.cpp.i

CMakeFiles/struck.dir/src/HistogramFeatures.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/struck.dir/src/HistogramFeatures.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/kog/lchpro/app/screening_tracking/struck/src/HistogramFeatures.cpp -o CMakeFiles/struck.dir/src/HistogramFeatures.cpp.s

CMakeFiles/struck.dir/src/HistogramFeatures.cpp.o.requires:

.PHONY : CMakeFiles/struck.dir/src/HistogramFeatures.cpp.o.requires

CMakeFiles/struck.dir/src/HistogramFeatures.cpp.o.provides: CMakeFiles/struck.dir/src/HistogramFeatures.cpp.o.requires
	$(MAKE) -f CMakeFiles/struck.dir/build.make CMakeFiles/struck.dir/src/HistogramFeatures.cpp.o.provides.build
.PHONY : CMakeFiles/struck.dir/src/HistogramFeatures.cpp.o.provides

CMakeFiles/struck.dir/src/HistogramFeatures.cpp.o.provides.build: CMakeFiles/struck.dir/src/HistogramFeatures.cpp.o


CMakeFiles/struck.dir/src/LaRank.cpp.o: CMakeFiles/struck.dir/flags.make
CMakeFiles/struck.dir/src/LaRank.cpp.o: ../src/LaRank.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/kog/lchpro/app/screening_tracking/struck/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_12) "Building CXX object CMakeFiles/struck.dir/src/LaRank.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/struck.dir/src/LaRank.cpp.o -c /home/kog/lchpro/app/screening_tracking/struck/src/LaRank.cpp

CMakeFiles/struck.dir/src/LaRank.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/struck.dir/src/LaRank.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/kog/lchpro/app/screening_tracking/struck/src/LaRank.cpp > CMakeFiles/struck.dir/src/LaRank.cpp.i

CMakeFiles/struck.dir/src/LaRank.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/struck.dir/src/LaRank.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/kog/lchpro/app/screening_tracking/struck/src/LaRank.cpp -o CMakeFiles/struck.dir/src/LaRank.cpp.s

CMakeFiles/struck.dir/src/LaRank.cpp.o.requires:

.PHONY : CMakeFiles/struck.dir/src/LaRank.cpp.o.requires

CMakeFiles/struck.dir/src/LaRank.cpp.o.provides: CMakeFiles/struck.dir/src/LaRank.cpp.o.requires
	$(MAKE) -f CMakeFiles/struck.dir/build.make CMakeFiles/struck.dir/src/LaRank.cpp.o.provides.build
.PHONY : CMakeFiles/struck.dir/src/LaRank.cpp.o.provides

CMakeFiles/struck.dir/src/LaRank.cpp.o.provides.build: CMakeFiles/struck.dir/src/LaRank.cpp.o


CMakeFiles/struck.dir/src/Tracker.cpp.o: CMakeFiles/struck.dir/flags.make
CMakeFiles/struck.dir/src/Tracker.cpp.o: ../src/Tracker.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/kog/lchpro/app/screening_tracking/struck/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_13) "Building CXX object CMakeFiles/struck.dir/src/Tracker.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/struck.dir/src/Tracker.cpp.o -c /home/kog/lchpro/app/screening_tracking/struck/src/Tracker.cpp

CMakeFiles/struck.dir/src/Tracker.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/struck.dir/src/Tracker.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/kog/lchpro/app/screening_tracking/struck/src/Tracker.cpp > CMakeFiles/struck.dir/src/Tracker.cpp.i

CMakeFiles/struck.dir/src/Tracker.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/struck.dir/src/Tracker.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/kog/lchpro/app/screening_tracking/struck/src/Tracker.cpp -o CMakeFiles/struck.dir/src/Tracker.cpp.s

CMakeFiles/struck.dir/src/Tracker.cpp.o.requires:

.PHONY : CMakeFiles/struck.dir/src/Tracker.cpp.o.requires

CMakeFiles/struck.dir/src/Tracker.cpp.o.provides: CMakeFiles/struck.dir/src/Tracker.cpp.o.requires
	$(MAKE) -f CMakeFiles/struck.dir/build.make CMakeFiles/struck.dir/src/Tracker.cpp.o.provides.build
.PHONY : CMakeFiles/struck.dir/src/Tracker.cpp.o.provides

CMakeFiles/struck.dir/src/Tracker.cpp.o.provides.build: CMakeFiles/struck.dir/src/Tracker.cpp.o


# Object files for target struck
struck_OBJECTS = \
"CMakeFiles/struck.dir/src/ImageRep.cpp.o" \
"CMakeFiles/struck.dir/src/HaarFeature.cpp.o" \
"CMakeFiles/struck.dir/src/MultiFeatures.cpp.o" \
"CMakeFiles/struck.dir/src/Features.cpp.o" \
"CMakeFiles/struck.dir/src/Config.cpp.o" \
"CMakeFiles/struck.dir/src/Sampler.cpp.o" \
"CMakeFiles/struck.dir/src/HaarFeatures.cpp.o" \
"CMakeFiles/struck.dir/src/GraphUtils/GraphUtils.cpp.o" \
"CMakeFiles/struck.dir/src/RawFeatures.cpp.o" \
"CMakeFiles/struck.dir/src/main.cpp.o" \
"CMakeFiles/struck.dir/src/HistogramFeatures.cpp.o" \
"CMakeFiles/struck.dir/src/LaRank.cpp.o" \
"CMakeFiles/struck.dir/src/Tracker.cpp.o"

# External object files for target struck
struck_EXTERNAL_OBJECTS =

bin/struck: CMakeFiles/struck.dir/src/ImageRep.cpp.o
bin/struck: CMakeFiles/struck.dir/src/HaarFeature.cpp.o
bin/struck: CMakeFiles/struck.dir/src/MultiFeatures.cpp.o
bin/struck: CMakeFiles/struck.dir/src/Features.cpp.o
bin/struck: CMakeFiles/struck.dir/src/Config.cpp.o
bin/struck: CMakeFiles/struck.dir/src/Sampler.cpp.o
bin/struck: CMakeFiles/struck.dir/src/HaarFeatures.cpp.o
bin/struck: CMakeFiles/struck.dir/src/GraphUtils/GraphUtils.cpp.o
bin/struck: CMakeFiles/struck.dir/src/RawFeatures.cpp.o
bin/struck: CMakeFiles/struck.dir/src/main.cpp.o
bin/struck: CMakeFiles/struck.dir/src/HistogramFeatures.cpp.o
bin/struck: CMakeFiles/struck.dir/src/LaRank.cpp.o
bin/struck: CMakeFiles/struck.dir/src/Tracker.cpp.o
bin/struck: CMakeFiles/struck.dir/build.make
bin/struck: /home/kog/local/lib/libopencv_cudabgsegm.so.3.1.0
bin/struck: /home/kog/local/lib/libopencv_cudaobjdetect.so.3.1.0
bin/struck: /home/kog/local/lib/libopencv_cudastereo.so.3.1.0
bin/struck: /home/kog/local/lib/libopencv_shape.so.3.1.0
bin/struck: /home/kog/local/lib/libopencv_stitching.so.3.1.0
bin/struck: /home/kog/local/lib/libopencv_superres.so.3.1.0
bin/struck: /home/kog/local/lib/libopencv_videostab.so.3.1.0
bin/struck: /home/kog/local/lib/libopencv_cudafeatures2d.so.3.1.0
bin/struck: /home/kog/local/lib/libopencv_cudacodec.so.3.1.0
bin/struck: /home/kog/local/lib/libopencv_cudaoptflow.so.3.1.0
bin/struck: /home/kog/local/lib/libopencv_cudalegacy.so.3.1.0
bin/struck: /home/kog/local/lib/libopencv_calib3d.so.3.1.0
bin/struck: /home/kog/local/lib/libopencv_cudawarping.so.3.1.0
bin/struck: /home/kog/local/lib/libopencv_features2d.so.3.1.0
bin/struck: /home/kog/local/lib/libopencv_flann.so.3.1.0
bin/struck: /home/kog/local/lib/libopencv_objdetect.so.3.1.0
bin/struck: /home/kog/local/lib/libopencv_highgui.so.3.1.0
bin/struck: /home/kog/local/lib/libopencv_ml.so.3.1.0
bin/struck: /home/kog/local/lib/libopencv_photo.so.3.1.0
bin/struck: /home/kog/local/lib/libopencv_cudaimgproc.so.3.1.0
bin/struck: /home/kog/local/lib/libopencv_cudafilters.so.3.1.0
bin/struck: /home/kog/local/lib/libopencv_cudaarithm.so.3.1.0
bin/struck: /home/kog/local/lib/libopencv_video.so.3.1.0
bin/struck: /home/kog/local/lib/libopencv_videoio.so.3.1.0
bin/struck: /home/kog/local/lib/libopencv_imgcodecs.so.3.1.0
bin/struck: /home/kog/local/lib/libopencv_imgproc.so.3.1.0
bin/struck: /home/kog/local/lib/libopencv_core.so.3.1.0
bin/struck: /home/kog/local/lib/libopencv_cudev.so.3.1.0
bin/struck: CMakeFiles/struck.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/kog/lchpro/app/screening_tracking/struck/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_14) "Linking CXX executable bin/struck"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/struck.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/struck.dir/build: bin/struck

.PHONY : CMakeFiles/struck.dir/build

CMakeFiles/struck.dir/requires: CMakeFiles/struck.dir/src/ImageRep.cpp.o.requires
CMakeFiles/struck.dir/requires: CMakeFiles/struck.dir/src/HaarFeature.cpp.o.requires
CMakeFiles/struck.dir/requires: CMakeFiles/struck.dir/src/MultiFeatures.cpp.o.requires
CMakeFiles/struck.dir/requires: CMakeFiles/struck.dir/src/Features.cpp.o.requires
CMakeFiles/struck.dir/requires: CMakeFiles/struck.dir/src/Config.cpp.o.requires
CMakeFiles/struck.dir/requires: CMakeFiles/struck.dir/src/Sampler.cpp.o.requires
CMakeFiles/struck.dir/requires: CMakeFiles/struck.dir/src/HaarFeatures.cpp.o.requires
CMakeFiles/struck.dir/requires: CMakeFiles/struck.dir/src/GraphUtils/GraphUtils.cpp.o.requires
CMakeFiles/struck.dir/requires: CMakeFiles/struck.dir/src/RawFeatures.cpp.o.requires
CMakeFiles/struck.dir/requires: CMakeFiles/struck.dir/src/main.cpp.o.requires
CMakeFiles/struck.dir/requires: CMakeFiles/struck.dir/src/HistogramFeatures.cpp.o.requires
CMakeFiles/struck.dir/requires: CMakeFiles/struck.dir/src/LaRank.cpp.o.requires
CMakeFiles/struck.dir/requires: CMakeFiles/struck.dir/src/Tracker.cpp.o.requires

.PHONY : CMakeFiles/struck.dir/requires

CMakeFiles/struck.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/struck.dir/cmake_clean.cmake
.PHONY : CMakeFiles/struck.dir/clean

CMakeFiles/struck.dir/depend:
	cd /home/kog/lchpro/app/screening_tracking/struck/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/kog/lchpro/app/screening_tracking/struck /home/kog/lchpro/app/screening_tracking/struck /home/kog/lchpro/app/screening_tracking/struck/build /home/kog/lchpro/app/screening_tracking/struck/build /home/kog/lchpro/app/screening_tracking/struck/build/CMakeFiles/struck.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/struck.dir/depend

