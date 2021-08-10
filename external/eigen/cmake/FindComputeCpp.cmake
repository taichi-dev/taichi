#.rst:
# FindComputeCpp
#---------------
#
#   Copyright 2016 Codeplay Software Ltd.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use these files except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

#########################
#  FindComputeCpp.cmake
#########################
#
#  Tools for finding and building with ComputeCpp.
#
#  User must define COMPUTECPP_PACKAGE_ROOT_DIR pointing to the ComputeCpp
#   installation.
#
#  Latest version of this file can be found at:
#    https://github.com/codeplaysoftware/computecpp-sdk

# Require CMake version 3.2.2 or higher
cmake_minimum_required(VERSION 3.2.2)

# Check that a supported host compiler can be found
if(CMAKE_COMPILER_IS_GNUCXX)
    # Require at least gcc 4.8
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.8)
      message(FATAL_ERROR
        "host compiler - Not found! (gcc version must be at least 4.8)")
    # Require the GCC dual ABI to be disabled for 5.1 or higher
    elseif (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 5.1)
      set(COMPUTECPP_DISABLE_GCC_DUAL_ABI "True")
      message(STATUS
        "host compiler - gcc ${CMAKE_CXX_COMPILER_VERSION} (note pre 5.1 gcc ABI enabled)")
    else()
      message(STATUS "host compiler - gcc ${CMAKE_CXX_COMPILER_VERSION}")
    endif()
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    # Require at least clang 3.6
    if (${CMAKE_CXX_COMPILER_VERSION} VERSION_LESS 3.6)
      message(FATAL_ERROR
        "host compiler - Not found! (clang version must be at least 3.6)")
    else()
      message(STATUS "host compiler - clang ${CMAKE_CXX_COMPILER_VERSION}")
    endif()
else()
  message(WARNING
    "host compiler - Not found! (ComputeCpp supports GCC and Clang, see readme)")
endif()

set(COMPUTECPP_64_BIT_DEFAULT ON)
option(COMPUTECPP_64_BIT_CODE "Compile device code in 64 bit mode"
        ${COMPUTECPP_64_BIT_DEFAULT})
mark_as_advanced(COMPUTECPP_64_BIT_CODE)

# Find OpenCL package
find_package(OpenCL REQUIRED)

# Find ComputeCpp packagee
if(NOT COMPUTECPP_PACKAGE_ROOT_DIR)
  message(FATAL_ERROR
    "ComputeCpp package - Not found! (please set COMPUTECPP_PACKAGE_ROOT_DIR")
else()
  message(STATUS "ComputeCpp package - Found")
endif()
option(COMPUTECPP_PACKAGE_ROOT_DIR "Path to the ComputeCpp Package")

# Obtain the path to compute++
find_program(COMPUTECPP_DEVICE_COMPILER compute++ PATHS
  ${COMPUTECPP_PACKAGE_ROOT_DIR} PATH_SUFFIXES bin)
if (EXISTS ${COMPUTECPP_DEVICE_COMPILER})
  mark_as_advanced(COMPUTECPP_DEVICE_COMPILER)
  message(STATUS "compute++ - Found")
else()
  message(FATAL_ERROR "compute++ - Not found! (${COMPUTECPP_DEVICE_COMPILER})")
endif()

# Obtain the path to computecpp_info
find_program(COMPUTECPP_INFO_TOOL computecpp_info PATHS
  ${COMPUTECPP_PACKAGE_ROOT_DIR} PATH_SUFFIXES bin)
if (EXISTS ${COMPUTECPP_INFO_TOOL})
  mark_as_advanced(${COMPUTECPP_INFO_TOOL})
  message(STATUS "computecpp_info - Found")
else()
  message(FATAL_ERROR "computecpp_info - Not found! (${COMPUTECPP_INFO_TOOL})")
endif()

# Obtain the path to the ComputeCpp runtime library
find_library(COMPUTECPP_RUNTIME_LIBRARY ComputeCpp PATHS ${COMPUTECPP_PACKAGE_ROOT_DIR}
  HINTS ${COMPUTECPP_PACKAGE_ROOT_DIR}/lib PATH_SUFFIXES lib
  DOC "ComputeCpp Runtime Library" NO_DEFAULT_PATH)

if (EXISTS ${COMPUTECPP_RUNTIME_LIBRARY})
  mark_as_advanced(COMPUTECPP_RUNTIME_LIBRARY)
  message(STATUS "libComputeCpp.so - Found")
else()
  message(FATAL_ERROR "libComputeCpp.so - Not found!")
endif()

# Obtain the ComputeCpp include directory
set(COMPUTECPP_INCLUDE_DIRECTORY ${COMPUTECPP_PACKAGE_ROOT_DIR}/include/)
if (NOT EXISTS ${COMPUTECPP_INCLUDE_DIRECTORY})
  message(FATAL_ERROR "ComputeCpp includes - Not found!")
else()
  message(STATUS "ComputeCpp includes - Found")
endif()

# Obtain the package version
execute_process(COMMAND ${COMPUTECPP_INFO_TOOL} "--dump-version"
  OUTPUT_VARIABLE COMPUTECPP_PACKAGE_VERSION
  RESULT_VARIABLE COMPUTECPP_INFO_TOOL_RESULT OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT COMPUTECPP_INFO_TOOL_RESULT EQUAL "0")
  message(FATAL_ERROR "Package version - Error obtaining version!")
else()
  mark_as_advanced(COMPUTECPP_PACKAGE_VERSION)
  message(STATUS "Package version - ${COMPUTECPP_PACKAGE_VERSION}")
endif()

# Obtain the device compiler flags
execute_process(COMMAND ${COMPUTECPP_INFO_TOOL} "--dump-device-compiler-flags"
  OUTPUT_VARIABLE COMPUTECPP_DEVICE_COMPILER_FLAGS
  RESULT_VARIABLE COMPUTECPP_INFO_TOOL_RESULT OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT COMPUTECPP_INFO_TOOL_RESULT EQUAL "0")
  message(FATAL_ERROR "compute++ flags - Error obtaining compute++ flags!")
else()
  mark_as_advanced(COMPUTECPP_COMPILER_FLAGS)
  message(STATUS "compute++ flags - ${COMPUTECPP_DEVICE_COMPILER_FLAGS}")
endif()

set(COMPUTECPP_DEVICE_COMPILER_FLAGS ${COMPUTECPP_DEVICE_COMPILER_FLAGS} -sycl-compress-name -no-serial-memop -DEIGEN_NO_ASSERTION_CHECKING=1)

# Check if the platform is supported
execute_process(COMMAND ${COMPUTECPP_INFO_TOOL} "--dump-is-supported"
  OUTPUT_VARIABLE COMPUTECPP_PLATFORM_IS_SUPPORTED
  RESULT_VARIABLE COMPUTECPP_INFO_TOOL_RESULT OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT COMPUTECPP_INFO_TOOL_RESULT EQUAL "0")
  message(FATAL_ERROR "platform - Error checking platform support!")
else()
  mark_as_advanced(COMPUTECPP_PLATFORM_IS_SUPPORTED)
  if (COMPUTECPP_PLATFORM_IS_SUPPORTED)
    message(STATUS "platform - your system can support ComputeCpp")
  else()
    message(STATUS "platform - your system CANNOT support ComputeCpp")
  endif()
endif()

####################
#   __build_sycl
####################
#
#  Adds a custom target for running compute++ and adding a dependency for the
#  resulting integration header.
#
#  targetName : Name of the target.
#  sourceFile : Source file to be compiled.
#  binaryDir : Intermediate directory to output the integration header.
#
function(__build_spir targetName sourceFile binaryDir)

  # Retrieve source file name.
  get_filename_component(sourceFileName ${sourceFile} NAME)

  # Set the path to the Sycl file.
  set(outputSyclFile ${binaryDir}/${sourceFileName}.sycl)

  # Add any user-defined include to the device compiler
  get_property(includeDirectories DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY
    INCLUDE_DIRECTORIES)
  set(device_compiler_includes "")
  foreach(directory ${includeDirectories})
    set(device_compiler_includes "-I${directory}" ${device_compiler_includes})
  endforeach()
  if (CMAKE_INCLUDE_PATH)
    foreach(directory ${CMAKE_INCLUDE_PATH})
      set(device_compiler_includes "-I${directory}"
        ${device_compiler_includes})
    endforeach()
  endif()

  # Convert argument list format
  separate_arguments(COMPUTECPP_DEVICE_COMPILER_FLAGS)

  # Add custom command for running compute++
  add_custom_command(
    OUTPUT ${outputSyclFile}
    COMMAND ${COMPUTECPP_DEVICE_COMPILER}
            ${COMPUTECPP_DEVICE_COMPILER_FLAGS}
            -isystem ${COMPUTECPP_INCLUDE_DIRECTORY}
            ${COMPUTECPP_PLATFORM_SPECIFIC_ARGS}
            ${device_compiler_includes}
            -o ${outputSyclFile}
            -c ${CMAKE_CURRENT_SOURCE_DIR}/${sourceFile}
    DEPENDS ${sourceFile}
    WORKING_DIRECTORY ${binaryDir}
  COMMENT "Building ComputeCpp integration header file ${outputSyclFile}")

  # Add a custom target for the generated integration header
  add_custom_target(${targetName}_integration_header DEPENDS ${outputSyclFile})

  # Add a dependency on the integration header
  add_dependencies(${targetName} ${targetName}_integration_header)

  # Set the host compiler C++ standard to C++11
  set_property(TARGET ${targetName} PROPERTY CXX_STANDARD 11)

  # Disable GCC dual ABI on GCC 5.1 and higher
  if(COMPUTECPP_DISABLE_GCC_DUAL_ABI)
    set_property(TARGET ${targetName} APPEND PROPERTY COMPILE_DEFINITIONS
      "_GLIBCXX_USE_CXX11_ABI=0")
  endif()

endfunction()

#######################
#  add_sycl_to_target
#######################
#
#  Adds a SYCL compilation custom command associated with an existing
#  target and sets a dependancy on that new command.
#
#  targetName : Name of the target to add a SYCL to.
#  sourceFile : Source file to be compiled for SYCL.
#  binaryDir : Intermediate directory to output the integration header.
#
function(add_sycl_to_target targetName sourceFile binaryDir)

  # Add custom target to run compute++ and generate the integration header
  __build_spir(${targetName} ${sourceFile} ${binaryDir})

  # Link with the ComputeCpp runtime library
  target_link_libraries(${targetName} PUBLIC ${COMPUTECPP_RUNTIME_LIBRARY}
                        PUBLIC ${OpenCL_LIBRARIES})

endfunction(add_sycl_to_target)
