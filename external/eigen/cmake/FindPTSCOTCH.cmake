###
#
# @copyright (c) 2009-2014 The University of Tennessee and The University
#                          of Tennessee Research Foundation.
#                          All rights reserved.
# @copyright (c) 2012-2016 Inria. All rights reserved.
# @copyright (c) 2012-2014 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria, Univ. Bordeaux. All rights reserved.
#
###
#
# - Find PTSCOTCH include dirs and libraries
# Use this module by invoking find_package with the form:
#  find_package(PTSCOTCH
#               [REQUIRED]             # Fail with error if ptscotch is not found
#               [COMPONENTS <comp1> <comp2> ...] # dependencies
#              )
#
#  PTSCOTCH depends on the following libraries:
#   - Threads
#   - MPI
#
#  COMPONENTS can be some of the following:
#   - ESMUMPS: to activate detection of PT-Scotch with the esmumps interface
#
# This module finds headers and ptscotch library.
# Results are reported in variables:
#  PTSCOTCH_FOUND            - True if headers and requested libraries were found
#  PTSCOTCH_LINKER_FLAGS     - list of required linker flags (excluding -l and -L)
#  PTSCOTCH_INCLUDE_DIRS     - ptscotch include directories
#  PTSCOTCH_LIBRARY_DIRS     - Link directories for ptscotch libraries
#  PTSCOTCH_LIBRARIES        - ptscotch component libraries to be linked
#  PTSCOTCH_INCLUDE_DIRS_DEP - ptscotch + dependencies include directories
#  PTSCOTCH_LIBRARY_DIRS_DEP - ptscotch + dependencies link directories
#  PTSCOTCH_LIBRARIES_DEP    - ptscotch libraries + dependencies
#  PTSCOTCH_INTSIZE          - Number of octets occupied by a SCOTCH_Num
#
# The user can give specific paths where to find the libraries adding cmake
# options at configure (ex: cmake path/to/project -DPTSCOTCH=path/to/ptscotch):
#  PTSCOTCH_DIR              - Where to find the base directory of ptscotch
#  PTSCOTCH_INCDIR           - Where to find the header files
#  PTSCOTCH_LIBDIR           - Where to find the library files
# The module can also look for the following environment variables if paths
# are not given as cmake variable: PTSCOTCH_DIR, PTSCOTCH_INCDIR, PTSCOTCH_LIBDIR

#=============================================================================
# Copyright 2012-2013 Inria
# Copyright 2012-2013 Emmanuel Agullo
# Copyright 2012-2013 Mathieu Faverge
# Copyright 2012      Cedric Castagnede
# Copyright 2013-2016 Florent Pruvost
#
# Distributed under the OSI-approved BSD License (the "License");
# see accompanying file MORSE-Copyright.txt for details.
#
# This software is distributed WITHOUT ANY WARRANTY; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the License for more information.
#=============================================================================
# (To distribute this file outside of Morse, substitute the full
#  License text for the above reference.)

if (NOT PTSCOTCH_FOUND)
  set(PTSCOTCH_DIR "" CACHE PATH "Installation directory of PTSCOTCH library")
  if (NOT PTSCOTCH_FIND_QUIETLY)
    message(STATUS "A cache variable, namely PTSCOTCH_DIR, has been set to specify the install directory of PTSCOTCH")
  endif()
endif()

# Set the version to find
set(PTSCOTCH_LOOK_FOR_ESMUMPS OFF)

if( PTSCOTCH_FIND_COMPONENTS )
  foreach( component ${PTSCOTCH_FIND_COMPONENTS} )
    if (${component} STREQUAL "ESMUMPS")
      # means we look for esmumps library
      set(PTSCOTCH_LOOK_FOR_ESMUMPS ON)
    endif()
  endforeach()
endif()

# PTSCOTCH depends on Threads, try to find it
if (NOT THREADS_FOUND)
  if (PTSCOTCH_FIND_REQUIRED)
    find_package(Threads REQUIRED)
  else()
    find_package(Threads)
  endif()
endif()

# PTSCOTCH depends on MPI, try to find it
if (NOT MPI_FOUND)
  if (PTSCOTCH_FIND_REQUIRED)
    find_package(MPI REQUIRED)
  else()
    find_package(MPI)
  endif()
endif()

# Looking for include
# -------------------

# Add system include paths to search include
# ------------------------------------------
unset(_inc_env)
set(ENV_PTSCOTCH_DIR "$ENV{PTSCOTCH_DIR}")
set(ENV_PTSCOTCH_INCDIR "$ENV{PTSCOTCH_INCDIR}")
if(ENV_PTSCOTCH_INCDIR)
  list(APPEND _inc_env "${ENV_PTSCOTCH_INCDIR}")
elseif(ENV_PTSCOTCH_DIR)
  list(APPEND _inc_env "${ENV_PTSCOTCH_DIR}")
  list(APPEND _inc_env "${ENV_PTSCOTCH_DIR}/include")
  list(APPEND _inc_env "${ENV_PTSCOTCH_DIR}/include/ptscotch")
else()
  if(WIN32)
    string(REPLACE ":" ";" _inc_env "$ENV{INCLUDE}")
  else()
    string(REPLACE ":" ";" _path_env "$ENV{INCLUDE}")
    list(APPEND _inc_env "${_path_env}")
    string(REPLACE ":" ";" _path_env "$ENV{C_INCLUDE_PATH}")
    list(APPEND _inc_env "${_path_env}")
    string(REPLACE ":" ";" _path_env "$ENV{CPATH}")
    list(APPEND _inc_env "${_path_env}")
    string(REPLACE ":" ";" _path_env "$ENV{INCLUDE_PATH}")
    list(APPEND _inc_env "${_path_env}")
  endif()
endif()
list(APPEND _inc_env "${CMAKE_PLATFORM_IMPLICIT_INCLUDE_DIRECTORIES}")
list(APPEND _inc_env "${CMAKE_C_IMPLICIT_INCLUDE_DIRECTORIES}")
list(REMOVE_DUPLICATES _inc_env)


# Try to find the ptscotch header in the given paths
# -------------------------------------------------

set(PTSCOTCH_hdrs_to_find "ptscotch.h;scotch.h")

# call cmake macro to find the header path
if(PTSCOTCH_INCDIR)
  foreach(ptscotch_hdr ${PTSCOTCH_hdrs_to_find})
    set(PTSCOTCH_${ptscotch_hdr}_DIRS "PTSCOTCH_${ptscotch_hdr}_DIRS-NOTFOUND")
    find_path(PTSCOTCH_${ptscotch_hdr}_DIRS
      NAMES ${ptscotch_hdr}
      HINTS ${PTSCOTCH_INCDIR})
    mark_as_advanced(PTSCOTCH_${ptscotch_hdr}_DIRS)
  endforeach()
else()
  if(PTSCOTCH_DIR)
    foreach(ptscotch_hdr ${PTSCOTCH_hdrs_to_find})
      set(PTSCOTCH_${ptscotch_hdr}_DIRS "PTSCOTCH_${ptscotch_hdr}_DIRS-NOTFOUND")
      find_path(PTSCOTCH_${ptscotch_hdr}_DIRS
	NAMES ${ptscotch_hdr}
	HINTS ${PTSCOTCH_DIR}
	PATH_SUFFIXES "include" "include/scotch")
      mark_as_advanced(PTSCOTCH_${ptscotch_hdr}_DIRS)
    endforeach()
  else()
    foreach(ptscotch_hdr ${PTSCOTCH_hdrs_to_find})
      set(PTSCOTCH_${ptscotch_hdr}_DIRS "PTSCOTCH_${ptscotch_hdr}_DIRS-NOTFOUND")
      find_path(PTSCOTCH_${ptscotch_hdr}_DIRS
	NAMES ${ptscotch_hdr}
	HINTS ${_inc_env}
	PATH_SUFFIXES "scotch")
      mark_as_advanced(PTSCOTCH_${ptscotch_hdr}_DIRS)
    endforeach()
  endif()
endif()

# If found, add path to cmake variable
# ------------------------------------
foreach(ptscotch_hdr ${PTSCOTCH_hdrs_to_find})
  if (PTSCOTCH_${ptscotch_hdr}_DIRS)
    list(APPEND PTSCOTCH_INCLUDE_DIRS "${PTSCOTCH_${ptscotch_hdr}_DIRS}")
  else ()
    set(PTSCOTCH_INCLUDE_DIRS "PTSCOTCH_INCLUDE_DIRS-NOTFOUND")
    if (NOT PTSCOTCH_FIND_QUIETLY)
      message(STATUS "Looking for ptscotch -- ${ptscotch_hdr} not found")
    endif()
  endif()
endforeach()
list(REMOVE_DUPLICATES PTSCOTCH_INCLUDE_DIRS)

# Looking for lib
# ---------------

# Add system library paths to search lib
# --------------------------------------
unset(_lib_env)
set(ENV_PTSCOTCH_LIBDIR "$ENV{PTSCOTCH_LIBDIR}")
if(ENV_PTSCOTCH_LIBDIR)
  list(APPEND _lib_env "${ENV_PTSCOTCH_LIBDIR}")
elseif(ENV_PTSCOTCH_DIR)
  list(APPEND _lib_env "${ENV_PTSCOTCH_DIR}")
  list(APPEND _lib_env "${ENV_PTSCOTCH_DIR}/lib")
else()
  if(WIN32)
    string(REPLACE ":" ";" _lib_env "$ENV{LIB}")
  else()
    if(APPLE)
      string(REPLACE ":" ";" _lib_env "$ENV{DYLD_LIBRARY_PATH}")
    else()
      string(REPLACE ":" ";" _lib_env "$ENV{LD_LIBRARY_PATH}")
    endif()
    list(APPEND _lib_env "${CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES}")
    list(APPEND _lib_env "${CMAKE_C_IMPLICIT_LINK_DIRECTORIES}")
  endif()
endif()
list(REMOVE_DUPLICATES _lib_env)

# Try to find the ptscotch lib in the given paths
# ----------------------------------------------

set(PTSCOTCH_libs_to_find "ptscotch;ptscotcherr")
if (PTSCOTCH_LOOK_FOR_ESMUMPS)
  list(INSERT PTSCOTCH_libs_to_find 0 "ptesmumps")
  list(APPEND PTSCOTCH_libs_to_find   "esmumps"  )
endif()
list(APPEND PTSCOTCH_libs_to_find "scotch;scotcherr")

# call cmake macro to find the lib path
if(PTSCOTCH_LIBDIR)
  foreach(ptscotch_lib ${PTSCOTCH_libs_to_find})
    set(PTSCOTCH_${ptscotch_lib}_LIBRARY "PTSCOTCH_${ptscotch_lib}_LIBRARY-NOTFOUND")
    find_library(PTSCOTCH_${ptscotch_lib}_LIBRARY
      NAMES ${ptscotch_lib}
      HINTS ${PTSCOTCH_LIBDIR})
  endforeach()
else()
  if(PTSCOTCH_DIR)
    foreach(ptscotch_lib ${PTSCOTCH_libs_to_find})
      set(PTSCOTCH_${ptscotch_lib}_LIBRARY "PTSCOTCH_${ptscotch_lib}_LIBRARY-NOTFOUND")
      find_library(PTSCOTCH_${ptscotch_lib}_LIBRARY
	NAMES ${ptscotch_lib}
	HINTS ${PTSCOTCH_DIR}
	PATH_SUFFIXES lib lib32 lib64)
    endforeach()
  else()
    foreach(ptscotch_lib ${PTSCOTCH_libs_to_find})
      set(PTSCOTCH_${ptscotch_lib}_LIBRARY "PTSCOTCH_${ptscotch_lib}_LIBRARY-NOTFOUND")
      find_library(PTSCOTCH_${ptscotch_lib}_LIBRARY
	NAMES ${ptscotch_lib}
	HINTS ${_lib_env})
    endforeach()
  endif()
endif()

set(PTSCOTCH_LIBRARIES "")
set(PTSCOTCH_LIBRARY_DIRS "")
# If found, add path to cmake variable
# ------------------------------------
foreach(ptscotch_lib ${PTSCOTCH_libs_to_find})

  if (PTSCOTCH_${ptscotch_lib}_LIBRARY)
    get_filename_component(${ptscotch_lib}_lib_path "${PTSCOTCH_${ptscotch_lib}_LIBRARY}" PATH)
    # set cmake variables
    list(APPEND PTSCOTCH_LIBRARIES "${PTSCOTCH_${ptscotch_lib}_LIBRARY}")
    list(APPEND PTSCOTCH_LIBRARY_DIRS "${${ptscotch_lib}_lib_path}")
  else ()
    list(APPEND PTSCOTCH_LIBRARIES "${PTSCOTCH_${ptscotch_lib}_LIBRARY}")
    if (NOT PTSCOTCH_FIND_QUIETLY)
      message(STATUS "Looking for ptscotch -- lib ${ptscotch_lib} not found")
    endif()
  endif ()

  mark_as_advanced(PTSCOTCH_${ptscotch_lib}_LIBRARY)

endforeach()
list(REMOVE_DUPLICATES PTSCOTCH_LIBRARY_DIRS)

# check a function to validate the find
if(PTSCOTCH_LIBRARIES)

  set(REQUIRED_LDFLAGS)
  set(REQUIRED_INCDIRS)
  set(REQUIRED_LIBDIRS)
  set(REQUIRED_LIBS)

  # PTSCOTCH
  if (PTSCOTCH_INCLUDE_DIRS)
    set(REQUIRED_INCDIRS  "${PTSCOTCH_INCLUDE_DIRS}")
  endif()
  if (PTSCOTCH_LIBRARY_DIRS)
    set(REQUIRED_LIBDIRS "${PTSCOTCH_LIBRARY_DIRS}")
  endif()
  set(REQUIRED_LIBS "${PTSCOTCH_LIBRARIES}")
  # MPI
  if (MPI_FOUND)
    if (MPI_C_INCLUDE_PATH)
      list(APPEND CMAKE_REQUIRED_INCLUDES "${MPI_C_INCLUDE_PATH}")
    endif()
    if (MPI_C_LINK_FLAGS)
      if (${MPI_C_LINK_FLAGS} MATCHES "  -")
	string(REGEX REPLACE " -" "-" MPI_C_LINK_FLAGS ${MPI_C_LINK_FLAGS})
      endif()
      list(APPEND REQUIRED_LDFLAGS "${MPI_C_LINK_FLAGS}")
    endif()
    list(APPEND REQUIRED_LIBS "${MPI_C_LIBRARIES}")
  endif()
  # THREADS
  if(CMAKE_THREAD_LIBS_INIT)
    list(APPEND REQUIRED_LIBS "${CMAKE_THREAD_LIBS_INIT}")
  endif()
  set(Z_LIBRARY "Z_LIBRARY-NOTFOUND")
  find_library(Z_LIBRARY NAMES z)
  mark_as_advanced(Z_LIBRARY)
  if(Z_LIBRARY)
    list(APPEND REQUIRED_LIBS "-lz")
  endif()
  set(M_LIBRARY "M_LIBRARY-NOTFOUND")
  find_library(M_LIBRARY NAMES m)
  mark_as_advanced(M_LIBRARY)
  if(M_LIBRARY)
    list(APPEND REQUIRED_LIBS "-lm")
  endif()
  set(RT_LIBRARY "RT_LIBRARY-NOTFOUND")
  find_library(RT_LIBRARY NAMES rt)
  mark_as_advanced(RT_LIBRARY)
  if(RT_LIBRARY)
    list(APPEND REQUIRED_LIBS "-lrt")
  endif()

  # set required libraries for link
  set(CMAKE_REQUIRED_INCLUDES "${REQUIRED_INCDIRS}")
  set(CMAKE_REQUIRED_LIBRARIES)
  list(APPEND CMAKE_REQUIRED_LIBRARIES "${REQUIRED_LDFLAGS}")
  foreach(lib_dir ${REQUIRED_LIBDIRS})
    list(APPEND CMAKE_REQUIRED_LIBRARIES "-L${lib_dir}")
  endforeach()
  list(APPEND CMAKE_REQUIRED_LIBRARIES "${REQUIRED_LIBS}")
  list(APPEND CMAKE_REQUIRED_FLAGS "${REQUIRED_FLAGS}")
  string(REGEX REPLACE "^ -" "-" CMAKE_REQUIRED_LIBRARIES "${CMAKE_REQUIRED_LIBRARIES}")

  # test link
  unset(PTSCOTCH_WORKS CACHE)
  include(CheckFunctionExists)
  check_function_exists(SCOTCH_dgraphInit PTSCOTCH_WORKS)
  mark_as_advanced(PTSCOTCH_WORKS)

  if(PTSCOTCH_WORKS)
    # save link with dependencies
    set(PTSCOTCH_LIBRARIES_DEP "${REQUIRED_LIBS}")
    set(PTSCOTCH_LIBRARY_DIRS_DEP "${REQUIRED_LIBDIRS}")
    set(PTSCOTCH_INCLUDE_DIRS_DEP "${REQUIRED_INCDIRS}")
    set(PTSCOTCH_LINKER_FLAGS "${REQUIRED_LDFLAGS}")
    list(REMOVE_DUPLICATES PTSCOTCH_LIBRARY_DIRS_DEP)
    list(REMOVE_DUPLICATES PTSCOTCH_INCLUDE_DIRS_DEP)
    list(REMOVE_DUPLICATES PTSCOTCH_LINKER_FLAGS)
  else()
    if(NOT PTSCOTCH_FIND_QUIETLY)
      message(STATUS "Looking for PTSCOTCH : test of SCOTCH_dgraphInit with PTSCOTCH library fails")
      message(STATUS "CMAKE_REQUIRED_LIBRARIES: ${CMAKE_REQUIRED_LIBRARIES}")
      message(STATUS "CMAKE_REQUIRED_INCLUDES: ${CMAKE_REQUIRED_INCLUDES}")
      message(STATUS "Check in CMakeFiles/CMakeError.log to figure out why it fails")
    endif()
  endif()
  set(CMAKE_REQUIRED_INCLUDES)
  set(CMAKE_REQUIRED_FLAGS)
  set(CMAKE_REQUIRED_LIBRARIES)
endif(PTSCOTCH_LIBRARIES)

if (PTSCOTCH_LIBRARIES)
  list(GET PTSCOTCH_LIBRARIES 0 first_lib)
  get_filename_component(first_lib_path "${first_lib}" PATH)
  if (${first_lib_path} MATCHES "/lib(32|64)?$")
    string(REGEX REPLACE "/lib(32|64)?$" "" not_cached_dir "${first_lib_path}")
    set(PTSCOTCH_DIR_FOUND "${not_cached_dir}" CACHE PATH "Installation directory of PTSCOTCH library" FORCE)
  else()
    set(PTSCOTCH_DIR_FOUND "${first_lib_path}" CACHE PATH "Installation directory of PTSCOTCH library" FORCE)
  endif()
endif()
mark_as_advanced(PTSCOTCH_DIR)
mark_as_advanced(PTSCOTCH_DIR_FOUND)

# Check the size of SCOTCH_Num
# ---------------------------------
set(CMAKE_REQUIRED_INCLUDES ${PTSCOTCH_INCLUDE_DIRS})

include(CheckCSourceRuns)
#stdio.h and stdint.h should be included by scotch.h directly
set(PTSCOTCH_C_TEST_SCOTCH_Num_4 "
#include <stdio.h>
#include <stdint.h>
#include <ptscotch.h>
int main(int argc, char **argv) {
  if (sizeof(SCOTCH_Num) == 4)
    return 0;
  else
    return 1;
}
")

set(PTSCOTCH_C_TEST_SCOTCH_Num_8 "
#include <stdio.h>
#include <stdint.h>
#include <ptscotch.h>
int main(int argc, char **argv) {
  if (sizeof(SCOTCH_Num) == 8)
    return 0;
  else
    return 1;
}
")
check_c_source_runs("${PTSCOTCH_C_TEST_SCOTCH_Num_4}" PTSCOTCH_Num_4)
if(NOT PTSCOTCH_Num_4)
  check_c_source_runs("${PTSCOTCH_C_TEST_SCOTCH_Num_8}" PTSCOTCH_Num_8)
  if(NOT PTSCOTCH_Num_8)
    set(PTSCOTCH_INTSIZE -1)
  else()
    set(PTSCOTCH_INTSIZE 8)
  endif()
else()
  set(PTSCOTCH_INTSIZE 4)
endif()
set(CMAKE_REQUIRED_INCLUDES "")

# check that PTSCOTCH has been found
# ---------------------------------
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(PTSCOTCH DEFAULT_MSG
  PTSCOTCH_LIBRARIES
  PTSCOTCH_WORKS)
#
# TODO: Add possibility to check for specific functions in the library
#
