###
#
# @copyright (c) 2009-2014 The University of Tennessee and The University
#                          of Tennessee Research Foundation.
#                          All rights reserved.
# @copyright (c) 2012-2014 Inria. All rights reserved.
# @copyright (c) 2012-2014 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria, Univ. Bordeaux. All rights reserved.
#
###
#
# - Find PASTIX include dirs and libraries
# Use this module by invoking find_package with the form:
#  find_package(PASTIX
#               [REQUIRED] # Fail with error if pastix is not found
#               [COMPONENTS <comp1> <comp2> ...] # dependencies
#              )
#
#  PASTIX depends on the following libraries:
#   - Threads, m, rt
#   - MPI
#   - HWLOC
#   - BLAS
#
#  COMPONENTS are optional libraries PASTIX could be linked with,
#  Use it to drive detection of a specific compilation chain
#  COMPONENTS can be some of the following:
#   - MPI: to activate detection of the parallel MPI version (default)
#        it looks for Threads, HWLOC, BLAS, MPI and ScaLAPACK libraries
#   - SEQ: to activate detection of the sequential version (exclude MPI version)
#   - STARPU: to activate detection of StarPU version
#   it looks for MPI version of StarPU (default behaviour)
#   if SEQ and STARPU are given, it looks for a StarPU without MPI
#   - STARPU_CUDA: to activate detection of StarPU with CUDA
#   - STARPU_FXT: to activate detection of StarPU with FxT
#   - SCOTCH: to activate detection of PASTIX linked with SCOTCH
#   - PTSCOTCH: to activate detection of PASTIX linked with SCOTCH
#   - METIS: to activate detection of PASTIX linked with SCOTCH
#
# This module finds headers and pastix library.
# Results are reported in variables:
#  PASTIX_FOUND            - True if headers and requested libraries were found
#  PASTIX_LINKER_FLAGS     - list of required linker flags (excluding -l and -L)
#  PASTIX_INCLUDE_DIRS     - pastix include directories
#  PASTIX_LIBRARY_DIRS     - Link directories for pastix libraries
#  PASTIX_LIBRARIES        - pastix libraries
#  PASTIX_INCLUDE_DIRS_DEP - pastix + dependencies include directories
#  PASTIX_LIBRARY_DIRS_DEP - pastix + dependencies link directories
#  PASTIX_LIBRARIES_DEP    - pastix libraries + dependencies
#
# The user can give specific paths where to find the libraries adding cmake
# options at configure (ex: cmake path/to/project -DPASTIX_DIR=path/to/pastix):
#  PASTIX_DIR              - Where to find the base directory of pastix
#  PASTIX_INCDIR           - Where to find the header files
#  PASTIX_LIBDIR           - Where to find the library files
# The module can also look for the following environment variables if paths
# are not given as cmake variable: PASTIX_DIR, PASTIX_INCDIR, PASTIX_LIBDIR

#=============================================================================
# Copyright 2012-2013 Inria
# Copyright 2012-2013 Emmanuel Agullo
# Copyright 2012-2013 Mathieu Faverge
# Copyright 2012      Cedric Castagnede
# Copyright 2013      Florent Pruvost
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


if (NOT PASTIX_FOUND)
  set(PASTIX_DIR "" CACHE PATH "Installation directory of PASTIX library")
  if (NOT PASTIX_FIND_QUIETLY)
    message(STATUS "A cache variable, namely PASTIX_DIR, has been set to specify the install directory of PASTIX")
  endif()
endif()

# Set the version to find
set(PASTIX_LOOK_FOR_MPI ON)
set(PASTIX_LOOK_FOR_SEQ OFF)
set(PASTIX_LOOK_FOR_STARPU OFF)
set(PASTIX_LOOK_FOR_STARPU_CUDA OFF)
set(PASTIX_LOOK_FOR_STARPU_FXT OFF)
set(PASTIX_LOOK_FOR_SCOTCH ON)
set(PASTIX_LOOK_FOR_PTSCOTCH OFF)
set(PASTIX_LOOK_FOR_METIS OFF)

if( PASTIX_FIND_COMPONENTS )
  foreach( component ${PASTIX_FIND_COMPONENTS} )
    if (${component} STREQUAL "SEQ")
      # means we look for the sequential version of PaStiX (without MPI)
      set(PASTIX_LOOK_FOR_SEQ ON)
      set(PASTIX_LOOK_FOR_MPI OFF)
    endif()
    if (${component} STREQUAL "MPI")
      # means we look for the MPI version of PaStiX (default)
      set(PASTIX_LOOK_FOR_SEQ OFF)
      set(PASTIX_LOOK_FOR_MPI ON)
    endif()
    if (${component} STREQUAL "STARPU")
      # means we look for PaStiX with StarPU
      set(PASTIX_LOOK_FOR_STARPU ON)
    endif()
    if (${component} STREQUAL "STARPU_CUDA")
      # means we look for PaStiX with StarPU + CUDA
      set(PASTIX_LOOK_FOR_STARPU ON)
      set(PASTIX_LOOK_FOR_STARPU_CUDA ON)
    endif()
    if (${component} STREQUAL "STARPU_FXT")
      # means we look for PaStiX with StarPU + FxT
      set(PASTIX_LOOK_FOR_STARPU_FXT ON)
    endif()
    if (${component} STREQUAL "SCOTCH")
      set(PASTIX_LOOK_FOR_SCOTCH ON)
    endif()
    if (${component} STREQUAL "SCOTCH")
      set(PASTIX_LOOK_FOR_PTSCOTCH ON)
    endif()
    if (${component} STREQUAL "METIS")
      set(PASTIX_LOOK_FOR_METIS ON)
    endif()
  endforeach()
endif()

# Dependencies detection
# ----------------------


# Required dependencies
# ---------------------

if (NOT PASTIX_FIND_QUIETLY)
  message(STATUS "Looking for PASTIX - Try to detect pthread")
endif()
if (PASTIX_FIND_REQUIRED)
  find_package(Threads REQUIRED QUIET)
else()
  find_package(Threads QUIET)
endif()
set(PASTIX_EXTRA_LIBRARIES "")
if( THREADS_FOUND )
  list(APPEND PASTIX_EXTRA_LIBRARIES ${CMAKE_THREAD_LIBS_INIT})
endif ()

# Add math library to the list of extra
# it normally exists on all common systems provided with a C compiler
if (NOT PASTIX_FIND_QUIETLY)
  message(STATUS "Looking for PASTIX - Try to detect libm")
endif()
set(PASTIX_M_LIBRARIES "")
if(UNIX OR WIN32)
  find_library(
    PASTIX_M_m_LIBRARY
    NAMES m
    )
  mark_as_advanced(PASTIX_M_m_LIBRARY)
  if (PASTIX_M_m_LIBRARY)
    list(APPEND PASTIX_M_LIBRARIES "${PASTIX_M_m_LIBRARY}")
    list(APPEND PASTIX_EXTRA_LIBRARIES "${PASTIX_M_m_LIBRARY}")
  else()
    if (PASTIX_FIND_REQUIRED)
      message(FATAL_ERROR "Could NOT find libm on your system."
	"Are you sure to a have a C compiler installed?")
    endif()
  endif()
endif()

# Try to find librt (libposix4 - POSIX.1b Realtime Extensions library)
# on Unix systems except Apple ones because it does not exist on it
if (NOT PASTIX_FIND_QUIETLY)
  message(STATUS "Looking for PASTIX - Try to detect librt")
endif()
set(PASTIX_RT_LIBRARIES "")
if(UNIX AND NOT APPLE)
  find_library(
    PASTIX_RT_rt_LIBRARY
    NAMES rt
    )
  mark_as_advanced(PASTIX_RT_rt_LIBRARY)
  if (PASTIX_RT_rt_LIBRARY)
    list(APPEND PASTIX_RT_LIBRARIES "${PASTIX_RT_rt_LIBRARY}")
    list(APPEND PASTIX_EXTRA_LIBRARIES "${PASTIX_RT_rt_LIBRARY}")
  else()
    if (PASTIX_FIND_REQUIRED)
      message(FATAL_ERROR "Could NOT find librt on your system")
    endif()
  endif()
endif()

# PASTIX depends on HWLOC
#------------------------
if (NOT PASTIX_FIND_QUIETLY)
  message(STATUS "Looking for PASTIX - Try to detect HWLOC")
endif()
if (PASTIX_FIND_REQUIRED)
  find_package(HWLOC REQUIRED QUIET)
else()
  find_package(HWLOC QUIET)
endif()

# PASTIX depends on BLAS
#-----------------------
if (NOT PASTIX_FIND_QUIETLY)
  message(STATUS "Looking for PASTIX - Try to detect BLAS")
endif()
if (PASTIX_FIND_REQUIRED)
  find_package(BLASEXT REQUIRED QUIET)
else()
  find_package(BLASEXT QUIET)
endif()

# Optional dependencies
# ---------------------

# PASTIX may depend on MPI
#-------------------------
if (NOT MPI_FOUND AND PASTIX_LOOK_FOR_MPI)
  if (NOT PASTIX_FIND_QUIETLY)
    message(STATUS "Looking for PASTIX - Try to detect MPI")
  endif()
  # allows to use an external mpi compilation by setting compilers with
  # -DMPI_C_COMPILER=path/to/mpicc -DMPI_Fortran_COMPILER=path/to/mpif90
  # at cmake configure
  if(NOT MPI_C_COMPILER)
    set(MPI_C_COMPILER mpicc)
  endif()
  if (PASTIX_FIND_REQUIRED AND PASTIX_FIND_REQUIRED_MPI)
    find_package(MPI REQUIRED QUIET)
  else()
    find_package(MPI QUIET)
  endif()
  if (MPI_FOUND)
    mark_as_advanced(MPI_LIBRARY)
    mark_as_advanced(MPI_EXTRA_LIBRARY)
  endif()
endif (NOT MPI_FOUND AND PASTIX_LOOK_FOR_MPI)

# PASTIX may depend on STARPU
#----------------------------
if( NOT STARPU_FOUND AND PASTIX_LOOK_FOR_STARPU)

  if (NOT PASTIX_FIND_QUIETLY)
    message(STATUS "Looking for PASTIX - Try to detect StarPU")
  endif()

  set(PASTIX_STARPU_VERSION "1.1" CACHE STRING "oldest STARPU version desired")

  # create list of components in order to make a single call to find_package(starpu...)
  # we explicitly need a StarPU version built with hwloc
  set(STARPU_COMPONENT_LIST "HWLOC")

  # StarPU may depend on MPI
  # allows to use an external mpi compilation by setting compilers with
  # -DMPI_C_COMPILER=path/to/mpicc -DMPI_Fortran_COMPILER=path/to/mpif90
  # at cmake configure
  if (PASTIX_LOOK_FOR_MPI)
    if(NOT MPI_C_COMPILER)
      set(MPI_C_COMPILER mpicc)
    endif()
    list(APPEND STARPU_COMPONENT_LIST "MPI")
  endif()
  if (PASTIX_LOOK_FOR_STARPU_CUDA)
    list(APPEND STARPU_COMPONENT_LIST "CUDA")
  endif()
  if (PASTIX_LOOK_FOR_STARPU_FXT)
    list(APPEND STARPU_COMPONENT_LIST "FXT")
  endif()
  # set the list of optional dependencies we may discover
  if (PASTIX_FIND_REQUIRED AND PASTIX_FIND_REQUIRED_STARPU)
    find_package(STARPU ${PASTIX_STARPU_VERSION} REQUIRED
      COMPONENTS ${STARPU_COMPONENT_LIST})
  else()
    find_package(STARPU ${PASTIX_STARPU_VERSION}
      COMPONENTS ${STARPU_COMPONENT_LIST})
  endif()

endif( NOT STARPU_FOUND AND PASTIX_LOOK_FOR_STARPU)

# PASTIX may depends on SCOTCH
#-----------------------------
if (NOT SCOTCH_FOUND AND PASTIX_LOOK_FOR_SCOTCH)
  if (NOT PASTIX_FIND_QUIETLY)
    message(STATUS "Looking for PASTIX - Try to detect SCOTCH")
  endif()
  if (PASTIX_FIND_REQUIRED AND PASTIX_FIND_REQUIRED_SCOTCH)
    find_package(SCOTCH REQUIRED QUIET)
  else()
    find_package(SCOTCH QUIET)
  endif()
endif()

# PASTIX may depends on PTSCOTCH
#-------------------------------
if (NOT PTSCOTCH_FOUND AND PASTIX_LOOK_FOR_PTSCOTCH)
  if (NOT PASTIX_FIND_QUIETLY)
    message(STATUS "Looking for PASTIX - Try to detect PTSCOTCH")
  endif()
  if (PASTIX_FIND_REQUIRED AND PASTIX_FIND_REQUIRED_PTSCOTCH)
    find_package(PTSCOTCH REQUIRED QUIET)
  else()
    find_package(PTSCOTCH QUIET)
  endif()
endif()

# PASTIX may depends on METIS
#----------------------------
if (NOT METIS_FOUND AND PASTIX_LOOK_FOR_METIS)
  if (NOT PASTIX_FIND_QUIETLY)
    message(STATUS "Looking for PASTIX - Try to detect METIS")
  endif()
  if (PASTIX_FIND_REQUIRED AND PASTIX_FIND_REQUIRED_METIS)
    find_package(METIS REQUIRED QUIET)
  else()
    find_package(METIS QUIET)
  endif()
endif()

# Error if pastix required and no partitioning lib found
if (PASTIX_FIND_REQUIRED AND NOT SCOTCH_FOUND AND NOT PTSCOTCH_FOUND AND NOT METIS_FOUND)
  message(FATAL_ERROR "Could NOT find any partitioning library on your system"
    " (install scotch, ptscotch or metis)")
endif()


# Looking for PaStiX
# ------------------

# Looking for include
# -------------------

# Add system include paths to search include
# ------------------------------------------
unset(_inc_env)
set(ENV_PASTIX_DIR "$ENV{PASTIX_DIR}")
set(ENV_PASTIX_INCDIR "$ENV{PASTIX_INCDIR}")
if(ENV_PASTIX_INCDIR)
  list(APPEND _inc_env "${ENV_PASTIX_INCDIR}")
elseif(ENV_PASTIX_DIR)
  list(APPEND _inc_env "${ENV_PASTIX_DIR}")
  list(APPEND _inc_env "${ENV_PASTIX_DIR}/include")
  list(APPEND _inc_env "${ENV_PASTIX_DIR}/include/pastix")
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


# Try to find the pastix header in the given paths
# ---------------------------------------------------
# call cmake macro to find the header path
if(PASTIX_INCDIR)
  set(PASTIX_pastix.h_DIRS "PASTIX_pastix.h_DIRS-NOTFOUND")
  find_path(PASTIX_pastix.h_DIRS
    NAMES pastix.h
    HINTS ${PASTIX_INCDIR})
else()
  if(PASTIX_DIR)
    set(PASTIX_pastix.h_DIRS "PASTIX_pastix.h_DIRS-NOTFOUND")
    find_path(PASTIX_pastix.h_DIRS
      NAMES pastix.h
      HINTS ${PASTIX_DIR}
      PATH_SUFFIXES "include" "include/pastix")
  else()
    set(PASTIX_pastix.h_DIRS "PASTIX_pastix.h_DIRS-NOTFOUND")
    find_path(PASTIX_pastix.h_DIRS
      NAMES pastix.h
      HINTS ${_inc_env}
      PATH_SUFFIXES "pastix")
  endif()
endif()
mark_as_advanced(PASTIX_pastix.h_DIRS)

# If found, add path to cmake variable
# ------------------------------------
if (PASTIX_pastix.h_DIRS)
  set(PASTIX_INCLUDE_DIRS "${PASTIX_pastix.h_DIRS}")
else ()
  set(PASTIX_INCLUDE_DIRS "PASTIX_INCLUDE_DIRS-NOTFOUND")
  if(NOT PASTIX_FIND_QUIETLY)
    message(STATUS "Looking for pastix -- pastix.h not found")
  endif()
endif()


# Looking for lib
# ---------------

# Add system library paths to search lib
# --------------------------------------
unset(_lib_env)
set(ENV_PASTIX_LIBDIR "$ENV{PASTIX_LIBDIR}")
if(ENV_PASTIX_LIBDIR)
  list(APPEND _lib_env "${ENV_PASTIX_LIBDIR}")
elseif(ENV_PASTIX_DIR)
  list(APPEND _lib_env "${ENV_PASTIX_DIR}")
  list(APPEND _lib_env "${ENV_PASTIX_DIR}/lib")
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

# Try to find the pastix lib in the given paths
# ------------------------------------------------

# create list of libs to find
set(PASTIX_libs_to_find "pastix_murge;pastix")

# call cmake macro to find the lib path
if(PASTIX_LIBDIR)
  foreach(pastix_lib ${PASTIX_libs_to_find})
    set(PASTIX_${pastix_lib}_LIBRARY "PASTIX_${pastix_lib}_LIBRARY-NOTFOUND")
    find_library(PASTIX_${pastix_lib}_LIBRARY
      NAMES ${pastix_lib}
      HINTS ${PASTIX_LIBDIR})
  endforeach()
else()
  if(PASTIX_DIR)
    foreach(pastix_lib ${PASTIX_libs_to_find})
      set(PASTIX_${pastix_lib}_LIBRARY "PASTIX_${pastix_lib}_LIBRARY-NOTFOUND")
      find_library(PASTIX_${pastix_lib}_LIBRARY
	NAMES ${pastix_lib}
	HINTS ${PASTIX_DIR}
	PATH_SUFFIXES lib lib32 lib64)
    endforeach()
  else()
    foreach(pastix_lib ${PASTIX_libs_to_find})
      set(PASTIX_${pastix_lib}_LIBRARY "PASTIX_${pastix_lib}_LIBRARY-NOTFOUND")
      find_library(PASTIX_${pastix_lib}_LIBRARY
	NAMES ${pastix_lib}
	HINTS ${_lib_env})
    endforeach()
  endif()
endif()

# If found, add path to cmake variable
# ------------------------------------
foreach(pastix_lib ${PASTIX_libs_to_find})

  get_filename_component(${pastix_lib}_lib_path ${PASTIX_${pastix_lib}_LIBRARY} PATH)
  # set cmake variables (respects naming convention)
  if (PASTIX_LIBRARIES)
    list(APPEND PASTIX_LIBRARIES "${PASTIX_${pastix_lib}_LIBRARY}")
  else()
    set(PASTIX_LIBRARIES "${PASTIX_${pastix_lib}_LIBRARY}")
  endif()
  if (PASTIX_LIBRARY_DIRS)
    list(APPEND PASTIX_LIBRARY_DIRS "${${pastix_lib}_lib_path}")
  else()
    set(PASTIX_LIBRARY_DIRS "${${pastix_lib}_lib_path}")
  endif()
  mark_as_advanced(PASTIX_${pastix_lib}_LIBRARY)

endforeach(pastix_lib ${PASTIX_libs_to_find})

# check a function to validate the find
if(PASTIX_LIBRARIES)

  set(REQUIRED_LDFLAGS)
  set(REQUIRED_INCDIRS)
  set(REQUIRED_LIBDIRS)
  set(REQUIRED_LIBS)

  # PASTIX
  if (PASTIX_INCLUDE_DIRS)
    set(REQUIRED_INCDIRS "${PASTIX_INCLUDE_DIRS}")
  endif()
  foreach(libdir ${PASTIX_LIBRARY_DIRS})
    if (libdir)
      list(APPEND REQUIRED_LIBDIRS "${libdir}")
    endif()
  endforeach()
  set(REQUIRED_LIBS "${PASTIX_LIBRARIES}")
  # STARPU
  if (PASTIX_LOOK_FOR_STARPU AND STARPU_FOUND)
    if (STARPU_INCLUDE_DIRS_DEP)
      list(APPEND REQUIRED_INCDIRS "${STARPU_INCLUDE_DIRS_DEP}")
    elseif (STARPU_INCLUDE_DIRS)
      list(APPEND REQUIRED_INCDIRS "${STARPU_INCLUDE_DIRS}")
    endif()
    if(STARPU_LIBRARY_DIRS_DEP)
      list(APPEND REQUIRED_LIBDIRS "${STARPU_LIBRARY_DIRS_DEP}")
    elseif(STARPU_LIBRARY_DIRS)
      list(APPEND REQUIRED_LIBDIRS "${STARPU_LIBRARY_DIRS}")
    endif()
    if (STARPU_LIBRARIES_DEP)
      list(APPEND REQUIRED_LIBS "${STARPU_LIBRARIES_DEP}")
    elseif (STARPU_LIBRARIES)
      foreach(lib ${STARPU_LIBRARIES})
	if (EXISTS ${lib} OR ${lib} MATCHES "^-")
	  list(APPEND REQUIRED_LIBS "${lib}")
	else()
	  list(APPEND REQUIRED_LIBS "-l${lib}")
	endif()
      endforeach()
    endif()
  endif()
  # CUDA
  if (PASTIX_LOOK_FOR_STARPU_CUDA AND CUDA_FOUND)
    if (CUDA_INCLUDE_DIRS)
      list(APPEND REQUIRED_INCDIRS "${CUDA_INCLUDE_DIRS}")
    endif()
    foreach(libdir ${CUDA_LIBRARY_DIRS})
      if (libdir)
	list(APPEND REQUIRED_LIBDIRS "${libdir}")
      endif()
    endforeach()
    list(APPEND REQUIRED_LIBS "${CUDA_CUBLAS_LIBRARIES};${CUDA_LIBRARIES}")
  endif()
  # MPI
  if (PASTIX_LOOK_FOR_MPI AND MPI_FOUND)
    if (MPI_C_INCLUDE_PATH)
      list(APPEND REQUIRED_INCDIRS "${MPI_C_INCLUDE_PATH}")
    endif()
    if (MPI_C_LINK_FLAGS)
      if (${MPI_C_LINK_FLAGS} MATCHES "  -")
	string(REGEX REPLACE " -" "-" MPI_C_LINK_FLAGS ${MPI_C_LINK_FLAGS})
      endif()
      list(APPEND REQUIRED_LDFLAGS "${MPI_C_LINK_FLAGS}")
    endif()
    list(APPEND REQUIRED_LIBS "${MPI_C_LIBRARIES}")
  endif()
  # HWLOC
  if (HWLOC_FOUND)
    if (HWLOC_INCLUDE_DIRS)
      list(APPEND REQUIRED_INCDIRS "${HWLOC_INCLUDE_DIRS}")
    endif()
    foreach(libdir ${HWLOC_LIBRARY_DIRS})
      if (libdir)
	list(APPEND REQUIRED_LIBDIRS "${libdir}")
      endif()
    endforeach()
    foreach(lib ${HWLOC_LIBRARIES})
      if (EXISTS ${lib} OR ${lib} MATCHES "^-")
	list(APPEND REQUIRED_LIBS "${lib}")
      else()
	list(APPEND REQUIRED_LIBS "-l${lib}")
      endif()
    endforeach()
  endif()
  # BLAS
  if (BLAS_FOUND)
    if (BLAS_INCLUDE_DIRS)
      list(APPEND REQUIRED_INCDIRS "${BLAS_INCLUDE_DIRS}")
    endif()
    foreach(libdir ${BLAS_LIBRARY_DIRS})
      if (libdir)
	list(APPEND REQUIRED_LIBDIRS "${libdir}")
      endif()
    endforeach()
    list(APPEND REQUIRED_LIBS "${BLAS_LIBRARIES}")
    if (BLAS_LINKER_FLAGS)
      list(APPEND REQUIRED_LDFLAGS "${BLAS_LINKER_FLAGS}")
    endif()
  endif()
  # SCOTCH
  if (PASTIX_LOOK_FOR_SCOTCH AND SCOTCH_FOUND)
    if (SCOTCH_INCLUDE_DIRS)
      list(APPEND REQUIRED_INCDIRS "${SCOTCH_INCLUDE_DIRS}")
    endif()
    foreach(libdir ${SCOTCH_LIBRARY_DIRS})
      if (libdir)
	list(APPEND REQUIRED_LIBDIRS "${libdir}")
      endif()
    endforeach()
    list(APPEND REQUIRED_LIBS "${SCOTCH_LIBRARIES}")
  endif()
  # PTSCOTCH
  if (PASTIX_LOOK_FOR_PTSCOTCH AND PTSCOTCH_FOUND)
    if (PTSCOTCH_INCLUDE_DIRS)
      list(APPEND REQUIRED_INCDIRS "${PTSCOTCH_INCLUDE_DIRS}")
    endif()
    foreach(libdir ${PTSCOTCH_LIBRARY_DIRS})
      if (libdir)
	list(APPEND REQUIRED_LIBDIRS "${libdir}")
      endif()
    endforeach()
    list(APPEND REQUIRED_LIBS "${PTSCOTCH_LIBRARIES}")
  endif()
  # METIS
  if (PASTIX_LOOK_FOR_METIS AND METIS_FOUND)
    if (METIS_INCLUDE_DIRS)
      list(APPEND REQUIRED_INCDIRS "${METIS_INCLUDE_DIRS}")
    endif()
    foreach(libdir ${METIS_LIBRARY_DIRS})
      if (libdir)
	list(APPEND REQUIRED_LIBDIRS "${libdir}")
      endif()
    endforeach()
    list(APPEND REQUIRED_LIBS "${METIS_LIBRARIES}")
  endif()
  # Fortran
  if (CMAKE_C_COMPILER_ID MATCHES "GNU")
    find_library(
      FORTRAN_gfortran_LIBRARY
      NAMES gfortran
      HINTS ${_lib_env}
      )
    mark_as_advanced(FORTRAN_gfortran_LIBRARY)
    if (FORTRAN_gfortran_LIBRARY)
      list(APPEND REQUIRED_LIBS "${FORTRAN_gfortran_LIBRARY}")
    endif()
  elseif (CMAKE_C_COMPILER_ID MATCHES "Intel")
    find_library(
      FORTRAN_ifcore_LIBRARY
      NAMES ifcore
      HINTS ${_lib_env}
      )
    mark_as_advanced(FORTRAN_ifcore_LIBRARY)
    if (FORTRAN_ifcore_LIBRARY)
      list(APPEND REQUIRED_LIBS "${FORTRAN_ifcore_LIBRARY}")
    endif()
  endif()
  # EXTRA LIBS such that pthread, m, rt
  list(APPEND REQUIRED_LIBS ${PASTIX_EXTRA_LIBRARIES})

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
  unset(PASTIX_WORKS CACHE)
  include(CheckFunctionExists)
  check_function_exists(pastix PASTIX_WORKS)
  mark_as_advanced(PASTIX_WORKS)

  if(PASTIX_WORKS)
    # save link with dependencies
    set(PASTIX_LIBRARIES_DEP "${REQUIRED_LIBS}")
    set(PASTIX_LIBRARY_DIRS_DEP "${REQUIRED_LIBDIRS}")
    set(PASTIX_INCLUDE_DIRS_DEP "${REQUIRED_INCDIRS}")
    set(PASTIX_LINKER_FLAGS "${REQUIRED_LDFLAGS}")
    list(REMOVE_DUPLICATES PASTIX_LIBRARY_DIRS_DEP)
    list(REMOVE_DUPLICATES PASTIX_INCLUDE_DIRS_DEP)
    list(REMOVE_DUPLICATES PASTIX_LINKER_FLAGS)
  else()
    if(NOT PASTIX_FIND_QUIETLY)
      message(STATUS "Looking for PASTIX : test of pastix() fails")
      message(STATUS "CMAKE_REQUIRED_LIBRARIES: ${CMAKE_REQUIRED_LIBRARIES}")
      message(STATUS "CMAKE_REQUIRED_INCLUDES: ${CMAKE_REQUIRED_INCLUDES}")
      message(STATUS "Check in CMakeFiles/CMakeError.log to figure out why it fails")
      message(STATUS "Maybe PASTIX is linked with specific libraries. "
	"Have you tried with COMPONENTS (MPI/SEQ, STARPU, STARPU_CUDA, SCOTCH, PTSCOTCH, METIS)? "
	"See the explanation in FindPASTIX.cmake.")
    endif()
  endif()
  set(CMAKE_REQUIRED_INCLUDES)
  set(CMAKE_REQUIRED_FLAGS)
  set(CMAKE_REQUIRED_LIBRARIES)
endif(PASTIX_LIBRARIES)

if (PASTIX_LIBRARIES)
  list(GET PASTIX_LIBRARIES 0 first_lib)
  get_filename_component(first_lib_path "${first_lib}" PATH)
  if (${first_lib_path} MATCHES "/lib(32|64)?$")
    string(REGEX REPLACE "/lib(32|64)?$" "" not_cached_dir "${first_lib_path}")
    set(PASTIX_DIR_FOUND "${not_cached_dir}" CACHE PATH "Installation directory of PASTIX library" FORCE)
  else()
    set(PASTIX_DIR_FOUND "${first_lib_path}" CACHE PATH "Installation directory of PASTIX library" FORCE)
  endif()
endif()
mark_as_advanced(PASTIX_DIR)
mark_as_advanced(PASTIX_DIR_FOUND)

# check that PASTIX has been found
# ---------------------------------
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(PASTIX DEFAULT_MSG
  PASTIX_LIBRARIES
  PASTIX_WORKS)
