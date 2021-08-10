# - Try to find Eigen2 lib
#
# This module supports requiring a minimum version, e.g. you can do
#   find_package(Eigen2 2.0.3)
# to require version 2.0.3 to newer of Eigen2.
#
# Once done this will define
#
#  EIGEN2_FOUND - system has eigen lib with correct version
#  EIGEN2_INCLUDE_DIR - the eigen include directory
#  EIGEN2_VERSION - eigen version

# Copyright (c) 2006, 2007 Montel Laurent, <montel@kde.org>
# Copyright (c) 2008, 2009 Gael Guennebaud, <g.gael@free.fr>
# Redistribution and use is allowed according to the terms of the BSD license.

if(NOT Eigen2_FIND_VERSION)
  if(NOT Eigen2_FIND_VERSION_MAJOR)
    set(Eigen2_FIND_VERSION_MAJOR 2)
  endif(NOT Eigen2_FIND_VERSION_MAJOR)
  if(NOT Eigen2_FIND_VERSION_MINOR)
    set(Eigen2_FIND_VERSION_MINOR 0)
  endif(NOT Eigen2_FIND_VERSION_MINOR)
  if(NOT Eigen2_FIND_VERSION_PATCH)
    set(Eigen2_FIND_VERSION_PATCH 0)
  endif(NOT Eigen2_FIND_VERSION_PATCH)

  set(Eigen2_FIND_VERSION "${Eigen2_FIND_VERSION_MAJOR}.${Eigen2_FIND_VERSION_MINOR}.${Eigen2_FIND_VERSION_PATCH}")
endif(NOT Eigen2_FIND_VERSION)

macro(_eigen2_check_version)
  file(READ "${EIGEN2_INCLUDE_DIR}/Eigen/src/Core/util/Macros.h" _eigen2_version_header)

  string(REGEX MATCH "define[ \t]+EIGEN_WORLD_VERSION[ \t]+([0-9]+)" _eigen2_world_version_match "${_eigen2_version_header}")
  set(EIGEN2_WORLD_VERSION "${CMAKE_MATCH_1}")
  string(REGEX MATCH "define[ \t]+EIGEN_MAJOR_VERSION[ \t]+([0-9]+)" _eigen2_major_version_match "${_eigen2_version_header}")
  set(EIGEN2_MAJOR_VERSION "${CMAKE_MATCH_1}")
  string(REGEX MATCH "define[ \t]+EIGEN_MINOR_VERSION[ \t]+([0-9]+)" _eigen2_minor_version_match "${_eigen2_version_header}")
  set(EIGEN2_MINOR_VERSION "${CMAKE_MATCH_1}")

  set(EIGEN2_VERSION ${EIGEN2_WORLD_VERSION}.${EIGEN2_MAJOR_VERSION}.${EIGEN2_MINOR_VERSION})
  if((${EIGEN2_WORLD_VERSION} NOTEQUAL 2) OR (${EIGEN2_MAJOR_VERSION} GREATER 10) OR (${EIGEN2_VERSION} VERSION_LESS ${Eigen2_FIND_VERSION}))
    set(EIGEN2_VERSION_OK FALSE)
  else()
    set(EIGEN2_VERSION_OK TRUE)
  endif()

  if(NOT EIGEN2_VERSION_OK)

    message(STATUS "Eigen2 version ${EIGEN2_VERSION} found in ${EIGEN2_INCLUDE_DIR}, "
                   "but at least version ${Eigen2_FIND_VERSION} is required")
  endif(NOT EIGEN2_VERSION_OK)
endmacro(_eigen2_check_version)

if (EIGEN2_INCLUDE_DIR)

  # in cache already
  _eigen2_check_version()
  set(EIGEN2_FOUND ${EIGEN2_VERSION_OK})

else (EIGEN2_INCLUDE_DIR)

find_path(EIGEN2_INCLUDE_DIR NAMES Eigen/Core
     PATHS
     ${INCLUDE_INSTALL_DIR}
     ${KDE4_INCLUDE_DIR}
     PATH_SUFFIXES eigen2
   )

if(EIGEN2_INCLUDE_DIR)
  _eigen2_check_version()
endif(EIGEN2_INCLUDE_DIR)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Eigen2 DEFAULT_MSG EIGEN2_INCLUDE_DIR EIGEN2_VERSION_OK)

mark_as_advanced(EIGEN2_INCLUDE_DIR)

endif(EIGEN2_INCLUDE_DIR)

