# Try to find the MPFR library
# See http://www.mpfr.org/
#
# This module supports requiring a minimum version, e.g. you can do
#   find_package(MPFR 2.3.0)
# to require version 2.3.0 to newer of MPFR.
#
# Once done this will define
#
#  MPFR_FOUND - system has MPFR lib with correct version
#  MPFR_INCLUDES - the MPFR include directory
#  MPFR_LIBRARIES - the MPFR library
#  MPFR_VERSION - MPFR version

# Copyright (c) 2006, 2007 Montel Laurent, <montel@kde.org>
# Copyright (c) 2008, 2009 Gael Guennebaud, <g.gael@free.fr>
# Copyright (c) 2010 Jitse Niesen, <jitse@maths.leeds.ac.uk>
# Redistribution and use is allowed according to the terms of the BSD license.

# Set MPFR_INCLUDES

find_path(MPFR_INCLUDES
  NAMES
  mpfr.h
  PATHS
  $ENV{GMPDIR}
  ${INCLUDE_INSTALL_DIR}
)

# Set MPFR_FIND_VERSION to 1.0.0 if no minimum version is specified

if(NOT MPFR_FIND_VERSION)
  if(NOT MPFR_FIND_VERSION_MAJOR)
    set(MPFR_FIND_VERSION_MAJOR 1)
  endif(NOT MPFR_FIND_VERSION_MAJOR)
  if(NOT MPFR_FIND_VERSION_MINOR)
    set(MPFR_FIND_VERSION_MINOR 0)
  endif(NOT MPFR_FIND_VERSION_MINOR)
  if(NOT MPFR_FIND_VERSION_PATCH)
    set(MPFR_FIND_VERSION_PATCH 0)
  endif(NOT MPFR_FIND_VERSION_PATCH)

  set(MPFR_FIND_VERSION "${MPFR_FIND_VERSION_MAJOR}.${MPFR_FIND_VERSION_MINOR}.${MPFR_FIND_VERSION_PATCH}")
endif(NOT MPFR_FIND_VERSION)


if(MPFR_INCLUDES)

  # Set MPFR_VERSION
  
  file(READ "${MPFR_INCLUDES}/mpfr.h" _mpfr_version_header)
  
  string(REGEX MATCH "define[ \t]+MPFR_VERSION_MAJOR[ \t]+([0-9]+)" _mpfr_major_version_match "${_mpfr_version_header}")
  set(MPFR_MAJOR_VERSION "${CMAKE_MATCH_1}")
  string(REGEX MATCH "define[ \t]+MPFR_VERSION_MINOR[ \t]+([0-9]+)" _mpfr_minor_version_match "${_mpfr_version_header}")
  set(MPFR_MINOR_VERSION "${CMAKE_MATCH_1}")
  string(REGEX MATCH "define[ \t]+MPFR_VERSION_PATCHLEVEL[ \t]+([0-9]+)" _mpfr_patchlevel_version_match "${_mpfr_version_header}")
  set(MPFR_PATCHLEVEL_VERSION "${CMAKE_MATCH_1}")
  
  set(MPFR_VERSION ${MPFR_MAJOR_VERSION}.${MPFR_MINOR_VERSION}.${MPFR_PATCHLEVEL_VERSION})
  
  # Check whether found version exceeds minimum version
  
  if(${MPFR_VERSION} VERSION_LESS ${MPFR_FIND_VERSION})
    set(MPFR_VERSION_OK FALSE)
    message(STATUS "MPFR version ${MPFR_VERSION} found in ${MPFR_INCLUDES}, "
                   "but at least version ${MPFR_FIND_VERSION} is required")
  else(${MPFR_VERSION} VERSION_LESS ${MPFR_FIND_VERSION})
    set(MPFR_VERSION_OK TRUE)
  endif(${MPFR_VERSION} VERSION_LESS ${MPFR_FIND_VERSION})

endif(MPFR_INCLUDES)

# Set MPFR_LIBRARIES

find_library(MPFR_LIBRARIES mpfr PATHS $ENV{GMPDIR} ${LIB_INSTALL_DIR})

# Epilogue

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MPFR DEFAULT_MSG
                                  MPFR_INCLUDES MPFR_LIBRARIES MPFR_VERSION_OK)
mark_as_advanced(MPFR_INCLUDES MPFR_LIBRARIES)
