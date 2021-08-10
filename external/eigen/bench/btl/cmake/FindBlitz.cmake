# - Try to find blitz lib
# Once done this will define
#
#  BLITZ_FOUND - system has blitz lib
#  BLITZ_INCLUDES - the blitz include directory
#  BLITZ_LIBRARIES - The libraries needed to use blitz

# Copyright (c) 2006, Montel Laurent, <montel@kde.org>
# Copyright (c) 2007, Allen Winter, <winter@kde.org>
# Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
# Redistribution and use is allowed according to the terms of the BSD license.
# For details see the accompanying COPYING-CMAKE-SCRIPTS file.

# include(FindLibraryWithDebug)

if (BLITZ_INCLUDES AND BLITZ_LIBRARIES)
  set(Blitz_FIND_QUIETLY TRUE)
endif (BLITZ_INCLUDES AND BLITZ_LIBRARIES)

find_path(BLITZ_INCLUDES
  NAMES
  blitz/array.h
  PATH_SUFFIXES blitz*
  PATHS
  $ENV{BLITZDIR}/include
  ${INCLUDE_INSTALL_DIR}
)

find_library(BLITZ_LIBRARIES
  blitz
  PATHS
  $ENV{BLITZDIR}/lib
  ${LIB_INSTALL_DIR}
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Blitz DEFAULT_MSG
                                  BLITZ_INCLUDES BLITZ_LIBRARIES)

mark_as_advanced(BLITZ_INCLUDES BLITZ_LIBRARIES)
