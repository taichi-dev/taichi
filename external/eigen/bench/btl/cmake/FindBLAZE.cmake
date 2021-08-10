# - Try to find eigen2 headers
# Once done this will define
#
#  BLAZE_FOUND - system has blaze lib
#  BLAZE_INCLUDE_DIR - the blaze include directory
#
# Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
# Adapted from FindEigen.cmake:
# Copyright (c) 2006, 2007 Montel Laurent, <montel@kde.org>
# Redistribution and use is allowed according to the terms of the BSD license.
# For details see the accompanying COPYING-CMAKE-SCRIPTS file.

if (BLAZE_INCLUDE_DIR)

  # in cache already
  set(BLAZE_FOUND TRUE)

else (BLAZE_INCLUDE_DIR)

find_path(BLAZE_INCLUDE_DIR NAMES blaze/Blaze.h
     PATHS
     ${INCLUDE_INSTALL_DIR}
   )

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(BLAZE DEFAULT_MSG BLAZE_INCLUDE_DIR)

mark_as_advanced(BLAZE_INCLUDE_DIR)

endif(BLAZE_INCLUDE_DIR)

