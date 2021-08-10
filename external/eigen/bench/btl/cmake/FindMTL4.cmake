# - Try to find eigen2 headers
# Once done this will define
#
#  MTL4_FOUND - system has eigen2 lib
#  MTL4_INCLUDE_DIR - the eigen2 include directory
#
# Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
# Adapted from FindEigen.cmake:
# Copyright (c) 2006, 2007 Montel Laurent, <montel@kde.org>
# Redistribution and use is allowed according to the terms of the BSD license.
# For details see the accompanying COPYING-CMAKE-SCRIPTS file.

if (MTL4_INCLUDE_DIR)

  # in cache already
  set(MTL4_FOUND TRUE)

else (MTL4_INCLUDE_DIR)

find_path(MTL4_INCLUDE_DIR NAMES boost/numeric/mtl/mtl.hpp
     PATHS
     ${INCLUDE_INSTALL_DIR}
   )

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MTL4 DEFAULT_MSG MTL4_INCLUDE_DIR)

mark_as_advanced(MTL4_INCLUDE_DIR)

endif(MTL4_INCLUDE_DIR)

