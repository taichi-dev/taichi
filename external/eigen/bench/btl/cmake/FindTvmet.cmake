# - Try to find tvmet headers
# Once done this will define
#
#  TVMET_FOUND - system has tvmet lib
#  TVMET_INCLUDE_DIR - the tvmet include directory
#
# Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
# Adapted from FindEigen.cmake:
# Copyright (c) 2006, 2007 Montel Laurent, <montel@kde.org>
# Redistribution and use is allowed according to the terms of the BSD license.
# For details see the accompanying COPYING-CMAKE-SCRIPTS file.

if (TVMET_INCLUDE_DIR)

  # in cache already
  set(TVMET_FOUND TRUE)

else (TVMET_INCLUDE_DIR)

find_path(TVMET_INCLUDE_DIR NAMES tvmet/tvmet.h
     PATHS
     ${TVMETDIR}/
     ${INCLUDE_INSTALL_DIR}
   )

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Tvmet DEFAULT_MSG TVMET_INCLUDE_DIR)

mark_as_advanced(TVMET_INCLUDE_DIR)

endif(TVMET_INCLUDE_DIR)

