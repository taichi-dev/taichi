
if (ADOLC_INCLUDES AND ADOLC_LIBRARIES)
  set(ADOLC_FIND_QUIETLY TRUE)
endif (ADOLC_INCLUDES AND ADOLC_LIBRARIES)

find_path(ADOLC_INCLUDES
  NAMES
  adolc/adtl.h
  PATHS
  $ENV{ADOLCDIR}
  ${INCLUDE_INSTALL_DIR}
)

find_library(ADOLC_LIBRARIES adolc PATHS $ENV{ADOLCDIR} ${LIB_INSTALL_DIR})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ADOLC DEFAULT_MSG
                                  ADOLC_INCLUDES ADOLC_LIBRARIES)

mark_as_advanced(ADOLC_INCLUDES ADOLC_LIBRARIES)
