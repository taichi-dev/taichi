if (GMM_INCLUDE_DIR)
  # in cache already
  set(GMM_FOUND TRUE)
else (GMM_INCLUDE_DIR)

find_path(GMM_INCLUDE_DIR NAMES gmm/gmm.h
     PATHS
     ${INCLUDE_INSTALL_DIR}
     ${GMM_INCLUDE_PATH}
   )

include(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(GMM DEFAULT_MSG GMM_INCLUDE_DIR )

mark_as_advanced(GMM_INCLUDE_DIR)

endif(GMM_INCLUDE_DIR)
