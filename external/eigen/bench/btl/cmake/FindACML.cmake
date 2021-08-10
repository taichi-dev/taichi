
if (ACML_LIBRARIES)
  set(ACML_FIND_QUIETLY TRUE)
endif (ACML_LIBRARIES)

find_library(ACML_LIBRARIES
  NAMES
  acml_mp acml_mv
  PATHS
  $ENV{ACMLDIR}/lib
  $ENV{ACML_DIR}/lib
  ${LIB_INSTALL_DIR}
)

find_file(ACML_LIBRARIES
  NAMES
  libacml_mp.so
  PATHS
  /usr/lib
  /usr/lib64
  $ENV{ACMLDIR}/lib
  ${LIB_INSTALL_DIR}
)

if(NOT ACML_LIBRARIES)
    message(STATUS "Multi-threaded library not found, looking for single-threaded")
    find_library(ACML_LIBRARIES
        NAMES
        acml acml_mv
        PATHS
        $ENV{ACMLDIR}/lib
        $ENV{ACML_DIR}/lib
        ${LIB_INSTALL_DIR}
        )
    find_file(ACML_LIBRARIES
        libacml.so libacml_mv.so
        PATHS
        /usr/lib
        /usr/lib64
        $ENV{ACMLDIR}/lib
        ${LIB_INSTALL_DIR}
        )
endif()




include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ACML DEFAULT_MSG ACML_LIBRARIES)

mark_as_advanced(ACML_LIBRARIES)
