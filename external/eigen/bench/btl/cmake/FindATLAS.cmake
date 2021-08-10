
if (ATLAS_LIBRARIES)
  set(ATLAS_FIND_QUIETLY TRUE)
endif (ATLAS_LIBRARIES)

find_file(ATLAS_LIB libatlas.so.3 PATHS /usr/lib /usr/lib/atlas /usr/lib64 /usr/lib64/atlas $ENV{ATLASDIR} ${LIB_INSTALL_DIR})
find_library(ATLAS_LIB satlas PATHS $ENV{ATLASDIR} ${LIB_INSTALL_DIR})

find_file(ATLAS_LAPACK NAMES liblapack_atlas.so.3 liblapack.so.3 PATHS /usr/lib /usr/lib/atlas /usr/lib64 /usr/lib64/atlas $ENV{ATLASDIR} ${LIB_INSTALL_DIR})
find_library(ATLAS_LAPACK NAMES lapack_atlas lapack PATHS $ENV{ATLASDIR} ${LIB_INSTALL_DIR})

find_file(ATLAS_F77BLAS libf77blas.so.3 PATHS /usr/lib /usr/lib/atlas /usr/lib64 /usr/lib64/atlas $ENV{ATLASDIR} ${LIB_INSTALL_DIR})
find_library(ATLAS_F77BLAS f77blas PATHS $ENV{ATLASDIR} ${LIB_INSTALL_DIR})

if(ATLAS_LIB AND ATLAS_CBLAS AND ATLAS_LAPACK AND ATLAS_F77BLAS)

  set(ATLAS_LIBRARIES ${ATLAS_LAPACK}  ${ATLAS_LIB})
  
  # search the default lapack lib link to it
  find_file(ATLAS_REFERENCE_LAPACK liblapack.so.3 PATHS /usr/lib /usr/lib64)
  find_library(ATLAS_REFERENCE_LAPACK NAMES lapack)
#   if(ATLAS_REFERENCE_LAPACK)
#     set(ATLAS_LIBRARIES ${ATLAS_LIBRARIES} ${ATLAS_REFERENCE_LAPACK})
#   endif()
  
endif(ATLAS_LIB AND ATLAS_CBLAS AND ATLAS_LAPACK AND ATLAS_F77BLAS)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ATLAS DEFAULT_MSG ATLAS_LIBRARIES)

mark_as_advanced(ATLAS_LIBRARIES)
