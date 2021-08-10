# SPQR lib usually requires linking to a blas and lapack library.
# It is up to the user of this module to find a BLAS and link to it.

# SPQR lib requires Cholmod, colamd and amd as well. 
# FindCholmod.cmake can be used to find those packages before finding spqr

if (SPQR_INCLUDES AND SPQR_LIBRARIES)
  set(SPQR_FIND_QUIETLY TRUE)
endif (SPQR_INCLUDES AND SPQR_LIBRARIES)

find_path(SPQR_INCLUDES
  NAMES
  SuiteSparseQR.hpp
  PATHS
  $ENV{SPQRDIR}
  ${INCLUDE_INSTALL_DIR}
  PATH_SUFFIXES
  suitesparse
  ufsparse
)

find_library(SPQR_LIBRARIES spqr $ENV{SPQRDIR} ${LIB_INSTALL_DIR})

if(SPQR_LIBRARIES)

  find_library(SUITESPARSE_LIBRARY SuiteSparse PATHS $ENV{SPQRDIR} ${LIB_INSTALL_DIR})
  if (SUITESPARSE_LIBRARY)
    set(SPQR_LIBRARIES ${SPQR_LIBRARIES} ${SUITESPARSE_LIBRARY})
  endif()

  find_library(CHOLMOD_LIBRARY cholmod PATHS $ENV{UMFPACK_LIBDIR} $ENV{UMFPACKDIR} ${LIB_INSTALL_DIR})
  if(CHOLMOD_LIBRARY)
    set(SPQR_LIBRARIES ${SPQR_LIBRARIES} ${CHOLMOD_LIBRARY})
  endif()
  
endif(SPQR_LIBRARIES)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(SPQR DEFAULT_MSG SPQR_INCLUDES SPQR_LIBRARIES)

mark_as_advanced(SPQR_INCLUDES SPQR_LIBRARIES)