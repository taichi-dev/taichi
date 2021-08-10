
# Umfpack lib usually requires linking to a blas library.
# It is up to the user of this module to find a BLAS and link to it.

if (SUPERLU_INCLUDES AND SUPERLU_LIBRARIES)
  set(SUPERLU_FIND_QUIETLY TRUE)
endif (SUPERLU_INCLUDES AND SUPERLU_LIBRARIES)

find_path(SUPERLU_INCLUDES
  NAMES
  supermatrix.h
  PATHS
  $ENV{SUPERLUDIR}
  ${INCLUDE_INSTALL_DIR}
  PATH_SUFFIXES
  superlu
  SRC
)

find_library(SUPERLU_LIBRARIES
  NAMES "superlu_5.2.1" "superlu_5.2" "superlu_5.1.1" "superlu_5.1" "superlu_5.0" "superlu_4.3" "superlu_4.2" "superlu_4.1" "superlu_4.0" "superlu_3.1" "superlu_3.0" "superlu"
  PATHS $ENV{SUPERLUDIR} ${LIB_INSTALL_DIR}
  PATH_SUFFIXES lib)

if(SUPERLU_INCLUDES AND SUPERLU_LIBRARIES)

include(CheckCXXSourceCompiles)
include(CMakePushCheckState)
cmake_push_check_state()

set(CMAKE_REQUIRED_INCLUDES ${CMAKE_REQUIRED_INCLUDES} ${SUPERLU_INCLUDES})

# check whether struct mem_usage_t is globally defined
check_cxx_source_compiles("
typedef int int_t;
#include <supermatrix.h>
#include <slu_util.h>
int main() {
  mem_usage_t mem;
  return 0;
}"
SUPERLU_HAS_GLOBAL_MEM_USAGE_T)


check_cxx_source_compiles("
typedef int int_t;
#include <supermatrix.h>
#include <superlu_enum_consts.h>
int main() {
  return SLU_SINGLE;
}"
SUPERLU_HAS_CLEAN_ENUMS)

check_cxx_source_compiles("
typedef int int_t;
#include <supermatrix.h>
#include <slu_util.h>
int main(void)
{
  GlobalLU_t glu;
  return 0;
}"
SUPERLU_HAS_GLOBALLU_T)

if(SUPERLU_HAS_GLOBALLU_T)
  # at least 5.0
  set(SUPERLU_VERSION_VAR "5.0")
elseif(SUPERLU_HAS_CLEAN_ENUMS)
  # at least 4.3
  set(SUPERLU_VERSION_VAR "4.3")
elseif(SUPERLU_HAS_GLOBAL_MEM_USAGE_T)
  # at least 4.0
  set(SUPERLU_VERSION_VAR "4.0")
else()
  set(SUPERLU_VERSION_VAR "3.0")
endif()

cmake_pop_check_state()

if(SuperLU_FIND_VERSION)
  if(${SUPERLU_VERSION_VAR} VERSION_LESS ${SuperLU_FIND_VERSION})
    set(SUPERLU_VERSION_OK FALSE)
  else()
    set(SUPERLU_VERSION_OK TRUE)
  endif()
else()
  set(SUPERLU_VERSION_OK TRUE)
endif()

endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(SUPERLU
                                  REQUIRED_VARS SUPERLU_INCLUDES SUPERLU_LIBRARIES SUPERLU_VERSION_OK
                                  VERSION_VAR SUPERLU_VERSION_VAR)

mark_as_advanced(SUPERLU_INCLUDES SUPERLU_LIBRARIES)
