
if (GOOGLEHASH_INCLUDES AND GOOGLEHASH_LIBRARIES)
  set(GOOGLEHASH_FIND_QUIETLY TRUE)
endif (GOOGLEHASH_INCLUDES AND GOOGLEHASH_LIBRARIES)

find_path(GOOGLEHASH_INCLUDES
  NAMES
  google/dense_hash_map
  PATHS
  ${INCLUDE_INSTALL_DIR}
)

if(GOOGLEHASH_INCLUDES)
  # let's make sure it compiles with the current compiler
  file(WRITE ${CMAKE_BINARY_DIR}/googlehash_test.cpp
  "#include <google/sparse_hash_map>\n#include <google/dense_hash_map>\nint main(int argc, char** argv) { google::dense_hash_map<int,float> a; google::sparse_hash_map<int,float> b; return 0;}\n")
  try_compile(GOOGLEHASH_COMPILE ${CMAKE_BINARY_DIR} ${CMAKE_BINARY_DIR}/googlehash_test.cpp OUTPUT_VARIABLE GOOGLEHASH_COMPILE_RESULT)
endif(GOOGLEHASH_INCLUDES)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GOOGLEHASH DEFAULT_MSG GOOGLEHASH_INCLUDES GOOGLEHASH_COMPILE)

mark_as_advanced(GOOGLEHASH_INCLUDES)
