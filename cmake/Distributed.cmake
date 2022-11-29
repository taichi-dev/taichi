if (DISTRIBUTED_COMPILE)

if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    message(FATAL_ERROR "Distruted compiling only supports Clang for now." )
endif()

message("Enabling distributed compiling quirks")

execute_process(
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  COMMAND ${CMAKE_CXX_COMPILER} -print-target-triple
  OUTPUT_VARIABLE TRIPLET
  OUTPUT_STRIP_TRAILING_WHITESPACE)

# Essentially cross compiling
add_compile_options(--target=${TRIPLET})

# False alarm caused by macro expansion
# if(LSB_SET(x)) => if((x & 1))
add_compile_options(-Wno-parentheses-equality)

# False alarm caused by macro expansion
# taichi/python/export_lang.cpp:1241
# MAKE_SPARSE_MATRIX(32, ColMajor, f)
add_compile_options(-Wno-self-assign-overloaded)

endif()  # DISTRIBUTED_COMPILE
