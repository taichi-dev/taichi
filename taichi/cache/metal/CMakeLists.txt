# ./taichi/cache/metal/CMakeLists.txt

add_library(metal_cache)
target_sources(metal_cache
  PRIVATE
    cache_manager.cpp
  )

target_include_directories(metal_cache
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/external/spdlog/include
    ${PROJECT_SOURCE_DIR}/external/eigen
    ${LLVM_INCLUDE_DIRS} # For "llvm/ADT/SmallVector.h" included in ir.h
  )

target_link_libraries(metal_cache PRIVATE metal_codegen)
