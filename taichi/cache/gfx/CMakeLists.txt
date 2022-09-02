# ./taichi/cache/gfx/CMakeLists.txt

add_library(gfx_cache)
target_sources(gfx_cache
  PRIVATE
    cache_manager.cpp
  )

target_include_directories(gfx_cache
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/external/spdlog/include
    ${PROJECT_SOURCE_DIR}/external/SPIRV-Tools/include
    ${LLVM_INCLUDE_DIRS} # For "llvm/ADT/SmallVector.h" included in ir.h
  )

target_link_libraries(gfx_cache PRIVATE spirv_codegen)
target_link_libraries(gfx_cache PRIVATE gfx_runtime)
