# ./taichi/runtime/gfx/CMakeLists.txt

add_library(gfx_runtime)
target_sources(gfx_runtime
  PRIVATE
    runtime.cpp
    snode_tree_manager.cpp
    aot_module_builder_impl.cpp
    aot_module_loader_impl.cpp
  )
target_include_directories(gfx_runtime
  PRIVATE
    ${PROJECT_SOURCE_DIR}/external/SPIRV-Tools/include
    ${PROJECT_SOURCE_DIR}/external/eigen
    ${PROJECT_SOURCE_DIR}/external/FP16/include
  )
