# ./taichi/runtime/gfx/CMakeLists.txt

add_library(gfx_runtime)
target_sources(gfx_runtime
  PRIVATE
    runtime.cpp
    snode_tree_manager.cpp
    aot_module_builder_impl.cpp
    aot_module_loader_impl.cpp
  )
#TODO 4832, some dependencies here should not be required as they
# are build requirements of other targets.
target_include_directories(gfx_runtime
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/external/SPIRV-Tools/include
    ${PROJECT_SOURCE_DIR}/external/eigen
    ${PROJECT_SOURCE_DIR}/external/FP16/include
  PRIVATE
    ${PROJECT_SOURCE_DIR}/external/spdlog/include
        ${LLVM_INCLUDE_DIRS}
  )
