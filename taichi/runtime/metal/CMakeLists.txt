# ./taichi/runtime/metal/CMakeLists.txt

add_library(metal_runtime)
target_sources(metal_runtime
  PRIVATE
    api.cpp
    aot_module_builder_impl.cpp
    aot_module_loader_impl.cpp
    data_types.cpp
    kernel_manager.cpp
    kernel_utils.cpp
    runtime_utils.cpp
  )

target_include_directories(metal_runtime
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders
    ${PROJECT_SOURCE_DIR}/external/SPIRV-Tools/include
    ${PROJECT_SOURCE_DIR}/external/glad/include
    ${PROJECT_SOURCE_DIR}/external/eigen
    ${PROJECT_SOURCE_DIR}/external/glfw/include
    ${LLVM_INCLUDE_DIRS}
  PUBLIC
    ${PROJECT_SOURCE_DIR}/external/spdlog/include
  )

target_link_libraries(metal_runtime PRIVATE metal_rhi)
