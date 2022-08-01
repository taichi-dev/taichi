# ./taichi/runtime/llvm/CMakeLists.txt

add_library(llvm_runtime)
target_sources(llvm_runtime
  PRIVATE
    llvm_runtime_executor.cpp
    llvm_offline_cache.cpp
    llvm_context.cpp
    llvm_aot_module_loader.cpp
    llvm_aot_module_builder.cpp
    launch_arg_info.cpp
    snode_tree_buffer_manager.cpp
  )

target_include_directories(llvm_runtime
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/external/eigen
    ${PROJECT_SOURCE_DIR}/external/spdlog/include
    ${LLVM_INCLUDE_DIRS}
  )

target_link_libraries(llvm_runtime PRIVATE llvm_rhi)
target_link_libraries(llvm_runtime PRIVATE ${llvm_libs})


if (APPLE AND "${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "arm64")
    target_link_libraries(llvm_runtime PRIVATE ${llvm_aarch64_libs})
endif()

if (TI_WITH_CUDA)
    target_link_libraries(llvm_runtime PRIVATE ${llvm_ptx_libs})
    target_link_libraries(llvm_runtime PRIVATE cuda_rhi)
endif()
