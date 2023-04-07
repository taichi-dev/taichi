# ./taichi/runtime/llvm/runtime_module/CMakeLists.txt

function(COMPILE_LLVM_RUNTIME rtm_arch)
    message(STATUS "Compiling LLVM byte code file for arch ${rtm_arch}")
    # Keep this for now, as .bc need to be generated.
    add_custom_target(
        "generate_llvm_runtime_${rtm_arch}"
        COMMAND ${CLANG_EXECUTABLE} ${CLANG_OSX_FLAGS} -c runtime.cpp -o "runtime_${rtm_arch}.bc" -fno-exceptions -emit-llvm -std=c++17 -D "ARCH_${rtm_arch}" -I ${PROJECT_SOURCE_DIR};
        # TODO, it's better to avoid polluting the source dir, keep in build
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    )
    add_dependencies(${CORE_LIBRARY_NAME} "generate_llvm_runtime_${rtm_arch}")
    install(FILES "${CMAKE_SOURCE_DIR}/taichi/runtime/llvm/runtime_module/runtime_${arch}.bc" DESTINATION ${CMAKE_INSTALL_PREFIX}/python/taichi/_lib/runtime)

endfunction()

function(COMPILE_CUSTOM_CUDA_LIBRARY cuda_path)
    add_custom_target(
        "generate_cuda_library"
        COMMAND ${CLANG_EXECUTABLE} -c cuda_runtime.cu --cuda-gpu-arch=sm_60 -emit-llvm -std=c++17 --no-cuda-version-check -nocudalib --cuda-path=${cuda_path}
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    )
    add_dependencies(${CORE_LIBRARY_NAME} "generate_cuda_library")
    install(FILES "${CMAKE_SOURCE_DIR}/taichi/runtime/llvm/runtime_module/cuda_runtime-cuda-nvptx64-nvidia-cuda-sm_60.bc" DESTINATION ${CMAKE_INSTALL_PREFIX}/python/taichi/_lib/runtime)

endfunction()

# Build llvm-runtime for host arch and cuda (if available)
foreach(arch IN LISTS HOST_ARCH CUDA_ARCH DX12_ARCH AMDGPU_ARCH)
  compile_llvm_runtime(${arch})
endforeach()

# You can implement hand-written CUDA Kernels in cuda_runtime.cu and then
# uncomment the following code to compile it into .bc files

#if(CUDA_ARCH)
#    find_package(CUDAToolkit)
#    if(${CUDAToolkit_FOUND})
#      if(${CUDAToolkit_VERSION_MAJOR} GREATER_EQUAL 8)
#          compile_custom_cuda_library(${CUDAToolkit_TARGET_DIR})
#      endif()
#    endif()
#endif()
