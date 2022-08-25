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

# Build llvm-runtime for host arch and cuda (if available)
foreach(arch IN LISTS HOST_ARCH CUDA_ARCH DX12_ARCH)
  compile_llvm_runtime(${arch})
endforeach()
