# ./taichi/codegen/spirv/CMakeLists.txt

add_library(spirv_codegen)
target_sources(spirv_codegen
  PRIVATE
    kernel_utils.cpp
    snode_struct_compiler.cpp
    spirv_codegen.cpp
    spirv_ir_builder.cpp
    spirv_types.cpp
    compiled_kernel_data.cpp
    kernel_compiler.cpp
  )

target_include_directories(spirv_codegen
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/external/eigen
    ${PROJECT_SOURCE_DIR}/external/include
    ${PROJECT_SOURCE_DIR}/external/SPIRV-Headers/include
    ${PROJECT_SOURCE_DIR}/external/SPIRV-Reflect
    ${PROJECT_SOURCE_DIR}/external/FP16/include
    ${LLVM_INCLUDE_DIRS}
  )

target_link_libraries(spirv_codegen PRIVATE taichi_util)
# NOTE: SPIRV-Tools-opt must come before SPIRV-Tools
# https://github.com/KhronosGroup/SPIRV-Tools/issues/1569#issuecomment-390250792
target_link_libraries(spirv_codegen PRIVATE SPIRV-Tools-opt ${SPIRV_TOOLS})
