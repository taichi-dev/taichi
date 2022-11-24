macro(make_sample executable_name src_file)
add_executable(${executable_name})
set_property(TARGET ${executable_name} PROPERTY CXX_STANDARD 20)
set_property(TARGET ${executable_name} PROPERTY C_STANDARD 17)
target_sources(${executable_name} PRIVATE ${src_file} "common.h")
target_include_directories(${executable_name}
PRIVATE
  ${PROJECT_SOURCE_DIR}

  ${PROJECT_SOURCE_DIR}/external/SPIRV-Tools/include
  ${PROJECT_SOURCE_DIR}/external/eigen
  ${PROJECT_SOURCE_DIR}/external/FP16/include
  ${PROJECT_SOURCE_DIR}/external/SPIRV-Reflect
  ${PROJECT_SOURCE_DIR}/external/spdlog/include
  ${LLVM_INCLUDE_DIRS}

  ${PROJECT_SOURCE_DIR}/external/volk
  ${PROJECT_SOURCE_DIR}/external/Vulkan-Headers/include
  ${PROJECT_SOURCE_DIR}/external/glfw/include
  ${PROJECT_SOURCE_DIR}/external/glm
  ${PROJECT_SOURCE_DIR}/external/imgui
)
target_include_directories(${executable_name} SYSTEM
  PUBLIC
    ${PROJECT_SOURCE_DIR}/external/VulkanMemoryAllocator/include
  )
target_link_libraries(${executable_name} taichi_c_api glfw)
endmacro()

make_sample(sample_1_window sample_1_window.cpp)
make_sample(sample_2_triangle sample_2_triangle.cpp)
