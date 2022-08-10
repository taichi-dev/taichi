# ./taichi/ui/backends/vulkan/CMakeLists.txt

add_library(taichi_ui_vulkan OBJECT)
target_sources(taichi_ui_vulkan
  PRIVATE
    app_context.cpp
    canvas.cpp
    gui.cpp
    renderable.cpp
    renderer.cpp
    swap_chain.cpp
    vertex.cpp
    renderables/circles.cpp
    renderables/lines.cpp
    renderables/mesh.cpp
    renderables/particles.cpp
    renderables/scene_lines.cpp
    renderables/set_image.cpp
    renderables/triangles.cpp
  )

if(NOT ANDROID)
  target_sources(taichi_ui_vulkan PRIVATE window.cpp)
endif()

target_include_directories(taichi_ui_vulkan
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/external/glm
    ${PROJECT_SOURCE_DIR}/external/eigen
    ${PROJECT_SOURCE_DIR}/external/volk
    ${PROJECT_SOURCE_DIR}/external/VulkanMemoryAllocator/include
    ${PROJECT_SOURCE_DIR}/external/imgui
    ${PROJECT_SOURCE_DIR}/external/imgui/backends
    ${PROJECT_SOURCE_DIR}/external/Vulkan-Headers/include
    ${LLVM_INCLUDE_DIRS}
  PUBLIC
    ${PROJECT_SOURCE_DIR}/external/glfw/include
  )

target_link_libraries(taichi_ui_vulkan PRIVATE taichi_core)
target_link_libraries(taichi_ui_vulkan PRIVATE taichi_common)
target_link_libraries(taichi_ui_vulkan PRIVATE imgui)

if(TI_WITH_VULKAN)
  target_link_libraries(taichi_ui_vulkan PUBLIC vulkan_rhi)
endif()
