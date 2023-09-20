# ./taichi/ui/ggui/CMakeLists.txt

add_library(taichi_ui_ggui OBJECT)
target_sources(taichi_ui_ggui
  PRIVATE
    app_context.cpp
    canvas.cpp
    scene.cpp
    sceneV2.cpp
    gui.cpp
    renderable.cpp
    renderer.cpp
    swap_chain.cpp
    renderables/circles.cpp
    renderables/lines.cpp
    renderables/mesh.cpp
    renderables/particles.cpp
    renderables/scene_lines.cpp
    renderables/set_image.cpp
    renderables/triangles.cpp
  )

if(NOT ANDROID)
  target_sources(taichi_ui_ggui PRIVATE window.cpp)
endif()

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  target_sources(taichi_ui_ggui PRIVATE gui_metal.mm nswindow_adapter.mm)
endif()

target_include_directories(taichi_ui_ggui
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

target_link_libraries(taichi_ui_ggui PRIVATE taichi_common)
target_link_libraries(taichi_ui_ggui PRIVATE imgui)
target_link_libraries(taichi_ui_ggui PUBLIC ti_device_api)
if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  target_link_libraries(taichi_ui_ggui PRIVATE "-framework QuartzCore")
endif()
