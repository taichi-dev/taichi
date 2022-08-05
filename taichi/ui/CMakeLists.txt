# ./taichi/ui/CMakeLists.txt

add_library(taichi_ui)
target_sources(taichi_ui
  PRIVATE
    gui/gui.cpp
    gui/android.cpp
    gui/cocoa.cpp
    gui/win32.cpp
    gui/x11.cpp
  )

if(NOT ANDROID)
  target_sources(taichi_ui PRIVATE common/window_base.cpp)
endif()

target_include_directories(taichi_ui
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/external/glfw/include
    ${PROJECT_SOURCE_DIR}/external/eigen
    ${LLVM_INCLUDE_DIRS}
  PUBLIC
    ${PROJECT_SOURCE_DIR}/external/glm
  )

if (TI_WITH_VULKAN)
    add_subdirectory(backends/vulkan)
    target_link_libraries(taichi_ui PRIVATE taichi_ui_vulkan)
else()
    message(STATUS "UI is not compiled with Vulkan on")
endif ()

target_link_libraries(taichi_ui PRIVATE taichi_common)
