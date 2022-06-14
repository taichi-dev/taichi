cmake_minimum_required(VERSION 3.0)

set(TAICHI_EXPORT_CORE_NAME taichi_export_core)

add_library(${TAICHI_EXPORT_CORE_NAME} SHARED)
target_link_libraries(${TAICHI_EXPORT_CORE_NAME} PRIVATE taichi_isolated_core)

set_target_properties(${TAICHI_EXPORT_CORE_NAME} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_INSTALL_LIBDIR}"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_INSTALL_LIBDIR}")

target_include_directories(${TAICHI_EXPORT_CORE_NAME}
    PUBLIC
        # Used when building the library:
        $<BUILD_INTERFACE:${taichi_export_core_BINARY_DIR}/include>
        $<BUILD_INTERFACE:${taichi_export_core_SOURCE_DIR}/include>
        # Used when installing the library:
        $<INSTALL_INTERFACE:include>
    PRIVATE
        # Used only when building the library:
        ${PROJECT_SOURCE_DIR})

include(GNUInstallDirs)
install(TARGETS ${TAICHI_EXPORT_CORE_NAME} EXPORT ${TAICHI_EXPORT_CORE_NAME}Targets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
  )
# Install the export set
install(EXPORT ${TAICHI_EXPORT_CORE_NAME}Targets
  FILE ${TAICHI_EXPORT_CORE_NAME}Targets.cmake
  NAMESPACE ${TAICHI_EXPORT_CORE_NAME}::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${TAICHI_EXPORT_CORE_NAME}
  )

# For generating the ${TAICHI_EXPORT_CORE_NAME}Config*.cmake
include(CMakePackageConfigHelpers)
configure_package_config_file(
    "${PROJECT_SOURCE_DIR}/cmake/${TAICHI_EXPORT_CORE_NAME}Config.cmake.in"
    "${PROJECT_BINARY_DIR}/${TAICHI_EXPORT_CORE_NAME}Config.cmake"
  INSTALL_DESTINATION
     ${CMAKE_INSTALL_LIBDIR}/cmake/${TAICHI_EXPORT_CORE_NAME}
  )

set(${TAICHI_EXPORT_CORE_NAME}_VERSION "${TI_VERSION_MAJOR}.${TI_VERSION_MINOR}.${TI_VERSION_PATCH}")
write_basic_package_version_file(
  "${TAICHI_EXPORT_CORE_NAME}ConfigVersion.cmake"
  VERSION ${${TAICHI_EXPORT_CORE_NAME}_VERSION}
  COMPATIBILITY SameMajorVersion
  )

# Install the meta data of targets, namely the export set
install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/${TAICHI_EXPORT_CORE_NAME}Config.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/${TAICHI_EXPORT_CORE_NAME}ConfigVersion.cmake"
  DESTINATION
  ${CMAKE_INSTALL_LIBDIR}/cmake/${TAICHI_EXPORT_CORE_NAME}
  )

## Install headers
install(DIRECTORY
    ${PROJECT_SOURCE_DIR}/taichi
    ${PROJECT_SOURCE_DIR}/external/spdlog/include/spdlog
    ${PROJECT_SOURCE_DIR}/external/Vulkan-Headers/include/vulkan
    ${PROJECT_SOURCE_DIR}/external/Vulkan-Headers/include/vulkan
    ${PROJECT_SOURCE_DIR}/external/glm
    ${PROJECT_SOURCE_DIR}/external/eigen
  DESTINATION include
  FILES_MATCHING
  PATTERN *.h
  PATTERN *.hpp
)
install(DIRECTORY
    ${PROJECT_SOURCE_DIR}/external/eigen/Eigen
  DESTINATION include
)

file(GLOB TAICHI_PUBLIC_HEADERS
	"external/volk/*.h" "external/volk/*.hpp"
	"external/imgui/*.h" "external/imgui/*.hpp"
	"external/imgui/backends/*.h" "external/imgui/backends/*.hpp"
	"external/VulkanMemoryAllocator/include/*.h"
    )
install(FILES
    ${TAICHI_PUBLIC_HEADERS}
  DESTINATION include
)
