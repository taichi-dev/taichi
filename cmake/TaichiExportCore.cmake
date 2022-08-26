cmake_minimum_required(VERSION 3.0)

set(TAICHI_EXPORT_CORE_NAME taichi_export_core)

message(WARNING "You are trying to build the taichi_export_core target, support for this target will be deprecated in the future, please considering using the taichi_c_api target.")

add_library(${TAICHI_EXPORT_CORE_NAME} SHARED)
target_link_libraries(${TAICHI_EXPORT_CORE_NAME} PRIVATE taichi_core)
set_target_properties(${TAICHI_EXPORT_CORE_NAME} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/build"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/build")

# [TODO] Remove the following two linkages after rewriting AOT Demos with Device APIS
if(TI_WITH_GGUI)
    target_link_libraries(${TAICHI_EXPORT_CORE_NAME} PRIVATE taichi_ui_vulkan)
    target_link_libraries(${TAICHI_EXPORT_CORE_NAME} PRIVATE taichi_ui)
endif()

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

# This helper provides us standard locations across Linux/Windows/MacOS
include(GNUInstallDirs)

install(TARGETS ${TAICHI_EXPORT_CORE_NAME} EXPORT ${TAICHI_EXPORT_CORE_NAME}Targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
    )

# Install the export set, which contains the meta data of the target
install(EXPORT ${TAICHI_EXPORT_CORE_NAME}Targets
    FILE ${TAICHI_EXPORT_CORE_NAME}Targets.cmake
    NAMESPACE ${TAICHI_EXPORT_CORE_NAME}::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${TAICHI_EXPORT_CORE_NAME}
    )

include(CMakePackageConfigHelpers)

# Generate the config file
configure_package_config_file(
      "${PROJECT_SOURCE_DIR}/cmake/${TAICHI_EXPORT_CORE_NAME}Config.cmake.in"
      "${PROJECT_BINARY_DIR}/${TAICHI_EXPORT_CORE_NAME}Config.cmake"
    INSTALL_DESTINATION
       ${CMAKE_INSTALL_LIBDIR}/cmake/${TAICHI_EXPORT_CORE_NAME}
    )

# Generate the config version file
set(${TAICHI_EXPORT_CORE_NAME}_VERSION "${TI_VERSION_MAJOR}.${TI_VERSION_MINOR}.${TI_VERSION_PATCH}")
write_basic_package_version_file(
    "${TAICHI_EXPORT_CORE_NAME}ConfigVersion.cmake"
    VERSION ${${TAICHI_EXPORT_CORE_NAME}_VERSION}
    COMPATIBILITY SameMajorVersion
    )

# Install the config files
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/${TAICHI_EXPORT_CORE_NAME}Config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/${TAICHI_EXPORT_CORE_NAME}ConfigVersion.cmake"
    DESTINATION
    ${CMAKE_INSTALL_LIBDIR}/cmake/${TAICHI_EXPORT_CORE_NAME}
    )

# Install public headers for this target
# TODO: Replace files here with public headers when ready.
install(DIRECTORY
      ${PROJECT_SOURCE_DIR}/taichi
      ${PROJECT_SOURCE_DIR}/external/spdlog/include/spdlog
      ${PROJECT_SOURCE_DIR}/external/Vulkan-Headers/include/vulkan
      ${PROJECT_SOURCE_DIR}/external/Vulkan-Headers/include/vulkan
      ${PROJECT_SOURCE_DIR}/external/glm/glm
      ${PROJECT_SOURCE_DIR}/external/eigen
    DESTINATION include
    FILES_MATCHING
    PATTERN *.h
    PATTERN *.hpp
    PATTERN *.inl
    )
install(DIRECTORY
      ${PROJECT_SOURCE_DIR}/external/eigen/Eigen
    DESTINATION include
    )
file(GLOB TAICHI_PUBLIC_HEADERS_TEMP
    "external/volk/*.h" "external/volk/*.hpp"
    "external/imgui/*.h" "external/imgui/*.hpp"
    "external/imgui/backends/*.h" "external/imgui/backends/*.hpp"
    "external/VulkanMemoryAllocator/include/*.h"
    )
install(FILES
      ${TAICHI_PUBLIC_HEADERS_TEMP}
    DESTINATION include
    )
