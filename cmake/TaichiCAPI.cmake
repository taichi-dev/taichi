cmake_minimum_required(VERSION 3.0)

set(TAICHI_C_API_NAME taichi_c_api)

file(GLOB_RECURSE C_API_SOURCE "c_api/src/taichi_core_impl.cpp")

if (TI_WITH_LLVM)
  list(APPEND C_API_SOURCE "c_api/src/taichi_llvm_impl.cpp")
endif()

if (TI_WITH_OPENGL OR TI_WITH_VULKAN)
  list(APPEND C_API_SOURCE "c_api/src/taichi_gfx_impl.cpp")
endif()

if (TI_WITH_OPENGL)
  list(APPEND C_API_SOURCE "c_api/src/taichi_opengl_impl.cpp")
endif()

if (TI_WITH_VULKAN)
  list(APPEND C_API_SOURCE "c_api/src/taichi_vulkan_impl.cpp")
endif()

if(TI_BUILD_TESTS)
  list(APPEND C_API_SOURCE "c_api/src/c_api_test_utils.cpp")
endif()

add_library(${TAICHI_C_API_NAME} SHARED ${C_API_SOURCE})
target_link_libraries(${TAICHI_C_API_NAME} PRIVATE taichi_core)

# [TODO] Remove the following two linkages after rewriting AOT Demos with Device APIS
if(TI_WITH_GGUI)
target_link_libraries(${TAICHI_C_API_NAME} PRIVATE taichi_ui_vulkan)
target_link_libraries(${TAICHI_C_API_NAME} PRIVATE taichi_ui)
endif()

set(C_API_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/build")
set_target_properties(${TAICHI_C_API_NAME} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${C_API_OUTPUT_DIRECTORY}
    ARCHIVE_OUTPUT_DIRECTORY ${C_API_OUTPUT_DIRECTORY})

target_include_directories(${TAICHI_C_API_NAME}
    PUBLIC
        # Used when building the library:
        $<BUILD_INTERFACE:${taichi_c_api_BINARY_DIR}/c_api/include>
        $<BUILD_INTERFACE:${taichi_c_api_SOURCE_DIR}/c_api/include>
        # Used when installing the library:
        $<INSTALL_INTERFACE:/c_api/include>
    PRIVATE
        # Used only when building the library:
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/c_api/include
        ${CMAKE_CURRENT_SOURCE_DIR}/external/spdlog/include
        ${CMAKE_CURRENT_SOURCE_DIR}/external/Vulkan-Headers/include
        ${CMAKE_CURRENT_SOURCE_DIR}/external/VulkanMemoryAllocator/include
        ${CMAKE_CURRENT_SOURCE_DIR}/external/SPIRV-Tools/include
        ${CMAKE_CURRENT_SOURCE_DIR}/external/volk
        ${CMAKE_CURRENT_SOURCE_DIR}/external/glad/include
        ${CMAKE_CURRENT_SOURCE_DIR}/external/glfw/include
    )

# This helper provides us standard locations across Linux/Windows/MacOS
include(GNUInstallDirs)

install(TARGETS ${TAICHI_C_API_NAME} EXPORT ${TAICHI_C_API_NAME}Targets
    LIBRARY DESTINATION c_api/lib
    ARCHIVE DESTINATION c_api/lib
    RUNTIME DESTINATION c_api/bin
    INCLUDES DESTINATION c_api/include
    )

# Install the export set, which contains the meta data of the target
install(EXPORT ${TAICHI_C_API_NAME}Targets
    FILE ${TAICHI_C_API_NAME}Targets.cmake
    NAMESPACE ${TAICHI_C_API_NAME}::
    DESTINATION c_api/${CMAKE_INSTALL_LIBDIR}/cmake/${TAICHI_C_API_NAME}
    )

include(CMakePackageConfigHelpers)

# Generate the config file
configure_package_config_file(
      "${PROJECT_SOURCE_DIR}/cmake/${TAICHI_C_API_NAME}Config.cmake.in"
      "${PROJECT_BINARY_DIR}/${TAICHI_C_API_NAME}Config.cmake"
    INSTALL_DESTINATION
       c_api/${CMAKE_INSTALL_LIBDIR}/cmake/${TAICHI_C_API_NAME}
    )

# Generate the config version file
set(${TAICHI_C_API_NAME}_VERSION "${TI_VERSION_MAJOR}.${TI_VERSION_MINOR}.${TI_VERSION_PATCH}")
write_basic_package_version_file(
    "${TAICHI_C_API_NAME}ConfigVersion.cmake"
    VERSION ${${TAICHI_C_API_NAME}_VERSION}
    COMPATIBILITY SameMajorVersion
    )

# Install the config files
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/${TAICHI_C_API_NAME}Config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/${TAICHI_C_API_NAME}ConfigVersion.cmake"
    DESTINATION
    c_api/${CMAKE_INSTALL_LIBDIR}/cmake/${TAICHI_C_API_NAME}
    )

# Install public headers for this target
# TODO: Replace files here with public headers when ready.
install(DIRECTORY
      ${PROJECT_SOURCE_DIR}/c_api/include
    DESTINATION c_api
    FILES_MATCHING
    PATTERN *.h
    PATTERN *.hpp
    )
