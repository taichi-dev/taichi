cmake_minimum_required(VERSION 3.0)

# This function creates a static target from OBJECT_TARGET, then link TARGET with the static target
#
# For now, we have to keep this hack because:
# 1. Existence of circular dependencies in Taichi repo (https://github.com/taichi-dev/taichi/issues/6838)
# 2. Link order restriction from `ld` linker on Linux (https://stackoverflow.com/questions/45135/why-does-the-order-in-which-libraries-are-linked-sometimes-cause-errors-in-gcc), which has zero-tolerance w.r.t circular dependencies.
function(target_link_static_library TARGET OBJECT_TARGET)

    set(STATIC_TARGET "${OBJECT_TARGET}_static")
    add_library(${STATIC_TARGET})
    target_link_libraries(${STATIC_TARGET} PUBLIC ${OBJECT_TARGET})
if(LINUX)
    get_target_property(LINK_LIBS ${OBJECT_TARGET} LINK_LIBRARIES)
    target_link_libraries(${TARGET} PRIVATE "-Wl,--start-group" "${STATIC_TARGET}" "${LINK_LIBS}" "-Wl,--end-group")
else()
    target_link_libraries(${TARGET} PRIVATE "${STATIC_TARGET}")
endif()

endfunction()

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
  if (APPLE)
    install(FILES ${MoltenVK_LIBRARY} DESTINATION c_api/lib)
  endif()
endif()

if(TI_BUILD_TESTS)
  list(APPEND C_API_SOURCE "c_api/src/c_api_test_utils.cpp")
endif()

add_library(${TAICHI_C_API_NAME} SHARED ${C_API_SOURCE})
if (${CMAKE_GENERATOR} STREQUAL "Xcode")
  target_link_libraries(${TAICHI_C_API_NAME} PRIVATE taichi_core)
  message(WARNING "Static wrapping does not work on Xcode, using object linking instead.")
else()
  target_link_static_library(${TAICHI_C_API_NAME} taichi_core)
endif()
target_enable_function_level_linking(${TAICHI_C_API_NAME})

# Strip shared library
set_target_properties(${TAICHI_C_API_NAME} PROPERTIES LINK_FLAGS_RELEASE -s)

# Avoid exporting third party symbols from libtaichi_c_api.so
# Note that on Windows, external symbols will be excluded from .dll automatically, by default.
if(LINUX)
    target_link_options(${TAICHI_C_API_NAME} PRIVATE -Wl,--version-script,${CMAKE_CURRENT_SOURCE_DIR}/c_api/version_scripts/export_symbols_linux.lds)
elseif(APPLE)
    # Unfortunately, ld on MacOS does not support --exclude-libs and we have to manually specify the exported symbols
    target_link_options(${TAICHI_C_API_NAME} PRIVATE -Wl,-exported_symbols_list,${CMAKE_CURRENT_SOURCE_DIR}/c_api/version_scripts/export_symbols_mac.lds)
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

if(TI_WITH_LLVM)
# Install runtime .bc files for LLVM backend
install(DIRECTORY
      ${INSTALL_LIB_DIR}/runtime
      DESTINATION c_api)
endif()
