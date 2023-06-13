# ./taichi/exports/CMakeLists.txt

set(TAICHI_CORE_EXPORTS_NAME taichi_core_exports)
add_library(${TAICHI_CORE_EXPORTS_NAME} SHARED)
target_sources(${TAICHI_CORE_EXPORTS_NAME}
  PRIVATE
    ${PROJECT_SOURCE_DIR}/taichi/exports/export_lang.cpp
  )

target_include_directories(${TAICHI_CORE_EXPORTS_NAME}
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/external/spdlog/include
    ${PROJECT_SOURCE_DIR}/external/eigen
    ${LLVM_INCLUDE_DIRS} # For "llvm/ADT/SmallVector.h" included in ir.h
  )

target_link_libraries(${TAICHI_CORE_EXPORTS_NAME} PRIVATE taichi_core)

set(CORE_EXPORTS_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/build")
set_target_properties(${TAICHI_CORE_EXPORTS_NAME} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CORE_EXPORTS_OUTPUT_DIRECTORY}
    ARCHIVE_OUTPUT_DIRECTORY ${CORE_EXPORTS_OUTPUT_DIRECTORY})

if (${CMAKE_GENERATOR} MATCHES "^Visual Studio")
  # Visual Studio is a multi-config generator, which appends ${CMAKE_BUILD_TYPE} to the output folder
  add_custom_command(
        TARGET ${TAICHI_CORE_EXPORTS_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CORE_EXPORTS_OUTPUT_DIRECTORY}/${CMAKE_BUILD_TYPE}/${TAICHI_CORE_EXPORTS_NAME}.dll
                ${CORE_EXPORTS_OUTPUT_DIRECTORY}/${TAICHI_CORE_EXPORTS_NAME}.dll)
elseif (${CMAKE_GENERATOR} STREQUAL "XCode")
  # XCode is also a multi-config generator
  add_custom_command(
        TARGET ${TAICHI_CORE_EXPORTS_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CORE_EXPORTS_OUTPUT_DIRECTORY}/${CMAKE_BUILD_TYPE}/lib${TAICHI_CORE_EXPORTS_NAME}.dylib
                ${CORE_EXPORTS_OUTPUT_DIRECTORY}/lib${TAICHI_CORE_EXPORTS_NAME}.dylib)
endif()

function(install_taichi_core_exports INSTALL_NAME TAICHI_CORE_EXPORTS_DIR)

  # This is the `CMAKE_INSTALL_PREFIX` from command line.
  set(CMAKE_INSTALL_PREFIX_BACKUP ${CMAKE_INSTALL_PREFIX})
  # This thing is read by `install(EXPORT ...)` to generate `_IMPORT_PREFIX` in
  # `TaichiTargets.cmake`. Replace the original value to avoid the absolute
  # path.
  set(CMAKE_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX_BACKUP}/${TAICHI_CORE_EXPORTS_DIR})

  message("Installing to ${CMAKE_INSTALL_PREFIX}")

  install(TARGETS ${TAICHI_CORE_EXPORTS_NAME} EXPORT TaichiExportTargets${INSTALL_NAME}
      LIBRARY DESTINATION ${TAICHI_CORE_EXPORTS_DIR}
      # ARCHIVE DESTINATION ${TAICHI_CORE_EXPORTS_DIR}
      RUNTIME DESTINATION ${TAICHI_CORE_EXPORTS_DIR}
      # PUBLIC_HEADER DESTINATION ${TAICHI_CORE_EXPORTS_DIR}/${CMAKE_INSTALL_INCLUDEDIR}/taichi/core
      )

  # Recover the original value in case it's used by other targets.
  set(CMAKE_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX_BACKUP})
endfunction()

if (TI_WITH_PYTHON)
  install_taichi_core_exports(PyTaichi python/taichi/_lib/core_exports)
endif()
