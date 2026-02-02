# ./taichi/rhi/flagos/CMakeLists.txt

set(FLAGOS_RHI flagos_rhi)
add_library(${FLAGOS_RHI})

target_sources(${FLAGOS_RHI}
  PRIVATE
    flagos_device.cpp
  )

target_include_directories(${FLAGOS_RHI}
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/external/eigen
    ${PROJECT_SOURCE_DIR}/external/spdlog/include
  )

# FlagOS SDK integration
if(TI_WITH_FLAGOS)
  target_compile_definitions(${FLAGOS_RHI} PRIVATE TI_WITH_FLAGOS)

  # Find FlagOS SDK
  find_package(FlagOS QUIET)
  if(FlagOS_FOUND)
    target_link_libraries(${FLAGOS_RHI} PRIVATE FlagOS::FlagTree)
    target_include_directories(${FLAGOS_RHI} PRIVATE ${FlagOS_INCLUDE_DIRS})
  else()
    message(WARNING "FlagOS SDK not found, building with stub implementation")
  endif()
endif()
