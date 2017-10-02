add_executable(taichi include/taichi/main.cpp)
target_link_libraries(taichi ${CORE_LIBRARY_NAME})
if (WIN32)
    set_target_properties(taichi PROPERTIES RUNTIME_OUTPUT_DIRECTORY
            "${CMAKE_CURRENT_SOURCE_DIR}/runtimes")
endif ()
