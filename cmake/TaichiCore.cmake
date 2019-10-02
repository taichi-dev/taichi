set(CORE_LIBRARY_NAME taichi_core)

include(cmake/PythonNumpyPybind11.cmake)

file(GLOB TAICHI_CORE_SOURCE
        "src/*/*/*/*.cpp" "src/*/*/*.cpp" "src/*/*.cpp" "src/*.cpp"
        "src/*/*/*/*.h" "src/*/*/*.h" "src/*/*.h" "src/*.h"
        "include/taichi/*/*/*/*.cpp" "include/taichi/*/*/*.cpp" "include/taichi/*/*.cpp"
        "include/taichi/*/*/*/*.h" "include/taichi/*/*/*.h" "include/taichi/*/*.h")


add_library(${CORE_LIBRARY_NAME} SHARED ${TAICHI_CORE_SOURCE})

# Optional dependencies

if (APPLE)
    target_link_libraries(${CORE_LIBRARY_NAME} "-framework Cocoa")
endif ()

if (NOT WIN32)
    target_link_libraries(${CORE_LIBRARY_NAME} pthread stdc++)
    if (APPLE)
        # OS X
    else()
        # Linux
        target_link_libraries(${CORE_LIBRARY_NAME} stdc++fs X11)
    endif()
endif ()
message("PYTHON_LIBRARIES" ${PYTHON_LIBRARIES})
target_link_libraries(${CORE_LIBRARY_NAME} ${PYTHON_LIBRARIES})

foreach (source IN LISTS TAICHI_CORE_SOURCE)
    file(RELATIVE_PATH source_rel ${CMAKE_CURRENT_LIST_DIR} ${source})
    get_filename_component(source_path "${source_rel}" PATH)
    string(REPLACE "/" "\\" source_path_msvc "${source_path}")
    source_group("${source_path_msvc}" FILES "${source}")
endforeach ()

if (MSVC)
    set_property(TARGET ${CORE_LIBRARY_NAME} APPEND PROPERTY LINK_FLAGS /DEBUG)
endif ()

if (WIN32)
    set_target_properties(${CORE_LIBRARY_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY
            "${CMAKE_CURRENT_SOURCE_DIR}/runtimes")
endif ()

include_directories(include)
include_directories(external/include)

#add_custom_target(
#        clangformat
#        COMMAND clang-format-6.0
#        -style=file
#        -i
#        ${TAICHI_CORE_SOURCE} ${TAICHI_PROJECT_SOURCE}
#)
#
#add_custom_target(
#        yapfformat
#        COMMAND yapf
#        -irp
#        ${CMAKE_CURRENT_LIST_DIR}/../
#)
