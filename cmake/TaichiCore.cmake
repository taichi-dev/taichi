set(CORE_LIBRARY_NAME taichi_core)

include(cmake/PythonNumpyPybind11.cmake)

file(GLOB TAICHI_CORE_SOURCE
        "src/*/*/*/*.cpp" "src/*/*/*.cpp" "src/*/*.cpp" "src/*.cpp"
        "src/*/*/*/*.h" "src/*/*/*.h" "src/*/*.h" "src/*.h"
        "include/taichi/*/*/*/*.cpp" "include/taichi/*/*/*.cpp" "include/taichi/*/*.cpp"
        "include/taichi/*/*/*/*.h" "include/taichi/*/*/*.h" "include/taichi/*/*.h")

file(GLOB TAICHI_PROJECT_SOURCE
        "projects/*/*/*/*.cpp"
        "projects/*/*/*/*.h"
        "projects/*/*/*.cpp"
        "projects/*/*/*.h"
        "projects/*/*.cpp"
        "projects/*/*.h"
        "projects/*.cpp"
        "projects/*.h"
        )

add_library(${CORE_LIBRARY_NAME} SHARED ${TAICHI_CORE_SOURCE})

# Optional dependencies

if (USE_OPENGL)
    #add_subdirectory(external/glfw)
    #include_directories(external/glfw/include)
    #target_link_libraries(${CORE_LIBRARY_NAME} glfw)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTC_USE_OPENGL")
    find_package(OpenGL REQUIRED)
    find_package(GLFW3 REQUIRED)
    include_directories(${GLFW_INCLUDE_DIRS})
    find_package(GLEW REQUIRED)
    include_directories(${GLEW_INCLUDE_DIRS})
    target_link_libraries(${CORE_LIBRARY_NAME} ${GLEW_LIBRARY})
    target_link_libraries(${CORE_LIBRARY_NAME} ${GLEW_LIBRARIES})
    target_link_libraries(${CORE_LIBRARY_NAME} ${OPENGL_LIBRARIES})
    target_link_libraries(${CORE_LIBRARY_NAME} ${GLFW3_LIBRARY})
    if (APPLE)
        target_link_libraries(${CORE_LIBRARY_NAME} glfw3)
    endif ()
endif ()

if (NOT WIN32)
    target_link_libraries(${CORE_LIBRARY_NAME} pthread stdc++)
    if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/external/lib)
        message("Runtimes exists.")
    else()
        message("Fetching runtimes..")
        execute_process(COMMAND git clone https://github.com/yuanming-hu/taichi_runtime ${CMAKE_CURRENT_SOURCE_DIR}/external/lib)
    endif()
    if (APPLE)
        target_link_libraries(${CORE_LIBRARY_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/external/lib/libembree.2.dylib)
        target_link_libraries(${CORE_LIBRARY_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/external/lib/libtbb.dylib)
        target_link_libraries(${CORE_LIBRARY_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/external/lib/libtbbmalloc.dylib)
    else()
        target_link_libraries(${CORE_LIBRARY_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/external/lib/libembree.so.2)
        target_link_libraries(${CORE_LIBRARY_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/external/lib/libtbb.so.2)
        target_link_libraries(${CORE_LIBRARY_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/external/lib/libtbbmalloc.so.2)
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
    target_link_libraries(${CORE_LIBRARY_NAME}
      ${CMAKE_CURRENT_SOURCE_DIR}/external/lib/embree.lib
      ${CMAKE_CURRENT_SOURCE_DIR}/external/lib/tbb.lib
      ${CMAKE_CURRENT_SOURCE_DIR}/external/lib/tbbmalloc.lib
    )
endif ()

include_directories(include)
include_directories(external/include)

add_custom_target(
        clangformat
        COMMAND clang-format-4.0
        -style=file
        -i
        ${TAICHI_CORE_SOURCE} ${TAICHI_PROJECT_SOURCE}
)

add_custom_target(
        yapfformat
        COMMAND yapf
        -irp
        ${CMAKE_CURRENT_LIST_DIR}/../
)
