# ./taichi/exports/CMakeLists.txt

# TODO&NOTE(PGZXB): The taichi_core_exports should be a shared library to be
# dynamically loaded by the Python module. But we cannot do this now, because
# the C API implementation is still in progress. There are many unfinished APIs
# that still need to be called from taichi_python, so we need to load both
# taichi_python and taichi_core_exports, which will cause the global state of
# taichi to be inconsistent. To solve this problem, we need to implement all C
# APIs in taichi_core_exports, and then gradually migrate the APIs in
# taichi_python to taichi_core_exports, and finally delete taichi_python. But
# this process will take a long time, so we cannot do it now. The current
# approach is to include taichi_core_exports as an object library into
# taichi_python, which will be dynamically loaded. When all C APIs are
# implemented, taichi_core_exports will be separated.

set(TAICHI_CORE_EXPORTS_NAME taichi_core_exports)
add_library(${TAICHI_CORE_EXPORTS_NAME} OBJECT)
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

function(generate_py_module_from_exports_h exports_header output_dir)
    # Rerun the script if the exports header or the script itself is changed.
    add_custom_command(
        OUTPUT ${output_dir}/__init__.py
        # Command: python misc/exports_to_py.py \
        #              --exports-header taichi/exports/exports.h \
        #              --cpp-path ${CMAKE_C_COMPILER} \
        #              --cpp-args "['-E', '-DTI_EXPORTS_TO_PY', '-Iexternal/pycparser/utils/fake_libc_include']" \
        #              --output-dir python/taichi/_lib/exports \
        #              --verbose 2
        COMMAND ${PYTHON_EXECUTABLE} ${PROJECT_SOURCE_DIR}/misc/exports_to_py.py
            --exports-header ${exports_header}
            --cpp-path ${CMAKE_C_COMPILER}
            --cpp-args "['-E', '-DTI_EXPORTS_TO_PY', '-I${PROJECT_SOURCE_DIR}/external/pycparser/utils/fake_libc_include']"
            --output-dir ${output_dir}
            --verbose 2
        DEPENDS ${PROJECT_SOURCE_DIR}/misc/exports_to_py.py ${exports_header}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
    add_custom_target(
        "taichi_generate_py_module_from_exports_h"
        ALL
        DEPENDS ${output_dir}/__init__.py
    )
endfunction()

if (TI_WITH_PYTHON)
  generate_py_module_from_exports_h(${PROJECT_SOURCE_DIR}/taichi/exports/exports.h ${PROJECT_SOURCE_DIR}/python/taichi/_lib/exports)
endif()
