# Python, numpy, and pybind11
execute_process(COMMAND ${PYTHON_EXECUTABLE} -m pybind11 --cmake
                OUTPUT_VARIABLE pybind11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import numpy;print(numpy.get_include())"
                OUTPUT_VARIABLE NUMPY_INCLUDE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)





execute_process(COMMAND ${PYTHON_EXECUTABLE} -c
        "import sys;\
        from distutils import sysconfig;\
        sys.stdout.write(\
        (sysconfig.get_config_var('INCLUDEPY')\
        if sysconfig.get_config_var('INCLUDEDIR') is not None else None)\
        or sysconfig.get_python_inc())"
        OUTPUT_VARIABLE PYTHON_INCLUDE_DIRS)
include_directories(${PYTHON_INCLUDE_DIRS})



execute_process(COMMAND ${PYTHON_EXECUTABLE} -c
        "import numpy.distutils, sys;\
        sys.stdout.write(':'.join(numpy.distutils.misc_util.get_numpy_include_dirs()))"
        OUTPUT_VARIABLE PYTHON_NUMPY_INCLUDE_DIR)

message("    numpy include: ${PYTHON_NUMPY_INCLUDE_DIR}")
include_directories(${PYTHON_NUMPY_INCLUDE_DIR})
#include_directories(${NUMPY_INCLUDE_DIR})


execute_process(COMMAND ${PYTHON_EXECUTABLE} -c
        "import sys; import pybind11; sys.stdout.write(pybind11.get_include() + ';' + pybind11.get_include(True))"
        OUTPUT_VARIABLE PYBIND11_INCLUDE_DIR
        RESULT_VARIABLE PYBIND11_IMPORT_RET)
if (NOT PYBIND11_IMPORT_RET)
    # returns zero if success
    message("  ======== old pybind11 include: ${PYBIND11_INCLUDE_DIR}")
else ()
    message(FATAL_ERROR "Cannot import pybind11. Please install. ([sudo] pip3 install --user pybind11)")
endif ()

include_directories(${PYBIND11_INCLUDE_DIR})



message("-- Python: Using ${PYTHON_EXECUTABLE} as the interpreter")
#message("    version: ${PYTHON_VERSION_STRING}")
message("    include: ${PYTHON_INCLUDE_DIR}")
message("    library: ${PYTHON_LIBRARY}")
message("    numpy include: ${NUMPY_INCLUDE_DIR}")
message("    pybind11 include: ${pybind11_DIR}")

find_package(pybind11 CONFIG REQUIRED)
