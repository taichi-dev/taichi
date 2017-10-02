# Python, numpy, and pybind11

if (PYTHON_EXECUTABLE)
    message("Using ${PYTHON_EXECUTABLE} as python executable.")
else ()
    message("Using 'python3' as python interpreter.")
    set(PYTHON_EXECUTABLE python3)
endif ()


if (WIN32)
    find_package(PythonLibs 3.5 REQUIRED)
else ()
    execute_process(COMMAND ${PYTHON_EXECUTABLE} --version)
    execute_process(COMMAND ${PYTHON_EXECUTABLE} -c
            "import sys;\
            from distutils import sysconfig;\
            sys.stdout.write(\
            (sysconfig.get_config_var('INCLUDEDIR') + '/python3.5/'\
            if sysconfig.get_config_var('INCLUDEDIR') is not None else None)\
            or sysconfig.get_python_inc())"
            OUTPUT_VARIABLE PYTHON_INCLUDE_DIRS)
    execute_process(COMMAND ${PYTHON_EXECUTABLE} -c
            "import sys;\
            from distutils import sysconfig;\
            sys.stdout.write(sysconfig.get_config_var('LIBDIR') or sysconfig.get_python_lib())"
            OUTPUT_VARIABLE PYTHON_LIBRARY_DIR)

    find_library(PYTHON_LIBRARY NAMES python3.5 python3.5m PATHS ${PYTHON_LIBRARY_DIR}
            NO_DEFAULT_PATH NO_SYSTEM_ENVIRONMENT_PATH PATH_SUFFIXES x86_64-linux-gnu)
    set(PYTHON_LIBRARIES ${PYTHON_LIBRARY})
endif ()

include_directories(${PYTHON_INCLUDE_DIRS})
message("    include: ${PYTHON_INCLUDE_DIRS}")
message("    library: ${PYTHON_LIBRARIES}")

execute_process(COMMAND ${PYTHON_EXECUTABLE} -c
        "import numpy.distutils, sys;\
        sys.stdout.write(':'.join(numpy.distutils.misc_util.get_numpy_include_dirs()))"
        OUTPUT_VARIABLE PYTHON_NUMPY_INCLUDE_DIR)

message("    numpy include: ${PYTHON_NUMPY_INCLUDE_DIR}")
include_directories(${PYTHON_NUMPY_INCLUDE_DIR})

execute_process(COMMAND ${PYTHON_EXECUTABLE} -c
        "import sys; import pybind11; sys.stdout.write(pybind11.get_include())"
        OUTPUT_VARIABLE PYBIND11_INCLUDE_DIR
        RESULT_VARIABLE PYBIND11_IMPORT_RET)
if (NOT PYBIND11_IMPORT_RET)
    # returns zero if success
    message("    pybind11 include: " ${PYBIND11_INCLUDE_DIR})
else ()
    message(FATAL_ERROR "Can not import pybind11. Please install. ([sudo] pip install pybind11)")
endif ()

include_directories(${PYBIND11_INCLUDE_DIR})
