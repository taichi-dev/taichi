# Python, numpy, and pybind11
execute_process(COMMAND ${PYTHON_EXECUTABLE} -m pybind11 --cmake
                OUTPUT_VARIABLE pybind11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import numpy;print(numpy.get_include())"
                OUTPUT_VARIABLE NUMPY_INCLUDE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)

message("-- Python: Using ${PYTHON_EXECUTABLE} as the interpreter")
message("    version: ${PYTHON_VERSION_STRING}")
message("    include: ${PYTHON_INCLUDE_DIR}")
message("    library: ${PYTHON_LIBRARY}")
message("    numpy include: ${NUMPY_INCLUDE_DIR}")

if (PYTHON_EXECUTABLE)
    message("Using ${PYTHON_EXECUTABLE} as python executable.")
else ()
    if (WIN32)
        message("Using 'python' as python interpreter.")
        set(PYTHON_EXECUTABLE python)
    else ()
        message("Using 'python3' as python interpreter.")
        set(PYTHON_EXECUTABLE python3)
    endif()
endif ()

if (WIN32)
    execute_process(COMMAND where ${PYTHON_EXECUTABLE}
        OUTPUT_VARIABLE PYTHON_EXECUTABLE_PATHS)
    if (${PYTHON_EXECUTABLE_PATHS})
        string(FIND ${PYTHON_EXECUTABLE_PATHS} "\n" _LINE_BREAK_LOC)
        string(SUBSTRING ${PYTHON_EXECUTABLE_PATHS} 0 ${_LINE_BREAK_LOC} PYTHON_EXECUTABLE_PATH)
    else ()
        set(PYTHON_EXECUTABLE_PATH ${PYTHON_EXECUTABLE})
    endif ()
else ()
    execute_process(COMMAND which ${PYTHON_EXECUTABLE}
            OUTPUT_VARIABLE PYTHON_EXECUTABLE_PATH)
endif()
execute_process(COMMAND ${PYTHON_EXECUTABLE} -c
        "import sys;\
        from distutils import sysconfig;\
        sys.stdout.write(sysconfig.get_python_version())"
        OUTPUT_VARIABLE PYTHON_VERSION)
execute_process(COMMAND ${PYTHON_EXECUTABLE} --version)
execute_process(COMMAND ${PYTHON_EXECUTABLE} -c
        "import sys;\
        from distutils import sysconfig;\
        sys.stdout.write(\
        (sysconfig.get_config_var('INCLUDEPY')\
        if sysconfig.get_config_var('INCLUDEDIR') is not None else None)\
        or sysconfig.get_python_inc())"
        OUTPUT_VARIABLE PYTHON_INCLUDE_DIRS)
execute_process(COMMAND ${PYTHON_EXECUTABLE} -c
        "import sys;\
        from distutils import sysconfig;\
        sys.stdout.write((sysconfig.get_config_var('LIBDIR') or sysconfig.get_python_lib()).replace('\\\\','/'))"
        OUTPUT_VARIABLE PYTHON_LIBRARY_DIR)


execute_process(COMMAND ${PYTHON_EXECUTABLE} -c
        "import sys;\
        sys.stdout.write(str(sys.version_info[1]))"
        OUTPUT_VARIABLE PYTHON_MINOR_VERSION)


if (WIN32)
  execute_process(COMMAND ${PYTHON_EXECUTABLE} -c
          "import sys;sys.stdout.write(sys.base_prefix.replace('\\\\', '/'))"
          OUTPUT_VARIABLE PYTHON_BASE_PREFIX)
  link_directories(${PYTHON_BASE_PREFIX}/libs)
  set(PYTHON_LIBRARIES ${PYTHON_BASE_PREFIX}/libs/python3.lib)
  set(PYTHON_LIBRARIES ${PYTHON_BASE_PREFIX}/libs/python3${PYTHON_MINOR_VERSION}.lib)
else()
  find_library(PYTHON_LIBRARY NAMES python${PYTHON_VERSION} python${PYTHON_VERSION}m PATHS ${PYTHON_LIBRARY_DIR}
          NO_DEFAULT_PATH NO_SYSTEM_ENVIRONMENT_PATH PATH_SUFFIXES x86_64-linux-gnu)
  set(PYTHON_LIBRARIES ${PYTHON_LIBRARY})
endif()


include_directories(${PYTHON_INCLUDE_DIRS})
message("    version: ${PYTHON_VERSION}")
message("    include: ${PYTHON_INCLUDE_DIRS}")
message("    library: ${PYTHON_LIBRARIES}")

execute_process(COMMAND ${PYTHON_EXECUTABLE} -c
        "import numpy.distutils, sys;\
        sys.stdout.write(':'.join(numpy.distutils.misc_util.get_numpy_include_dirs()))"
        OUTPUT_VARIABLE PYTHON_NUMPY_INCLUDE_DIR)

message("    numpy include: ${PYTHON_NUMPY_INCLUDE_DIR}")
include_directories(${PYTHON_NUMPY_INCLUDE_DIR})

execute_process(COMMAND ${PYTHON_EXECUTABLE} -c
        "import sys; import pybind11; sys.stdout.write(pybind11.get_include() + ';' + pybind11.get_include(True))"
        OUTPUT_VARIABLE PYBIND11_INCLUDE_DIR
        RESULT_VARIABLE PYBIND11_IMPORT_RET)
if (NOT PYBIND11_IMPORT_RET)
    # returns zero if success
    message("    pybind11 include: ${PYBIND11_INCLUDE_DIR}")
else ()
    message(FATAL_ERROR "Cannot import pybind11. Please install. ([sudo] pip3 install --user pybind11)")
endif ()

include_directories(${PYBIND11_INCLUDE_DIR})
