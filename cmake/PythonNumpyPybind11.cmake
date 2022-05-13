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

include_directories(${NUMPY_INCLUDE_DIR})

find_package(pybind11 CONFIG REQUIRED)
