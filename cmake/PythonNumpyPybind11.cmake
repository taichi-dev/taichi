# Python, numpy, and pybind11
find_package(Python REQUIRED COMPONENTS Interpreter Development NumPy)
set(pybind11_DIR ${PYBIND11_CMAKE_DIR})
find_package(pybind11 CONFIG REQUIRED)

message("Using ${Python_EXECUTABLE} as the interpreter")
message("    version: ${Python_VERSION}")
message("    include: ${Python_INCLUDE_DIRS}")
message("    library: ${Python_LIBRARIES}")
message("    numpy include: ${Python_NumPy_INCLUDE_DIRS}")
