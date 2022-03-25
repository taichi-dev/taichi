# Python, numpy, and pybind11
# Currently, Scikit-build does not support FindPython, so we convert the
# provided hints ourselves.
if(SKBUILD)
  set(Python_EXECUTABLE "${PYTHON_EXECUTABLE}")
  set(Python_INCLUDE_DIR "${PYTHON_INCLUDE_DIR}")
  set(Python_LIBRARY "${PYTHON_LIBRARY}")
endif()

find_package(Python REQUIRED COMPONENTS Development.Module NumPy)
set(pybind11_DIR ${PYBIND11_CMAKE_DIR})
find_package(pybind11 CONFIG REQUIRED)

message("Using ${Python_EXECUTABLE} as the interpreter")
message("    version: ${Python_VERSION}")
message("    include: ${Python_INCLUDE_DIRS}")
message("    library: ${Python_LIBRARIES}")
message("    numpy include: ${Python_NumPy_INCLUDE_DIRS}")
