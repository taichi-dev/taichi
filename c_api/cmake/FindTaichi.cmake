#[=======================================================================[.rst:
FindTaichi
----------

Finds the Taichi library.

The module first attempts to locate ``TaichiConfig.cmake`` in any Taichi
installation in CMake variable ``TAICHI_C_API_INSTALL_DIR`` or environment
variable of the same name. If the config file cannot be found, the libraries are
heuristically searched by names and paths in ``TAICHI_C_API_INSTALL_DIR``.

Imported Targets
^^^^^^^^^^^^^^^^

This module provides the following imported targets, if found:

``taichi::runtime``
  The Taichi Runtime library.

Result Variables
^^^^^^^^^^^^^^^^

This will define the following variables:

``taichi_FOUND``
  True if a Taichi installation is found.
``taichi_VERSION``
  Version of installed Taichi. Components might have lower version numbers.
``taichi_INCLUDE_DIRS``
  Paths to Include directories needed to use Taichi.
``taichi_LIBRARIES``
  Paths to Taichi linking libraries (``.libs``).
``taichi_REDIST_LIBRARIES``
  Paths to Taichi redistributed runtime libraries (``.so`` and ``.dll``). You
  might want to copy them next to your executables.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``taichi_runtime_VERSION``
  Taichi runtime library version.
``taichi_runtime_INCLUDE_DIR``
  The directory containing ``taichi/taichi.h``.
``taichi_runtime_LIBRARY``
  Path to linking library of ``taichi_runtime``.
``taichi_runtime_REDIST_LIBRARY``
  Path to redistributed runtime library of ``taichi_runtime``.

#]=======================================================================]

cmake_policy(PUSH)

# Support `IN_LIST` in CMake `if` command.
if(POLICY CMP0057)
    cmake_policy(SET CMP0057 NEW)
endif()

find_package(Python QUIET COMPONENTS Interpreter)

# Find TiRT in the installation directory. The installation directory is
# traditionally specified by an environment variable
# `TAICHI_C_API_INSTALL_DIR`.
if(NOT EXISTS "${TAICHI_C_API_INSTALL_DIR}")
    message("-- Looking for Taichi libraries via environment variable TAICHI_C_API_INSTALL_DIR")
    set(TAICHI_C_API_INSTALL_DIR $ENV{TAICHI_C_API_INSTALL_DIR})
endif()
# If the user didn't specity the environment variable, try find the C-API
# library in Python wheel.
if(NOT EXISTS "${TAICHI_C_API_INSTALL_DIR}" AND EXISTS "${Python_EXECUTABLE}")
    message("-- Looking for Taichi libraries via Python package installation")
    execute_process(COMMAND ${Python_EXECUTABLE} -c "import sys; import pathlib; print([pathlib.Path(x + '/../../../c_api').resolve() for x in sys.path if pathlib.Path(x + '/../../../c_api').exists()][0], end='')" OUTPUT_VARIABLE TAICHI_C_API_INSTALL_DIR)
endif()
message("-- TAICHI_C_API_INSTALL_DIR=${TAICHI_C_API_INSTALL_DIR}")
if(EXISTS "${TAICHI_C_API_INSTALL_DIR}")
    get_filename_component(TAICHI_C_API_INSTALL_DIR "${TAICHI_C_API_INSTALL_DIR}" ABSOLUTE)
    set(TAICHI_C_API_INSTALL_DIR "${TAICHI_C_API_INSTALL_DIR}" CACHE PATH "Root directory to Taichi installation")
else()
    message(WARNING "-- TAICHI_C_API_INSTALL_DIR doesn't point to a valid Taichi installation; configuration is very likely to fail")
endif()

# Set up default find components
if("${taichi_FIND_COMPONENTS}" STREQUAL "")
    # (penguinliong) Currently we only have Taichi Runtime. We might make the
    # codegen a library in the future?
    set(taichi_FIND_COMPONENTS "runtime")
endif()

message("-- Looking for Taichi components: ${taichi_FIND_COMPONENTS}")

# (penguinliong) Note that the config files only exposes libraries and their
# public headers. We still need to encapsulate the libraries into semantical
# CMake targets in this list. So please DO NOT find Taichi in config mode
# directly.
find_package(taichi CONFIG QUIET HINTS "${TAICHI_C_API_INSTALL_DIR}")

if(taichi_FOUND)
    message("-- Found Taichi ${taichi_VERSION} in config mode: ${taichi_DIR}")
else()
    message("-- Could NOT find Taichi in config mode; fallback to heuristic search")
endif()

# - [taichi::runtime] ----------------------------------------------------------

if(("runtime" IN_LIST taichi_FIND_COMPONENTS) AND (NOT TARGET taichi::runtime))
    if(taichi_FOUND)
        if(NOT TARGET taichi_c_api)
            message(FATAL_ERROR "taichi is marked found but target taichi_c_api doesn't exists")
        endif()

        # Already found in config mode.
        get_target_property(taichi_runtime_CONFIG taichi_c_api IMPORTED_CONFIGURATIONS)
        if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
            get_target_property(taichi_runtime_LIBRARY taichi_c_api IMPORTED_IMPLIB_${taichi_runtime_CONFIG})
            get_target_property(taichi_runtime_REDIST_LIBRARY taichi_c_api IMPORTED_LOCATION_${taichi_runtime_CONFIG})
        else()
            get_target_property(taichi_runtime_LIBRARY taichi_c_api IMPORTED_LOCATION_${taichi_runtime_CONFIG})
            get_target_property(taichi_runtime_REDIST_LIBRARY taichi_c_api IMPORTED_LOCATION_${taichi_runtime_CONFIG})
            endif()
        get_target_property(taichi_runtime_INCLUDE_DIR taichi_c_api INTERFACE_INCLUDE_DIRECTORIES)
    else()
        find_library(taichi_runtime_LIBRARY
            NAMES taichi_runtime taichi_c_api
            HINTS ${TAICHI_C_API_INSTALL_DIR}
            PATH_SUFFIXES lib
            # CMake find root is overriden by Android toolchain.
            NO_CMAKE_FIND_ROOT_PATH)

        find_library(taichi_runtime_REDIST_LIBRARY
            NAMES taichi_runtime taichi_c_api
            HINTS ${TAICHI_C_API_INSTALL_DIR}
            PATH_SUFFIXES bin lib
            # CMake find root is overriden by Android toolchain.
            NO_CMAKE_FIND_ROOT_PATH)

        find_path(taichi_runtime_INCLUDE_DIR
            NAMES taichi/taichi.h
            HINTS ${TAICHI_C_API_INSTALL_DIR}
            PATH_SUFFIXES include
            NO_CMAKE_FIND_ROOT_PATH)
    endif()

    # Capture Taichi Runtime version from header definition.
    if(EXISTS "${taichi_runtime_INCLUDE_DIR}/taichi/taichi_core.h")
        file(READ "${taichi_runtime_INCLUDE_DIR}/taichi/taichi_core.h" taichi_runtime_VERSION_LITERAL)
        string(REGEX MATCH "#define TI_C_API_VERSION ([0-9]+)" taichi_runtime_VERSION_LITERAL ${taichi_runtime_VERSION_LITERAL})
        set(taichi_runtime_VERSION_LITERAL ${CMAKE_MATCH_1})
        math(EXPR taichi_runtime_VERSION_MAJOR "${taichi_runtime_VERSION_LITERAL} / 1000000")
        math(EXPR taichi_runtime_VERSION_MINOR "(${taichi_runtime_VERSION_LITERAL} / 1000) % 1000")
        math(EXPR taichi_runtime_VERSION_PATCH "${taichi_runtime_VERSION_LITERAL} % 1000")
        set(taichi_runtime_VERSION "${taichi_runtime_VERSION_MAJOR}.${taichi_runtime_VERSION_MINOR}.${taichi_runtime_VERSION_PATCH}")
    endif()

    # Ensure the version string is valid.
    if("${taichi_runtime_VERSION}" VERSION_GREATER "0")
        message("-- Found Taichi Runtime ${taichi_runtime_VERSION}: ${taichi_runtime_LIBRARY}")

        add_library(taichi::runtime UNKNOWN IMPORTED)
        set_target_properties(taichi::runtime PROPERTIES
            IMPORTED_LOCATION "${taichi_runtime_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${taichi_runtime_INCLUDE_DIR}")

        list(APPEND COMPONENT_VARS
            taichi_runtime_LIBRARY
            taichi_runtime_INCLUDE_DIR)
        list(APPEND taichi_LIBRARIES "${taichi_runtime_LIBRARY}")
        if(EXISTS ${taichi_runtime_REDIST_LIBRARY})
            list(APPEND taichi_REDIST_LIBRARIES ${taichi_runtime_REDIST_LIBRARY})
        endif()
        list(APPEND taichi_INCLUDE_DIRS "${taichi_runtime_INCLUDE_DIR}")
    endif()
endif()

# - [taichi] -------------------------------------------------------------------

set(taichi_VERSION taichi_runtime_VERSION)
set(taichi_FOUND TRUE)

# Handle `QUIET` and `REQUIRED` args in the recommended way in `find_package`.
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(taichi DEFAULT_MSG ${COMPONENT_VARS})

cmake_policy(POP)
