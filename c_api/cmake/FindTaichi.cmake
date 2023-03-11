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

``Taichi::Runtime``
  The Taichi Runtime library.

Result Variables
^^^^^^^^^^^^^^^^

This will define the following variables:

``Taichi_FOUND``
  True if a Taichi installation is found.
``Taichi_VERSION``
  Version of installed Taichi. Components might have lower version numbers.
``Taichi_INCLUDE_DIRS``
  Paths to Include directories needed to use Taichi.
``Taichi_LIBRARIES``
  Paths to Taichi linking libraries (``.libs``).
``Taichi_REDIST_LIBRARIES``
  Paths to Taichi redistributed runtime libraries (``.so`` and ``.dll``). You
  might want to copy them next to your executables.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``Taichi_Runtime_VERSION``
  Taichi runtime library version.
``Taichi_Runtime_INCLUDE_DIR``
  The directory containing ``taichi/taichi.h``.
``Taichi_Runtime_LIBRARY``
  Path to linking library of ``Taichi_Runtime``.
``Taichi_Runtime_REDIST_LIBRARY``
  Path to redistributed runtime library of ``Taichi_Runtime``.

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
# New installation location after 2022-03-11.
if(NOT EXISTS "${TAICHI_C_API_INSTALL_DIR}" AND EXISTS "${Python_EXECUTABLE}")
    message("-- Looking for Taichi libraries via Python package installation (v2)")
    execute_process(COMMAND ${Python_EXECUTABLE} -c "import sys; import pathlib; print([pathlib.Path(x + '/taichi/_lib/c_api').resolve() for x in sys.path if pathlib.Path(x + '/taichi/_lib/c_api').exists()][0], end='')" OUTPUT_VARIABLE TAICHI_C_API_INSTALL_DIR)
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
if(TRUE)
    # (penguinliong) Currently we only have Taichi Runtime. We might make the
    # codegen a library in the future?
    set(Taichi_FIND_COMPONENTS "Runtime")
endif()

message("-- Looking for Taichi components: ${Taichi_FIND_COMPONENTS}")

# (penguinliong) Note that the config files only exposes libraries and their
# public headers. We still need to encapsulate the libraries into semantical
# CMake targets in this list. So please DO NOT find Taichi in config mode
# directly.
find_package(Taichi CONFIG QUIET HINTS "${TAICHI_C_API_INSTALL_DIR}")

if(Taichi_FOUND)
    message("-- Found Taichi ${Taichi_VERSION} in config mode: ${Taichi_DIR}")
else()
    message("-- Could NOT find Taichi in config mode; fallback to heuristic search")
endif()

# - [Taichi::Runtime] ----------------------------------------------------------

if(("Runtime" IN_LIST Taichi_FIND_COMPONENTS) AND (NOT TARGET Taichi::Runtime))
    if(Taichi_FOUND)
        if(NOT TARGET taichi_c_api)
            message(FATAL_ERROR "taichi is marked found but target taichi_c_api doesn't exists")
        endif()

        # Already found in config mode.
        get_target_property(Taichi_Runtime_CONFIG taichi_c_api IMPORTED_CONFIGURATIONS)
        if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
            get_target_property(Taichi_Runtime_LIBRARY taichi_c_api IMPORTED_IMPLIB)
        else()
            get_target_property(Taichi_Runtime_LIBRARY taichi_c_api LOCATION)
        endif()
        get_target_property(Taichi_Runtime_REDIST_LIBRARY taichi_c_api LOCATION)
        get_target_property(Taichi_Runtime_INCLUDE_DIR taichi_c_api INTERFACE_INCLUDE_DIRECTORIES)
    else()
        find_library(Taichi_Runtime_LIBRARY
            NAMES taichi_runtime taichi_c_api
            HINTS ${TAICHI_C_API_INSTALL_DIR}
            PATH_SUFFIXES lib
            # CMake find root is overriden by Android toolchain.
            NO_CMAKE_FIND_ROOT_PATH)

        find_library(Taichi_Runtime_REDIST_LIBRARY
            NAMES taichi_runtime taichi_c_api
            HINTS ${TAICHI_C_API_INSTALL_DIR}
            PATH_SUFFIXES bin lib
            # CMake find root is overriden by Android toolchain.
            NO_CMAKE_FIND_ROOT_PATH)

        find_path(Taichi_Runtime_INCLUDE_DIR
            NAMES taichi/taichi.h
            HINTS ${TAICHI_C_API_INSTALL_DIR}
            PATH_SUFFIXES include
            NO_CMAKE_FIND_ROOT_PATH)
    endif()

    # Capture Taichi Runtime version from header definition.
    if(EXISTS "${Taichi_Runtime_INCLUDE_DIR}/taichi/taichi_core.h")
        file(READ "${Taichi_Runtime_INCLUDE_DIR}/taichi/taichi_core.h" Taichi_Runtime_VERSION_LITERAL)
        string(REGEX MATCH "#define TI_C_API_VERSION ([0-9]+)" Taichi_Runtime_VERSION_LITERAL ${Taichi_Runtime_VERSION_LITERAL})
        set(Taichi_Runtime_VERSION_LITERAL ${CMAKE_MATCH_1})
        math(EXPR Taichi_Runtime_VERSION_MAJOR "${Taichi_Runtime_VERSION_LITERAL} / 1000000")
        math(EXPR Taichi_Runtime_VERSION_MINOR "(${Taichi_Runtime_VERSION_LITERAL} / 1000) % 1000")
        math(EXPR Taichi_Runtime_VERSION_PATCH "${Taichi_Runtime_VERSION_LITERAL} % 1000")
        set(Taichi_Runtime_VERSION "${Taichi_Runtime_VERSION_MAJOR}.${Taichi_Runtime_VERSION_MINOR}.${Taichi_Runtime_VERSION_PATCH}")
    endif()

    # Ensure the version string is valid.
    if("${Taichi_Runtime_VERSION}" VERSION_GREATER "0")
        message("-- Found Taichi Runtime ${Taichi_Runtime_VERSION}: ${Taichi_Runtime_LIBRARY}")

        add_library(Taichi::Runtime UNKNOWN IMPORTED)
        set_target_properties(Taichi::Runtime PROPERTIES
            IMPORTED_LOCATION "${Taichi_Runtime_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${Taichi_Runtime_INCLUDE_DIR}")

        list(APPEND COMPONENT_VARS
            Taichi_Runtime_REDIST_LIBRARY
            Taichi_Runtime_LIBRARY
            Taichi_Runtime_INCLUDE_DIR)
        list(APPEND Taichi_LIBRARIES "${Taichi_Runtime_LIBRARY}")
        if(EXISTS ${Taichi_Runtime_REDIST_LIBRARY})
            list(APPEND Taichi_REDIST_LIBRARIES "${Taichi_Runtime_REDIST_LIBRARY}")
        endif()
        list(APPEND Taichi_INCLUDE_DIRS "${Taichi_Runtime_INCLUDE_DIR}")
    endif()
endif()

# - [taichi] -------------------------------------------------------------------

# Handle `QUIET` and `REQUIRED` args in the recommended way in `find_package`.
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Taichi DEFAULT_MSG ${COMPONENT_VARS})
set(Taichi_VERSION ${Taichi_Runtime_VERSION})
set(Taichi_INCLUDE_DIRS ${Taichi_INCLUDE_DIRS})
set(Taichi_LIBRARIES ${Taichi_LIBRARIES})
set(Taichi_REDIST_LIBRARIES ${Taichi_REDIST_LIBRARIES})

set(Taichi_Runtime_VERSION ${Taichi_Runtime_VERSION})
set(Taichi_Runtime_INCLUDE_DIR ${Taichi_Runtime_INCLUDE_DIR})
set(Taichi_Runtime_LIBRARY ${Taichi_Runtime_LIBRARY})
set(Taichi_Runtime_REDIST_LIBRARY ${Taichi_Runtime_REDIST_LIBRARY})


cmake_policy(POP)
