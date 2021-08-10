# Find LAPACK library
#
# This module finds an installed library that implements the LAPACK
# linear-algebra interface (see http://www.netlib.org/lapack/).
# The approach follows mostly that taken for the autoconf macro file, acx_lapack.m4
# (distributed at http://ac-archive.sourceforge.net/ac-archive/acx_lapack.html).
#
# This module sets the following variables:
#  LAPACK_FOUND - set to true if a library implementing the LAPACK interface
#    is found
#  LAPACK_INCLUDE_DIR - Directories containing the LAPACK header files
#  LAPACK_DEFINITIONS - Compilation options to use LAPACK
#  LAPACK_LINKER_FLAGS - Linker flags to use LAPACK (excluding -l
#    and -L).
#  LAPACK_LIBRARIES_DIR - Directories containing the LAPACK libraries.
#     May be null if LAPACK_LIBRARIES contains libraries name using full path.
#  LAPACK_LIBRARIES - List of libraries to link against LAPACK interface.
#     May be null if the compiler supports auto-link (e.g. VC++).
#  LAPACK_USE_FILE - The name of the cmake module to include to compile
#     applications or libraries using LAPACK.
#
# This module was modified by CGAL team:
# - find libraries for a C++ compiler, instead of Fortran
# - added LAPACK_INCLUDE_DIR, LAPACK_DEFINITIONS and LAPACK_LIBRARIES_DIR
# - removed LAPACK95_LIBRARIES


include(CheckFunctionExists)

# This macro checks for the existence of the combination of fortran libraries
# given by _list.  If the combination is found, this macro checks (using the
# check_function_exists macro) whether can link against that library
# combination using the name of a routine given by _name using the linker
# flags given by _flags.  If the combination of libraries is found and passes
# the link test, LIBRARIES is set to the list of complete library paths that
# have been found and DEFINITIONS to the required definitions.
# Otherwise, LIBRARIES is set to FALSE.
# N.B. _prefix is the prefix applied to the names of all cached variables that
# are generated internally and marked advanced by this macro.
macro(check_lapack_libraries DEFINITIONS LIBRARIES _prefix _name _flags _list _blas _path)
  #message("DEBUG: check_lapack_libraries(${_list} in ${_path} with ${_blas})")

  # Check for the existence of the libraries given by _list
  set(_libraries_found TRUE)
  set(_libraries_work FALSE)
  set(${DEFINITIONS} "")
  set(${LIBRARIES} "")
  set(_combined_name)
  foreach(_library ${_list})
    set(_combined_name ${_combined_name}_${_library})

    if(_libraries_found)
      # search first in ${_path}
      find_library(${_prefix}_${_library}_LIBRARY
                  NAMES ${_library}
                  PATHS ${_path} NO_DEFAULT_PATH
                  )
      # if not found, search in environment variables and system
      if ( WIN32 )
        find_library(${_prefix}_${_library}_LIBRARY
                    NAMES ${_library}
                    PATHS ENV LIB
                    )
      elseif ( APPLE )
        find_library(${_prefix}_${_library}_LIBRARY
                    NAMES ${_library}
                    PATHS /usr/local/lib /usr/lib /usr/local/lib64 /usr/lib64 ENV DYLD_LIBRARY_PATH
                    )
      else ()
        find_library(${_prefix}_${_library}_LIBRARY
                    NAMES ${_library}
                    PATHS /usr/local/lib /usr/lib /usr/local/lib64 /usr/lib64 ENV LD_LIBRARY_PATH
                    )
      endif()
      mark_as_advanced(${_prefix}_${_library}_LIBRARY)
      set(${LIBRARIES} ${${LIBRARIES}} ${${_prefix}_${_library}_LIBRARY})
      set(_libraries_found ${${_prefix}_${_library}_LIBRARY})
    endif(_libraries_found)
  endforeach(_library ${_list})
  if(_libraries_found)
    set(_libraries_found ${${LIBRARIES}})
  endif()

  # Test this combination of libraries with the Fortran/f2c interface.
  # We test the Fortran interface first as it is well standardized.
  if(_libraries_found AND NOT _libraries_work)
    set(${DEFINITIONS}  "-D${_prefix}_USE_F2C")
    set(${LIBRARIES}    ${_libraries_found})
    # Some C++ linkers require the f2c library to link with Fortran libraries.
    # I do not know which ones, thus I just add the f2c library if it is available.
    find_package( F2C QUIET )
    if ( F2C_FOUND )
      set(${DEFINITIONS}  ${${DEFINITIONS}} ${F2C_DEFINITIONS})
      set(${LIBRARIES}    ${${LIBRARIES}} ${F2C_LIBRARIES})
    endif()
    set(CMAKE_REQUIRED_DEFINITIONS  ${${DEFINITIONS}})
    set(CMAKE_REQUIRED_LIBRARIES    ${_flags} ${${LIBRARIES}} ${_blas})
    #message("DEBUG: CMAKE_REQUIRED_DEFINITIONS = ${CMAKE_REQUIRED_DEFINITIONS}")
    #message("DEBUG: CMAKE_REQUIRED_LIBRARIES = ${CMAKE_REQUIRED_LIBRARIES}")
    # Check if function exists with f2c calling convention (ie a trailing underscore)
    check_function_exists(${_name}_ ${_prefix}_${_name}_${_combined_name}_f2c_WORKS)
    set(CMAKE_REQUIRED_DEFINITIONS} "")
    set(CMAKE_REQUIRED_LIBRARIES    "")
    mark_as_advanced(${_prefix}_${_name}_${_combined_name}_f2c_WORKS)
    set(_libraries_work ${${_prefix}_${_name}_${_combined_name}_f2c_WORKS})
  endif(_libraries_found AND NOT _libraries_work)

  # If not found, test this combination of libraries with a C interface.
  # A few implementations (ie ACML) provide a C interface. Unfortunately, there is no standard.
  if(_libraries_found AND NOT _libraries_work)
    set(${DEFINITIONS} "")
    set(${LIBRARIES}   ${_libraries_found})
    set(CMAKE_REQUIRED_DEFINITIONS "")
    set(CMAKE_REQUIRED_LIBRARIES   ${_flags} ${${LIBRARIES}} ${_blas})
    #message("DEBUG: CMAKE_REQUIRED_LIBRARIES = ${CMAKE_REQUIRED_LIBRARIES}")
    check_function_exists(${_name} ${_prefix}_${_name}${_combined_name}_WORKS)
    set(CMAKE_REQUIRED_LIBRARIES "")
    mark_as_advanced(${_prefix}_${_name}${_combined_name}_WORKS)
    set(_libraries_work ${${_prefix}_${_name}${_combined_name}_WORKS})
  endif(_libraries_found AND NOT _libraries_work)

  # on failure
  if(NOT _libraries_work)
    set(${DEFINITIONS} "")
    set(${LIBRARIES}   FALSE)
  endif()
  #message("DEBUG: ${DEFINITIONS} = ${${DEFINITIONS}}")
  #message("DEBUG: ${LIBRARIES} = ${${LIBRARIES}}")
endmacro(check_lapack_libraries)


#
# main
#

# LAPACK requires BLAS
if(LAPACK_FIND_QUIETLY OR NOT LAPACK_FIND_REQUIRED)
  find_package(BLAS)
else()
  find_package(BLAS REQUIRED)
endif()

if (NOT BLAS_FOUND)

  message(STATUS "LAPACK requires BLAS.")
  set(LAPACK_FOUND FALSE)

# Is it already configured?
elseif (LAPACK_LIBRARIES_DIR OR LAPACK_LIBRARIES)

  set(LAPACK_FOUND TRUE)

else()

  # reset variables
  set( LAPACK_INCLUDE_DIR "" )
  set( LAPACK_DEFINITIONS "" )
  set( LAPACK_LINKER_FLAGS "" ) # unused (yet)
  set( LAPACK_LIBRARIES "" )
  set( LAPACK_LIBRARIES_DIR "" )

    #
    # If Unix, search for LAPACK function in possible libraries
    #

    #intel mkl lapack?
    if(NOT LAPACK_LIBRARIES)
      check_lapack_libraries(
      LAPACK_DEFINITIONS
      LAPACK_LIBRARIES
      LAPACK
      cheev
      ""
      "mkl_lapack"
      "${BLAS_LIBRARIES}"
      "${CGAL_TAUCS_LIBRARIES_DIR} ENV LAPACK_LIB_DIR"
      )
    endif()

    #acml lapack?
    if(NOT LAPACK_LIBRARIES)
      check_lapack_libraries(
      LAPACK_DEFINITIONS
      LAPACK_LIBRARIES
      LAPACK
      cheev
      ""
      "acml"
      "${BLAS_LIBRARIES}"
      "${CGAL_TAUCS_LIBRARIES_DIR} ENV LAPACK_LIB_DIR"
      )
    endif()

    # Apple LAPACK library?
    if(NOT LAPACK_LIBRARIES)
      check_lapack_libraries(
      LAPACK_DEFINITIONS
      LAPACK_LIBRARIES
      LAPACK
      cheev
      ""
      "Accelerate"
      "${BLAS_LIBRARIES}"
      "${CGAL_TAUCS_LIBRARIES_DIR} ENV LAPACK_LIB_DIR"
      )
    endif()

    if ( NOT LAPACK_LIBRARIES )
      check_lapack_libraries(
      LAPACK_DEFINITIONS
      LAPACK_LIBRARIES
      LAPACK
      cheev
      ""
      "vecLib"
      "${BLAS_LIBRARIES}"
      "${CGAL_TAUCS_LIBRARIES_DIR} ENV LAPACK_LIB_DIR"
      )
    endif ( NOT LAPACK_LIBRARIES )

    # Generic LAPACK library?
    # This configuration *must* be the last try as this library is notably slow.
    if ( NOT LAPACK_LIBRARIES )
      check_lapack_libraries(
      LAPACK_DEFINITIONS
      LAPACK_LIBRARIES
      LAPACK
      cheev
      ""
      "lapack"
      "${BLAS_LIBRARIES}"
      "${CGAL_TAUCS_LIBRARIES_DIR} ENV LAPACK_LIB_DIR"
      )
    endif()

  if(LAPACK_LIBRARIES_DIR OR LAPACK_LIBRARIES)
    set(LAPACK_FOUND TRUE)
  else()
    set(LAPACK_FOUND FALSE)
  endif()

  if(NOT LAPACK_FIND_QUIETLY)
    if(LAPACK_FOUND)
      message(STATUS "A library with LAPACK API found.")
    else(LAPACK_FOUND)
      if(LAPACK_FIND_REQUIRED)
        message(FATAL_ERROR "A required library with LAPACK API not found. Please specify library location.")
      else()
        message(STATUS "A library with LAPACK API not found. Please specify library location.")
      endif()
    endif(LAPACK_FOUND)
  endif(NOT LAPACK_FIND_QUIETLY)

  # Add variables to cache
  set( LAPACK_INCLUDE_DIR   "${LAPACK_INCLUDE_DIR}"
                            CACHE PATH "Directories containing the LAPACK header files" FORCE )
  set( LAPACK_DEFINITIONS   "${LAPACK_DEFINITIONS}"
                            CACHE STRING "Compilation options to use LAPACK" FORCE )
  set( LAPACK_LINKER_FLAGS  "${LAPACK_LINKER_FLAGS}"
                            CACHE STRING "Linker flags to use LAPACK" FORCE )
  set( LAPACK_LIBRARIES     "${LAPACK_LIBRARIES}"
                            CACHE FILEPATH "LAPACK libraries name" FORCE )
  set( LAPACK_LIBRARIES_DIR "${LAPACK_LIBRARIES_DIR}"
                            CACHE PATH "Directories containing the LAPACK libraries" FORCE )

  #message("DEBUG: LAPACK_INCLUDE_DIR = ${LAPACK_INCLUDE_DIR}")
  #message("DEBUG: LAPACK_DEFINITIONS = ${LAPACK_DEFINITIONS}")
  #message("DEBUG: LAPACK_LINKER_FLAGS = ${LAPACK_LINKER_FLAGS}")
  #message("DEBUG: LAPACK_LIBRARIES = ${LAPACK_LIBRARIES}")
  #message("DEBUG: LAPACK_LIBRARIES_DIR = ${LAPACK_LIBRARIES_DIR}")
  #message("DEBUG: LAPACK_FOUND = ${LAPACK_FOUND}")

endif(NOT BLAS_FOUND)
