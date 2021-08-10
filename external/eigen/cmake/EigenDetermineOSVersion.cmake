# The utility function DetermineOSVersion aims at providing an
# improved version of the CMake variable ${CMAKE_SYSTEM} on Windows
# machines.
#
# Usage:
#  include(EigenDetermineOSVersion)
#  DetermineOSVersion(OS_VERSION)
#  message("OS: ${OS_VERSION}")

# - A little helper variable which should not be directly called
function(DetermineShortWindowsName WIN_VERSION win_num_version)
   if    (${win_num_version} VERSION_EQUAL "6.1")
       set(_version "win7")
   elseif(${win_num_version} VERSION_EQUAL "6.0")
       set(_version "winVista")
   elseif(${win_num_version} VERSION_EQUAL "5.2")
       set(_version "winXpProf")
   elseif(${win_num_version} VERSION_EQUAL "5.1")
       set(_version "winXp")
   elseif(${win_num_version} VERSION_EQUAL "5.0")
       set(_version "win2000Prof")
   else()
       set(_version "unknownWin")
   endif()
   set(${WIN_VERSION} ${_version} PARENT_SCOPE)
endfunction()

function(DetermineOSVersion OS_VERSION)
  if (WIN32 AND CMAKE_HOST_SYSTEM_NAME MATCHES Windows)
    file (TO_NATIVE_PATH "$ENV{COMSPEC}" SHELL)
    exec_program( ${SHELL} ARGS "/c" "ver" OUTPUT_VARIABLE ver_output)
				
      string(REGEX MATCHALL "[0-9]+"
           ver_list "${ver_output}")
      list(GET ver_list 0 _major)		   
      list(GET ver_list 1 _minor)
				
    set(win_num_version ${_major}.${_minor})
    DetermineShortWindowsName(win_version "${win_num_version}")
    if(win_version)
      set(${OS_VERSION} ${win_version} PARENT_SCOPE)
    endif()
  else()
    set(${OS_VERSION} ${CMAKE_SYSTEM} PARENT_SCOPE)
  endif()
endfunction()
