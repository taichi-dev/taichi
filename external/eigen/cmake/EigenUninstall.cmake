################ CMake Uninstall Template #######################
# CMake Template file for uninstallation of files
# mentioned in 'install_manifest.txt'
#
# Used by uinstall target
#################################################################

set(MANIFEST "${CMAKE_CURRENT_BINARY_DIR}/install_manifest.txt")

if(EXISTS ${MANIFEST})
  message(STATUS "============== Uninstalling Eigen  ===================")

  file(STRINGS ${MANIFEST} files)
  foreach(file ${files})
    if(EXISTS ${file})
      message(STATUS "Removing file: '${file}'")

      execute_process(
        COMMAND ${CMAKE_COMMAND} -E remove ${file}
        OUTPUT_VARIABLE rm_out
        RESULT_VARIABLE rm_retval
        )

      if(NOT "${rm_retval}" STREQUAL 0)
        message(FATAL_ERROR "Failed to remove file: '${file}'.")
      endif()
    else()
      message(STATUS "File '${file}' does not exist.")
    endif()
  endforeach(file)

  message(STATUS "========== Finished Uninstalling Eigen  ==============")
else()
  message(STATUS "Cannot find install manifest: '${MANIFEST}'")
  message(STATUS "Probably make install has not been performed")
  message(STATUS "   or install_manifest.txt has been deleted.")
endif()



