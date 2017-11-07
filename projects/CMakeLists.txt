file(GLOB children RELATIVE ${CMAKE_CURRENT_LIST_DIR} *)

foreach(dir ${children})
    if(IS_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/${dir}")
        message("")
        message("Adding project ${dir}:")
        message("")
        add_subdirectory(${dir})
    endif()
endforeach()

set(TAICHI_PROJECT_LIBRARIES ${TAICHI_PROJECT_LIBRARIES} PARENT_SCOPE)
