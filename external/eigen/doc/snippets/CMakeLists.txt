file(GLOB snippets_SRCS "*.cpp")

add_custom_target(all_snippets)

foreach(snippet_src ${snippets_SRCS})
  get_filename_component(snippet ${snippet_src} NAME_WE)
  set(compile_snippet_target compile_${snippet})
  set(compile_snippet_src ${compile_snippet_target}.cpp)
  file(READ ${snippet_src} snippet_source_code)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/compile_snippet.cpp.in
                 ${CMAKE_CURRENT_BINARY_DIR}/${compile_snippet_src})
  add_executable(${compile_snippet_target}
                 ${CMAKE_CURRENT_BINARY_DIR}/${compile_snippet_src})
  if(EIGEN_STANDARD_LIBRARIES_TO_LINK_TO)
    target_link_libraries(${compile_snippet_target} ${EIGEN_STANDARD_LIBRARIES_TO_LINK_TO})
  endif()
  add_custom_command(
    TARGET ${compile_snippet_target}
    POST_BUILD
    COMMAND ${compile_snippet_target}
    ARGS >${CMAKE_CURRENT_BINARY_DIR}/${snippet}.out
  )
  add_dependencies(all_snippets ${compile_snippet_target})
  set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/${compile_snippet_src}
                              PROPERTIES OBJECT_DEPENDS ${snippet_src})
endforeach(snippet_src)
