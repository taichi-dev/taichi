file(GLOB examples_SRCS "*.cpp")

foreach(example_src ${examples_SRCS})
  get_filename_component(example ${example_src} NAME_WE)
  add_executable(${example} ${example_src})
  if(EIGEN_STANDARD_LIBRARIES_TO_LINK_TO)
    target_link_libraries(${example} ${EIGEN_STANDARD_LIBRARIES_TO_LINK_TO})
  endif()
  add_custom_command(
    TARGET ${example}
    POST_BUILD
    COMMAND ${example}
    ARGS >${CMAKE_CURRENT_BINARY_DIR}/${example}.out
  )
  add_dependencies(all_examples ${example})
endforeach(example_src)

check_cxx_compiler_flag("-std=c++11" EIGEN_COMPILER_SUPPORT_CPP11)
if(EIGEN_COMPILER_SUPPORT_CPP11)
ei_add_target_property(nullary_indexing COMPILE_FLAGS "-std=c++11")
endif()