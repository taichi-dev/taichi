
ADD_CUSTOM_TARGET(copy_scripts)

SET(script_files go_mean mk_mean_script.sh mk_new_gnuplot.sh
    perlib_plot_settings.txt action_settings.txt gnuplot_common_settings.hh )

FOREACH(script_file ${script_files})
ADD_CUSTOM_COMMAND(
  TARGET copy_scripts
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${script_file} ${CMAKE_CURRENT_BINARY_DIR}/
  ARGS
)
ENDFOREACH(script_file)

ADD_CUSTOM_COMMAND(
  TARGET copy_scripts
  POST_BUILD
  COMMAND ${CMAKE_CXX_COMPILER} --version | head -n 1 > ${CMAKE_CURRENT_BINARY_DIR}/compiler_version.txt
  ARGS
)
ADD_CUSTOM_COMMAND(
  TARGET copy_scripts
  POST_BUILD
  COMMAND echo "${Eigen_SOURCE_DIR}" > ${CMAKE_CURRENT_BINARY_DIR}/eigen_root_dir.txt
  ARGS
)

add_executable(smooth smooth.cxx)
add_executable(regularize regularize.cxx)
add_executable(main mean.cxx)
add_dependencies(main copy_scripts)
