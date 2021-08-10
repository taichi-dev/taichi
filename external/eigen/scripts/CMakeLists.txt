get_property(EIGEN_TESTS_LIST GLOBAL PROPERTY EIGEN_TESTS_LIST)
configure_file(buildtests.in ${CMAKE_BINARY_DIR}/buildtests.sh @ONLY)

configure_file(check.in ${CMAKE_BINARY_DIR}/check.sh COPYONLY)
configure_file(debug.in ${CMAKE_BINARY_DIR}/debug.sh COPYONLY)
configure_file(release.in ${CMAKE_BINARY_DIR}/release.sh COPYONLY)
