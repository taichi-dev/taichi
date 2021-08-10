project(EigenDemos)

add_custom_target(demos)

if(NOT EIGEN_TEST_NOQT)
  find_package(Qt4)
  if(QT4_FOUND)
    add_subdirectory(mandelbrot)
    add_subdirectory(opengl)
  else(QT4_FOUND)
    message(STATUS "Qt4 not found, so disabling the mandelbrot and opengl demos")
  endif(QT4_FOUND)
endif()
