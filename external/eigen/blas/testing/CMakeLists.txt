
macro(ei_add_blas_test testname)

  set(targetname ${testname})

  set(filename ${testname}.f)
  add_executable(${targetname} ${filename})

  target_link_libraries(${targetname} eigen_blas)

  if(EIGEN_STANDARD_LIBRARIES_TO_LINK_TO)
    target_link_libraries(${targetname} ${EIGEN_STANDARD_LIBRARIES_TO_LINK_TO})
  endif()

  target_link_libraries(${targetname} ${EXTERNAL_LIBS})

  add_test(${testname} "${Eigen_SOURCE_DIR}/blas/testing/runblastest.sh" "${testname}" "${Eigen_SOURCE_DIR}/blas/testing/${testname}.dat")
  add_dependencies(buildtests ${targetname})
  
endmacro(ei_add_blas_test)

ei_add_blas_test(sblat1)
ei_add_blas_test(sblat2)
ei_add_blas_test(sblat3)

ei_add_blas_test(dblat1)
ei_add_blas_test(dblat2)
ei_add_blas_test(dblat3)

ei_add_blas_test(cblat1)
ei_add_blas_test(cblat2)
ei_add_blas_test(cblat3)

ei_add_blas_test(zblat1)
ei_add_blas_test(zblat2)
ei_add_blas_test(zblat3)

# add_custom_target(level1)
# add_dependencies(level1 sblat1)

