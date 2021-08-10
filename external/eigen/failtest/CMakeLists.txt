message(STATUS "Running the failtests")

ei_add_failtest("failtest_sanity_check")

ei_add_failtest("block_nonconst_ctor_on_const_xpr_0")
ei_add_failtest("block_nonconst_ctor_on_const_xpr_1")
ei_add_failtest("block_nonconst_ctor_on_const_xpr_2")
ei_add_failtest("transpose_nonconst_ctor_on_const_xpr")
ei_add_failtest("diagonal_nonconst_ctor_on_const_xpr")
ei_add_failtest("cwiseunaryview_nonconst_ctor_on_const_xpr")
ei_add_failtest("triangularview_nonconst_ctor_on_const_xpr")
ei_add_failtest("selfadjointview_nonconst_ctor_on_const_xpr")

ei_add_failtest("const_qualified_block_method_retval_0")
ei_add_failtest("const_qualified_block_method_retval_1")
ei_add_failtest("const_qualified_transpose_method_retval")
ei_add_failtest("const_qualified_diagonal_method_retval")

ei_add_failtest("map_nonconst_ctor_on_const_ptr_0")
ei_add_failtest("map_nonconst_ctor_on_const_ptr_1")
ei_add_failtest("map_nonconst_ctor_on_const_ptr_2")
ei_add_failtest("map_nonconst_ctor_on_const_ptr_3")
ei_add_failtest("map_nonconst_ctor_on_const_ptr_4")

ei_add_failtest("map_on_const_type_actually_const_0")
ei_add_failtest("map_on_const_type_actually_const_1")
ei_add_failtest("block_on_const_type_actually_const_0")
ei_add_failtest("block_on_const_type_actually_const_1")
ei_add_failtest("transpose_on_const_type_actually_const")
ei_add_failtest("diagonal_on_const_type_actually_const")
ei_add_failtest("cwiseunaryview_on_const_type_actually_const")
ei_add_failtest("triangularview_on_const_type_actually_const")
ei_add_failtest("selfadjointview_on_const_type_actually_const")

ei_add_failtest("ref_1")
ei_add_failtest("ref_2")
ei_add_failtest("ref_3")
ei_add_failtest("ref_4")
ei_add_failtest("ref_5")

ei_add_failtest("swap_1")
ei_add_failtest("swap_2")

ei_add_failtest("ternary_1")
ei_add_failtest("ternary_2")

ei_add_failtest("sparse_ref_1")
ei_add_failtest("sparse_ref_2")
ei_add_failtest("sparse_ref_3")
ei_add_failtest("sparse_ref_4")
ei_add_failtest("sparse_ref_5")

ei_add_failtest("sparse_storage_mismatch")

ei_add_failtest("partialpivlu_int")
ei_add_failtest("fullpivlu_int")
ei_add_failtest("llt_int")
ei_add_failtest("ldlt_int")
ei_add_failtest("qr_int")
ei_add_failtest("colpivqr_int")
ei_add_failtest("fullpivqr_int")
ei_add_failtest("jacobisvd_int")
ei_add_failtest("bdcsvd_int")
ei_add_failtest("eigensolver_int")
ei_add_failtest("eigensolver_cplx")

if (EIGEN_FAILTEST_FAILURE_COUNT)
  message(FATAL_ERROR
          "${EIGEN_FAILTEST_FAILURE_COUNT} out of ${EIGEN_FAILTEST_COUNT} failtests FAILED. "
          "To debug these failures, manually compile these programs in ${CMAKE_CURRENT_SOURCE_DIR}, "
          "with and without #define EIGEN_SHOULD_FAIL_TO_BUILD.")
else()
  message(STATUS "Failtest SUCCESS: all ${EIGEN_FAILTEST_COUNT} failtests passed.")
  message(STATUS "")
endif()
