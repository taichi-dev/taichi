FILE(GLOB Eigen_Eigen2Support_Geometry_SRCS "*.h")

INSTALL(FILES
  ${Eigen_Eigen2Support_Geometry_SRCS}
  DESTINATION ${INCLUDE_INSTALL_DIR}/Eigen/src/Eigen2Support/Geometry
  )
