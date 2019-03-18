FILE(GLOB Eigen_Eigen2Support_SRCS "*.h")

INSTALL(FILES
  ${Eigen_Eigen2Support_SRCS}
  DESTINATION ${INCLUDE_INSTALL_DIR}/Eigen/src/Eigen2Support COMPONENT Devel
  )

ADD_SUBDIRECTORY(Geometry)