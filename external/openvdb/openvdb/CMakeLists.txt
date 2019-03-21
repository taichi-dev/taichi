# Copyright (c) 2012-2016 DreamWorks Animation LLC
#
# All rights reserved. This software is distributed under the
# Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
#
# Redistributions of source code must retain the above copyright
# and license notice and the following restrictions and disclaimer.
#
# *     Neither the name of DreamWorks Animation nor the names of
# its contributors may be used to endorse or promote products derived
# from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# IN NO EVENT SHALL THE COPYRIGHT HOLDERS' AND CONTRIBUTORS' AGGREGATE
# LIABILITY FOR ALL CLAIMS REGARDLESS OF THEIR BASIS EXCEED US$250.00.
#
cmake_minimum_required(VERSION 3.1)
PROJECT ( OpenVDBCore )

set(CMAKE_CXX_STANDARD 11)

IF ( USE_GLFW3 )
  FIND_PACKAGE ( GLFW3 REQUIRED )
  SET ( OPENVDB_USE_GLFW_FLAG "-DOPENVDB_USE_GLFW_3" )
  IF ( UNIX AND NOT APPLE )
    SET ( GLFW_DEPENDENT_LIBRARIES "X11;Xi;Xrandr;Xinerama;Xcursor;Xxf86vm" )
  ENDIF ( UNIX AND NOT APPLE )
  SET ( GLFW_LINK_LIBRARY ${GLFW3_glfw_LIBRARY} )
  SET ( GLFW_INCLUDE_DIRECTORY  ${GLFW3_INCLUDE_DIR} CACHE STRING "GLFW3 include directory")
ELSE ()
  FIND_PACKAGE ( GLFW REQUIRED )
  IF ( UNIX AND NOT APPLE )
    SET ( GLFW_DEPENDENT_LIBRARIES "X11;Xi;Xrandr;Xinerama;Xcursor;Xxf86vm" )
  ENDIF ()
  SET ( GLFW_LINK_LIBRARY ${GLFW_glfw_LIBRARY} )
  SET ( GLFW_INCLUDE_DIRECTORY  ${GLFW_INCLUDE_DIR} CACHE STRING "GLFW include directory")
ENDIF ()

IF (WIN32)
  FIND_PACKAGE ( GLEW REQUIRED )
ELSE ()
  FIND_PACKAGE ( Threads REQUIRED )
ENDIF()

# Determine OpenVDB version
#define OPENVDB_LIBRARY_MAJOR_VERSION_NUMBER 4
#define OPENVDB_LIBRARY_MINOR_VERSION_NUMBER 0
#define OPENVDB_LIBRARY_PATCH_VERSION_NUMBER 0
SET ( OPENVDB_VERSION_FILE ${CMAKE_CURRENT_SOURCE_DIR}/version.h )

FILE ( STRINGS "${OPENVDB_VERSION_FILE}" openvdb_major_version_str REGEX "^#define[\t ]+OPENVDB_LIBRARY_MAJOR_VERSION_NUMBER[\t ]+.*")
FILE ( STRINGS "${OPENVDB_VERSION_FILE}" openvdb_minor_version_str REGEX "^#define[\t ]+OPENVDB_LIBRARY_MINOR_VERSION_NUMBER[\t ]+.*")
FILE ( STRINGS "${OPENVDB_VERSION_FILE}" openvdb_patch_version_str REGEX "^#define[\t ]+OPENVDB_LIBRARY_PATCH_VERSION_NUMBER[\t ]+.*")

STRING (REGEX REPLACE "^.*OPENVDB_LIBRARY_MAJOR_VERSION_NUMBER[\t ]+([0-9]*).*$" "\\1"
  _openvdb_major_version_number "${openvdb_major_version_str}")
STRING (REGEX REPLACE "^.*OPENVDB_LIBRARY_MINOR_VERSION_NUMBER[\t ]+([0-9]*).*$" "\\1"
  _openvdb_minor_version_number "${openvdb_minor_version_str}")
STRING (REGEX REPLACE "^.*OPENVDB_LIBRARY_PATCH_VERSION_NUMBER[\t ]+([0-9]*).*$" "\\1"
  _openvdb_patch_version_number "${openvdb_patch_version_str}")

SET ( OPENVDB_MAJOR_VERSION_NUMBER ${_openvdb_major_version_number} CACHE STRING "OpenVDB major version number" )
SET ( OPENVDB_MINOR_VERSION_NUMBER ${_openvdb_minor_version_number} CACHE STRING "OpenVDB minor version number" )
SET ( OPENVDB_PATCH_VERSION_NUMBER ${_openvdb_patch_version_number} CACHE STRING "OpenVDB build version number" )

# MESSAGE ( "OPENVDB_VERSION_FILE = ${OPENVDB_VERSION_FILE}")
# MESSAGE ( "OPENVDB_MAJOR_VERSION_NUMBER = ${OPENVDB_MAJOR_VERSION_NUMBER}" )
# MESSAGE ( "OPENVDB_MINOR_VERSION_NUMBER = ${OPENVDB_MINOR_VERSION_NUMBER}" )
# MESSAGE ( "OPENVDB_PATCH_VERSION_NUMBER = ${OPENVDB_PATCH_VERSION_NUMBER}" )

FIND_PACKAGE ( Boost ${MINIMUM_BOOST_VERSION} REQUIRED COMPONENTS iostreams system thread ${OPENVDB_BOOST_PYTHON_LIBRARY_COMPONENT_NAME} )
IF (WIN32 AND OPENVDB_DISABLE_BOOST_IMPLICIT_LINKING)
  ADD_DEFINITIONS ( -DBOOST_ALL_NO_LIB )
ENDIF ()

FIND_PACKAGE ( Blosc REQUIRED )
FIND_PACKAGE ( TBB REQUIRED )
FIND_PACKAGE ( ZLIB REQUIRED )
FIND_PACKAGE ( ILMBase REQUIRED )
FIND_PACKAGE ( OpenEXR REQUIRED )
FIND_PACKAGE ( OpenGL REQUIRED )
IF (OPENVDB_BUILD_DOCS)
  FIND_PACKAGE ( Doxygen REQUIRED )
ENDIF ()
IF (OPENVDB_BUILD_UNITTESTS)
  FIND_PACKAGE ( CPPUnit REQUIRED )
ENDIF ()

IF (APPLE)
  FIND_LIBRARY ( COCOA_LIBRARY Cocoa )
  FIND_LIBRARY ( IOKIT_LIBRARY IOKit )
  FIND_LIBRARY ( COREVIDEO_LIBRARY CoreVideo )
ENDIF ()

IF ( NOT OPENVDB_BUILD_HOUDINI_SOPS )
  # If we are building OpenVDB Houdini SOP, we need to use the headers
  # shipped with Houdini to ensure version compatibility
  INCLUDE_DIRECTORIES ( SYSTEM ${Boost_INCLUDE_DIR} )
  INCLUDE_DIRECTORIES ( SYSTEM ${TBB_INCLUDE_DIR} )
  # INCLUDE_DIRECTORIES ( ${OPENGL_INCLUDE_DIR} )
  INCLUDE_DIRECTORIES ( SYSTEM ${ZLIB_INCLUDE_DIR} )
ENDIF ()

INCLUDE_DIRECTORIES ( SYSTEM ${ILMBASE_INCLUDE_DIRS} )
INCLUDE_DIRECTORIES ( SYSTEM ${OPENEXR_INCLUDE_DIRS} )
INCLUDE_DIRECTORIES ( SYSTEM ${GLFW_INCLUDE_DIRECTORY} )

IF (WIN32)
  INCLUDE_DIRECTORIES ( SYSTEM ${GLEW_INCLUDE_DIR} )
ENDIF ()

INCLUDE_DIRECTORIES ( . ) # So they can simulate referencing installed headers

IF (WIN32)
  # Because of implicit linking!
  LINK_DIRECTORIES ( ${Boost_LIBRARY_DIR} )
ENDIF ()

INCLUDE_DIRECTORIES ( SYSTEM ${BLOSC_INCLUDE_DIR} )
INCLUDE_DIRECTORIES ( SYSTEM ${CPPUNIT_INCLUDE_DIR} )

IF (WIN32)
  ADD_DEFINITIONS ( -D_WIN32 -DNOMINMAX -DHALF_EXPORTS -DOPENVDB_DLL )
ELSE ()
  ADD_DEFINITIONS ( -pthread -fPIC )
ENDIF ()

# Working around hardcoding of path at DWA
FILE ( GLOB VIEWER_SOURCE_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} viewer/* )
FILE ( MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/dwa/openvdb_viewer )
FILE ( COPY ${VIEWER_SOURCE_FILES} DESTINATION ${CMAKE_BINARY_DIR}/dwa/openvdb_viewer )
INCLUDE_DIRECTORIES ( ${CMAKE_BINARY_DIR}/dwa )

SET ( OPENVDB_LIBRARY_SOURCE_FILES
  Grid.cc
  MetaMap.cc
  Metadata.cc
  Platform.cc
  io/Archive.cc
  io/Compression.cc
  io/File.cc
  io/GridDescriptor.cc
  io/Queue.cc
  io/Stream.cc
  io/TempFile.cc
  math/Maps.cc
  math/Proximity.cc
  math/QuantizedUnitVec.cc
  math/Transform.cc
  points/AttributeArray.cc
  points/AttributeArrayString.cc
  points/AttributeGroup.cc
  points/AttributeSet.cc
  points/StreamCompression.cc
  points/points.cc
  openvdb.cc
  util/Formats.cc
  util/Util.cc
  )

SET_SOURCE_FILES_PROPERTIES ( ${OPENVDB_LIBRARY_SOURCE_FILES}
  PROPERTIES
  COMPILE_FLAGS "-DOPENVDB_PRIVATE -DOPENVDB_USE_BLOSC ${OPENVDB_USE_GLFW_FLAG}"
  )

ADD_LIBRARY ( openvdb_static STATIC
  ${OPENVDB_LIBRARY_SOURCE_FILES}
  )
ADD_LIBRARY ( openvdb_shared SHARED
  ${OPENVDB_LIBRARY_SOURCE_FILES}
  )

TARGET_LINK_LIBRARIES ( openvdb_static
  ${Boost_IOSTREAMS_LIBRARY}
  ${Boost_SYSTEM_LIBRARY}
  ${Tbb_TBB_LIBRARY}
  ${Ilmbase_HALF_LIBRARY}
  ${ZLIB_LIBRARY}
  ${BLOSC_blosc_LIBRARY}
  )
TARGET_LINK_LIBRARIES ( openvdb_shared
  ${Boost_IOSTREAMS_LIBRARY}
  ${Boost_SYSTEM_LIBRARY}
  ${Tbb_TBB_LIBRARY}
  ${Ilmbase_HALF_LIBRARY}
  ${ZLIB_LIBRARY}
  ${BLOSC_blosc_LIBRARY}
  )

IF (WIN32)
  SET_TARGET_PROPERTIES ( openvdb_static
	PROPERTIES OUTPUT_NAME libopenvdb
	)
ELSE()
  SET_TARGET_PROPERTIES ( openvdb_static
	PROPERTIES OUTPUT_NAME openvdb
	)
ENDIF()

SET_TARGET_PROPERTIES(
  openvdb_shared
  PROPERTIES
  OUTPUT_NAME openvdb
  SOVERSION ${OPENVDB_MAJOR_VERSION_NUMBER}.${OPENVDB_MINOR_VERSION_NUMBER}
  VERSION ${OPENVDB_MAJOR_VERSION_NUMBER}.${OPENVDB_MINOR_VERSION_NUMBER}.${OPENVDB_PATCH_VERSION_NUMBER}
  )

SET ( VDB_PRINT_SOURCE_FILES  cmd/openvdb_print/main.cc )
SET_SOURCE_FILES_PROPERTIES ( ${VDB_PRINT_SOURCE_FILES}
  PROPERTIES
  COMPILE_FLAGS "-DOPENVDB_USE_BLOSC ${OPENVDB_USE_GLFW_FLAG}"
  )

ADD_EXECUTABLE ( vdb_print
  ${VDB_PRINT_SOURCE_FILES}
  )

TARGET_LINK_LIBRARIES ( vdb_print
  openvdb_shared
  ${CMAKE_THREAD_LIBS_INIT}
  ${BLOSC_blosc_LIBRARY}
  m
  stdc++
  )

SET ( VDB_RENDER_SOURCE_FILES cmd/openvdb_render/main.cc )
SET_SOURCE_FILES_PROPERTIES ( ${VDB_RENDER_SOURCE_FILES}
  PROPERTIES
  COMPILE_FLAGS "-DOPENVDB_USE_BLOSC ${OPENVDB_USE_GLFW_FLAG}"
  )
ADD_EXECUTABLE ( vdb_render
  ${VDB_RENDER_SOURCE_FILES}
  )

TARGET_LINK_LIBRARIES ( vdb_render
  openvdb_shared
  ${CMAKE_THREAD_LIBS_INIT}
  ${BLOSC_blosc_LIBRARY}
  ${Openexr_ILMIMF_LIBRARY}
  ${Ilmbase_ILMTHREAD_LIBRARY}
  ${Ilmbase_IEX_LIBRARY}
  m
  stdc++
  )

SET ( VDB_VIEW_SOURCE_FILES
  cmd/openvdb_view/main.cc
  viewer/Camera.cc
  viewer/ClipBox.cc
  viewer/Font.cc
  viewer/RenderModules.cc
  viewer/Viewer.cc
  )
SET_SOURCE_FILES_PROPERTIES ( ${VDB_VIEW_SOURCE_FILES}
  PROPERTIES
  COMPILE_FLAGS "-DOPENVDB_USE_BLOSC ${OPENVDB_USE_GLFW_FLAG} -DGL_GLEXT_PROTOTYPES=1"
  )
IF (NOT WIN32)
  ADD_EXECUTABLE ( vdb_view
    ${VDB_VIEW_SOURCE_FILES}
    )
  
  target_include_directories ( vdb_view SYSTEM PRIVATE ${Boost_INCLUDE_DIR} )

  TARGET_LINK_LIBRARIES ( vdb_view
    openvdb_shared
    ${Boost_THREAD_LIBRARY}
    ${OPENGL_gl_LIBRARY}
    ${OPENGL_glu_LIBRARY}
    ${COCOA_LIBRARY}
    ${IOKIT_LIBRARY}
    ${COREVIDEO_LIBRARY}
    ${GLFW_LINK_LIBRARY}
    ${GLFW_DEPENDENT_LIBRARIES}
    ${GLEW_GLEW_LIBRARY}
    m
    stdc++
    )
ENDIF ()

SET ( UNITTEST_SOURCE_FILES
  unittest/main.cc
  unittest/TestAttributeArray.cc
  unittest/TestAttributeArrayString.cc
  unittest/TestAttributeGroup.cc
  unittest/TestAttributeSet.cc
  unittest/TestBBox.cc
  unittest/TestConjGradient.cc
  unittest/TestCoord.cc
  unittest/TestCpt.cc
  unittest/TestCurl.cc
  unittest/TestDense.cc
  unittest/TestDenseSparseTools.cc
  unittest/TestDiagnostics.cc
  unittest/TestDivergence.cc
  unittest/TestDoubleMetadata.cc
  unittest/TestExceptions.cc
  unittest/TestFile.cc
  unittest/TestFloatMetadata.cc
  unittest/TestGradient.cc
  unittest/TestGrid.cc
  unittest/TestGridBbox.cc
  unittest/TestGridDescriptor.cc
  unittest/TestGridIO.cc
  unittest/TestGridTransformer.cc
  unittest/TestIndexFilter.cc
  unittest/TestIndexIterator.cc
  unittest/TestInit.cc
  unittest/TestInt32Metadata.cc
  unittest/TestInt64Metadata.cc
  unittest/TestInternalOrigin.cc
  unittest/TestLaplacian.cc
  unittest/TestLeaf.cc
  unittest/TestLeafBool.cc
  unittest/TestLeafManager.cc
  unittest/TestLeafMask.cc
  unittest/TestLeafIO.cc
  unittest/TestLeafOrigin.cc
  unittest/TestLevelSetRayIntersector.cc
  unittest/TestLevelSetUtil.cc
  unittest/TestLinearInterp.cc
  unittest/TestMaps.cc
  unittest/TestMat4Metadata.cc
  unittest/TestMath.cc
  unittest/TestMeanCurvature.cc
  unittest/TestMeshToVolume.cc
  unittest/TestMetadata.cc
  unittest/TestMetadataIO.cc
  unittest/TestMetaMap.cc
  unittest/TestMultiResGrid.cc
  unittest/TestName.cc
  unittest/TestNodeIterator.cc
  unittest/TestNodeManager.cc
  unittest/TestNodeMask.cc
  unittest/TestParticleAtlas.cc
  unittest/TestParticlesToLevelSet.cc
  unittest/TestPointAttribute.cc
  unittest/TestPointConversion.cc
  unittest/TestPointCount.cc
  unittest/TestPointDataLeaf.cc
  unittest/TestPointGroup.cc
  unittest/TestPointIndexGrid.cc
  unittest/TestPointPartitioner.cc
  unittest/TestPointsToMask.cc
  unittest/TestPoissonSolver.cc
  unittest/TestPrePostAPI.cc
  unittest/TestQuadraticInterp.cc
  unittest/TestQuantizedUnitVec.cc
  unittest/TestQuat.cc
  unittest/TestRay.cc
  unittest/TestStats.cc
  unittest/TestStream.cc
  unittest/TestStreamCompression.cc
  unittest/TestStringMetadata.cc
  unittest/TestTools.cc
  unittest/TestTopologyToLevelSet.cc
  unittest/TestTransform.cc
  unittest/TestTree.cc
  unittest/TestTreeCombine.cc
  unittest/TestTreeGetSetValues.cc
  unittest/TestTreeIterators.cc
  unittest/TestTreeVisitor.cc
  unittest/TestUtil.cc
  unittest/TestValueAccessor.cc
  unittest/TestVec2Metadata.cc
  unittest/TestVec3Metadata.cc
  unittest/TestVolumeRayIntersector.cc
  unittest/TestVolumeToMesh.cc
  )

SET_SOURCE_FILES_PROPERTIES ( ${UNITTEST_SOURCE_FILES}
  PROPERTIES
  COMPILE_FLAGS "-DOPENVDB_USE_BLOSC"
  )

IF ( OPENVDB_BUILD_UNITTESTS )

  ADD_EXECUTABLE ( vdb_test
    ${UNITTEST_SOURCE_FILES}
    )

  TARGET_LINK_LIBRARIES ( vdb_test
    ${CPPUnit_cppunit_LIBRARY}
    openvdb_shared
    m
    stdc++
    )

  ADD_TEST ( vdb_unit_test vdb_test )

ENDIF (OPENVDB_BUILD_UNITTESTS)

# Doxygen docmentation
IF (OPENVDB_BUILD_DOCS)

  FILE ( READ ${CMAKE_CURRENT_SOURCE_DIR}/doxygen-config DOXYGEN_CONFIG_CONTENT )
  FILE ( WRITE ${CMAKE_CURRENT_BINARY_DIR}/cmake-doxygen-config ${DOXYGEN_CONFIG_CONTENT} )
  FILE ( APPEND ${CMAKE_CURRENT_BINARY_DIR}/cmake-doxygen-config "OUTPUT_DIRECTORY=${CMAKE_CURRENT_BINARY_DIR}/doc\n" )

  ADD_CUSTOM_TARGET ( doc
    ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/cmake-doxygen-config
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    COMMENT "Generating API documentation with Doxygen" VERBATIM
    )

  INSTALL ( CODE "EXECUTE_PROCESS(COMMAND \"${CMAKE_COMMAND}\" --build  \"${PROJECT_BINARY_DIR}\" --target doc)")
  INSTALL ( DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/doc/html DESTINATION docs)

ENDIF ()

# Installation
IF ( NOT WIN32 )
  INSTALL ( TARGETS
    vdb_view
    DESTINATION
    bin
    )
ENDIF ()
INSTALL ( TARGETS
  vdb_print
  vdb_render
  DESTINATION
  bin
  )

INSTALL ( TARGETS
  openvdb_static
  DESTINATION
  lib
  )

IF (WIN32)
  INSTALL ( TARGETS
	openvdb_shared
	RUNTIME DESTINATION bin COMPONENT Runtime
	ARCHIVE DESTINATION lib COMPONENT Runtime
	)
ELSE()
  INSTALL ( TARGETS
	openvdb_shared
	DESTINATION
	lib
	)
ENDIF ()

INSTALL ( DIRECTORY ${PROJECT_SOURCE_DIR} DESTINATION include
  FILES_MATCHING PATTERN "*.h"
  PATTERN "python" EXCLUDE
  PATTERN "cmd" EXCLUDE
  PATTERN "doc" EXCLUDE
  )
