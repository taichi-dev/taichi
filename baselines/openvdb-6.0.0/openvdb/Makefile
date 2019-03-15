# Copyright (c) 2012-2018 DreamWorks Animation LLC
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
# Makefile for the OpenVDB library

# See INSTALL for a list of requirements.
#
# Targets:
#   lib                 the OpenVDB library
#
#   doc                 HTML documentation (doc/html/index.html)
#   pdfdoc              PDF documentation (doc/latex/refman.pdf;
#                       requires LaTeX and ghostscript)
#   python              OpenVDB Python module
#   pytest              unit tests for the Python module
#   pydoc               HTML documentation for the Python module
#                       (doc/html/python/index.html)
#   vdb_lod             command-line tool to generate volume mipmaps
#   vdb_print           command-line tool to inspect OpenVDB files
#   vdb_render          command-line tool to ray-trace OpenVDB files
#   vdb_view            command-line tool to view OpenVDB files
#   vdb_test            unit tests for the OpenVDB library
#
#   all                 [default target] all of the above
#   install             install all of the above except vdb_test
#                       into subdirectories of DESTDIR
#   install_lib         install just the OpenVDB library and its headers
#                       into subdirectories of DESTDIR
#   depend              recompute source file header dependencies
#   header_test         check for missing or indirectly included headers
#   clean               delete generated files from the local directory
#   test                run tests
#
# Options:
#   abi=N               build for compatibility with version N of the
#                       OpenVDB Grid ABI, where N is 2, 3, 4, etc.
#                       (some newer features will be disabled)
#   shared=no           link executables against static OpenVDB libraries
#                       (default: link against shared libraries)
#   debug=yes           build with debugging symbols and without optimization
#   verbose=yes         run commands (e.g., doxygen) in verbose mode


#
# The following variables must be defined, either here or on the command line
# (e.g., "make install DESTDIR=/usr/local"):
#
# Note that if you plan to build the Houdini OpenVDB tools (distributed
# separately), you must build the OpenVDB library and the Houdini tools
# against compatible versions of the Boost, OpenEXR and TBB libraries.
# Until Houdini 16.5, all three were included in the HDK, so the relevant
# variables below point by default to the HDK library and header directories:
# $(HDSO) and $(HT)/include, respectively.  (Source the houdini_setup script
# to set those two environment variables.)  Beginning with Houdini 16.5,
# you must provide your own distribution of Boost.
#
# To build the OpenVDB Python module, you will need local distributions of
# Python, Boost.Python, and optionally NumPy.  The Houdini HDK includes
# Python 2.x, but not the libboost_python library or NumPy or, as of
# Houdini 16.5, the Boost.Python headers, so both Boost.Python and NumPy
# have to be built separately.  Point the variables $(BOOST_PYTHON_LIB_DIR),
# $(BOOST_PYTHON_LIB) and $(NUMPY_INCL_DIR) below to your local distributions
# of those libraries.
#

# The directory into which to install libraries, executables and header files
DESTDIR := /tmp/OpenVDB

# The directory into which to install libraries (e.g., for Linux multiarch support)
DESTDIR_LIB_DIR := $(DESTDIR)/lib

# The parent directory of the boost/ header directory
BOOST_INCL_DIR := $(HT)/include
# The directory containing libboost_iostreams, libboost_system, etc.
BOOST_LIB_DIR := $(HDSO)
BOOST_LIB := -lboost_iostreams -lboost_system
BOOST_THREAD_LIB := -lboost_thread

# The parent directory of the OpenEXR/ header directory
EXR_INCL_DIR := $(HT)/include
# The directory containing IlmImf
EXR_LIB_DIR := $(HDSO)
EXR_LIB := -lIlmImf

# The parent directory of the OpenEXR/ header directory (which contains half.h)
ILMBASE_INCL_DIR := $(EXR_INCL_DIR)
# The directory containing libIlmThread, libIlmThread, libHalf etc.
ILMBASE_LIB_DIR := $(EXR_LIB_DIR)
ILMBASE_LIB := -lIlmThread -lIex -lImath
HALF_LIB := -lHalf

# The parent directory of the tbb/ header directory
TBB_INCL_DIR := $(HT)/include
# The directory containing libtbb
TBB_LIB_DIR := $(HDSO)
TBB_LIB := -ltbb

# The parent directory of the blosc.h header
# (leave blank if Blosc is unavailable)
BLOSC_INCL_DIR := $(HT)/include
# The directory containing libblosc
BLOSC_LIB_DIR := $(HDSO)
BLOSC_LIB := -lblosc

# A scalable, concurrent malloc replacement library
# such as jemalloc (included in the Houdini HDK) or TBB malloc
# (leave blank if unavailable)
CONCURRENT_MALLOC_LIB := -ljemalloc
#CONCURRENT_MALLOC_LIB := -ltbbmalloc_proxy -ltbbmalloc
# The directory containing the malloc replacement library
CONCURRENT_MALLOC_LIB_DIR := $(HDSO)

# The parent directory of the cppunit/ header directory
# (leave blank if CppUnit is unavailable)
CPPUNIT_INCL_DIR :=
# The directory containing libcppunit
CPPUNIT_LIB_DIR :=
CPPUNIT_LIB := -lcppunit

# The parent directory of the log4cplus/ header directory
# (leave blank if log4cplus is unavailable)
LOG4CPLUS_INCL_DIR := /rel/folio/log4cplus/log4cplus-1.1.2-latest/sys_include
# The directory containing liblog4cplus
LOG4CPLUS_LIB_DIR := /rel/folio/log4cplus/log4cplus-1.1.2-latest/library/c++11
LOG4CPLUS_LIB := -llog4cplus

# The directory containing glfw.h
# (leave blank if GLFW is unavailable)
GLFW_INCL_DIR := /rel/third_party/glfw/glfw-3.0.1/include
# The directory containing libglfw
GLFW_LIB_DIR := /rel/third_party/glfw/glfw-3.0.1/lib
GLFW_LIB := -lglfw
# The major version number of the GLFW library
# (header filenames changed between GLFW 2 and 3, so this must be specified explicitly)
GLFW_MAJOR_VERSION := 3

# The version of Python for which to build the OpenVDB module
# (leave blank if Python is unavailable)
PYTHON_VERSION := 2.7
# The directory containing Python.h
PYTHON_INCL_DIR := $(HFS)/python/include/python$(PYTHON_VERSION)
# The directory containing pyconfig.h
PYCONFIG_INCL_DIR := $(PYTHON_INCL_DIR)
# The directory containing libpython
PYTHON_LIB_DIR := $(HFS)/python/lib
PYTHON_LIB := -lpython$(PYTHON_VERSION)
# The directory containing libboost_python
BOOST_PYTHON_LIB_DIR := /rel/config/studio/baseline/boost/1.55.0.0/gcc-4.8.1.0/python-2.7.10.0/ext/lib/
BOOST_PYTHON_LIB := -lboost_python-mt
# The directory containing arrayobject.h
# (leave blank if NumPy is unavailable)
NUMPY_INCL_DIR := /rel/lang/python/2.7.9-1/lib/python2.7/site-packages/numpy/core/include/numpy/
# The Epydoc executable
# (leave blank if Epydoc is unavailable)
EPYDOC := /rel/map/generic_default-2014.24.237/bin/epydoc
# Set PYTHON_WRAP_ALL_GRID_TYPES to "yes" to specify that the Python module
# should expose (almost) all of the grid types defined in openvdb.h
# Otherwise, only FloatGrid, BoolGrid and Vec3SGrid will be exposed
# (see, e.g., exportIntGrid() in python/pyIntGrid.cc).
# Compiling the Python module with PYTHON_WRAP_ALL_GRID_TYPES set to "yes"
# can be very memory-intensive.
PYTHON_WRAP_ALL_GRID_TYPES := no

# The Doxygen executable
# (leave blank if Doxygen is unavailable)
DOXYGEN := doxygen


#
# Ideally, users shouldn't need to change anything below this line.
#

SHELL = /bin/bash

# Turn off implicit rules for speed.
.SUFFIXES:

ifneq (,$(INSTALL_DIR))
    $(warning Warning: $$(INSTALL_DIR) is no longer used; set $$(DESTDIR) instead.)
endif

# Determine the platform.
ifeq ("$(OS)","Windows_NT")
    WINDOWS_NT := 1
else
    UNAME_S := $(shell uname -s)
    ifeq ("$(UNAME_S)","Linux")
        LINUX := 1
    else
        ifeq ("$(UNAME_S)","Darwin")
            MBSD := 1
        endif
    endif
endif

ifeq (yes,$(strip $(debug)))
    OPTIMIZE := -g
else
    OPTIMIZE := -O3 -DNDEBUG
endif

ifeq (yes,$(strip $(verbose)))
    QUIET :=
    QUIET_TEST := -v
else
    QUIET := > /dev/null
    QUIET_TEST := $(QUIET)
endif

has_blosc := no
ifneq (,$(and $(BLOSC_LIB_DIR),$(BLOSC_INCL_DIR),$(BLOSC_LIB)))
    has_blosc := yes
endif

has_cppunit := no
ifneq (,$(and $(CPPUNIT_INCL_DIR),$(CPPUNIT_LIB_DIR),$(CPPUNIT_LIB)))
    has_cppunit := yes
endif

has_glfw := no
ifneq (,$(and $(GLFW_LIB_DIR),$(GLFW_INCL_DIR),$(GLFW_LIB)))
    has_glfw := yes
endif

has_log4cplus := no
ifneq (,$(and $(LOG4CPLUS_LIB_DIR),$(LOG4CPLUS_INCL_DIR),$(LOG4CPLUS_LIB)))
    has_log4cplus := yes
endif

has_python := no
ifneq (,$(and $(PYTHON_VERSION),$(PYTHON_LIB_DIR),$(PYTHON_INCL_DIR), \
    $(PYCONFIG_INCL_DIR),$(PYTHON_LIB),$(BOOST_PYTHON_LIB_DIR),$(BOOST_PYTHON_LIB)))
    has_python := yes
endif

INCLDIRS := -I . -I .. -isystem $(BOOST_INCL_DIR) -isystem $(ILMBASE_INCL_DIR) -isystem $(TBB_INCL_DIR)
ifeq (yes,$(has_blosc))
    INCLDIRS += -isystem $(BLOSC_INCL_DIR)
endif
ifeq (yes,$(has_log4cplus))
    INCLDIRS += -isystem $(LOG4CPLUS_INCL_DIR)
endif

CXXFLAGS += -std=c++11

CXXFLAGS += -pthread $(OPTIMIZE) $(INCLDIRS)
ifeq (yes,$(has_blosc))
    CXXFLAGS += -DOPENVDB_USE_BLOSC
endif
ifeq (yes,$(has_log4cplus))
    CXXFLAGS += -DOPENVDB_USE_LOG4CPLUS
endif
abi := $(strip $(abi))
ifneq (,$(abi))
    CXXFLAGS += -DOPENVDB_3_ABI_COMPATIBLE  # TODO: deprecated
    CXXFLAGS += -DOPENVDB_ABI_VERSION_NUMBER=$(abi)
endif
ifneq (2,$(strip $(GLFW_MAJOR_VERSION)))
    CXXFLAGS += -DOPENVDB_USE_GLFW_3
endif

LIBS := \
    -ldl -lm -lz \
    -L$(ILMBASE_LIB_DIR) $(HALF_LIB) \
    -L$(TBB_LIB_DIR) $(TBB_LIB) \
    -L$(BOOST_LIB_DIR) $(BOOST_LIB) \
#
LIBS_RPATH := \
    -ldl -lm -lz \
    -Wl,-rpath,$(ILMBASE_LIB_DIR) -L$(ILMBASE_LIB_DIR) $(HALF_LIB) \
    -Wl,-rpath,$(TBB_LIB_DIR) -L$(TBB_LIB_DIR) $(TBB_LIB) \
    -Wl,-rpath,$(BOOST_LIB_DIR) -L$(BOOST_LIB_DIR) $(BOOST_LIB) \
#
ifeq (yes,$(has_blosc))
    LIBS += -L$(BLOSC_LIB_DIR) $(BLOSC_LIB)
    LIBS_RPATH += -Wl,-rpath,$(BLOSC_LIB_DIR) -L$(BLOSC_LIB_DIR) $(BLOSC_LIB)
endif
ifeq (yes,$(has_log4cplus))
    LIBS += -L$(LOG4CPLUS_LIB_DIR) $(LOG4CPLUS_LIB)
    LIBS_RPATH += -Wl,-rpath,$(LOG4CPLUS_LIB_DIR) -L$(LOG4CPLUS_LIB_DIR) $(LOG4CPLUS_LIB)
endif
ifneq (,$(strip $(CONCURRENT_MALLOC_LIB)))
ifneq (,$(strip $(CONCURRENT_MALLOC_LIB_DIR)))
    LIBS_RPATH += -Wl,-rpath,$(CONCURRENT_MALLOC_LIB_DIR) -L$(CONCURRENT_MALLOC_LIB_DIR)
endif
endif
ifdef LINUX
    LIBS += -lrt
    LIBS_RPATH += -lrt
endif

INCLUDE_NAMES := \
    Exceptions.h \
    Grid.h \
    io/Archive.h \
    io/Compression.h \
    io/File.h \
    io/GridDescriptor.h \
    io/io.h \
    io/Queue.h \
    io/Stream.h \
    io/TempFile.h \
    math/BBox.h \
    math/ConjGradient.h \
    math/Coord.h \
    math/DDA.h \
    math/FiniteDifference.h \
    math/LegacyFrustum.h \
    math/Maps.h \
    math/Mat.h \
    math/Mat3.h \
    math/Mat4.h \
    math/Math.h \
    math/Operators.h \
    math/Proximity.h \
    math/QuantizedUnitVec.h \
    math/Quat.h \
    math/Ray.h \
    math/Stats.h \
    math/Stencils.h \
    math/Transform.h\
    math/Tuple.h\
    math/Vec2.h \
    math/Vec3.h \
    math/Vec4.h \
    Metadata.h \
    MetaMap.h \
    openvdb.h \
    Platform.h \
    PlatformConfig.h \
    points/AttributeArray.h \
    points/AttributeArrayString.h \
    points/AttributeGroup.h \
    points/AttributeSet.h \
    points/IndexFilter.h \
    points/IndexIterator.h \
    points/PointAdvect.h \
    points/PointAttribute.h \
    points/PointConversion.h \
    points/PointCount.h \
    points/PointDataGrid.h \
    points/PointDelete.h \
    points/PointGroup.h \
    points/PointMask.h \
    points/PointMove.h \
    points/PointSample.h \
    points/PointScatter.h \
    points/StreamCompression.h \
    tools/ChangeBackground.h \
    tools/Clip.h \
    tools/Composite.h \
    tools/Dense.h \
    tools/DenseSparseTools.h \
    tools/Diagnostics.h \
    tools/Filter.h \
    tools/GridOperators.h \
    tools/GridTransformer.h \
    tools/Interpolation.h \
    tools/LevelSetAdvect.h \
    tools/LevelSetFilter.h \
    tools/LevelSetFracture.h \
    tools/LevelSetMeasure.h \
    tools/LevelSetMorph.h \
    tools/LevelSetPlatonic.h \
    tools/LevelSetRebuild.h \
    tools/LevelSetSphere.h \
    tools/LevelSetTracker.h \
    tools/LevelSetUtil.h \
    tools/Mask.h \
    tools/MeshToVolume.h \
    tools/Morphology.h \
    tools/MultiResGrid.h \
    tools/ParticleAtlas.h \
    tools/ParticlesToLevelSet.h \
    tools/PointAdvect.h \
    tools/PointIndexGrid.h \
    tools/PointPartitioner.h \
    tools/PointScatter.h \
    tools/PointsToMask.h \
    tools/PoissonSolver.h \
    tools/PotentialFlow.h \
    tools/Prune.h \
    tools/RayIntersector.h \
    tools/RayTracer.h \
    tools/SignedFloodFill.h \
    tools/Statistics.h \
    tools/TopologyToLevelSet.h \
    tools/ValueTransformer.h \
    tools/VectorTransformer.h \
    tools/VelocityFields.h \
    tools/VolumeAdvect.h \
    tools/VolumeToMesh.h \
    tools/VolumeToSpheres.h \
    tree/InternalNode.h \
    tree/Iterator.h \
    tree/LeafBuffer.h \
    tree/LeafManager.h \
    tree/LeafNode.h \
    tree/LeafNodeBool.h \
    tree/LeafNodeMask.h \
    tree/NodeManager.h \
    tree/NodeUnion.h \
    tree/RootNode.h \
    tree/Tree.h \
    tree/TreeIterator.h \
    tree/ValueAccessor.h \
    Types.h \
    util/CpuTimer.h \
    util/Formats.h \
    util/logging.h \
    util/MapsUtil.h \
    util/Name.h \
    util/NodeMasks.h \
    util/NullInterrupter.h \
    util/PagedArray.h \
    util/Util.h \
    version.h \
#

SRC_NAMES := \
    Grid.cc \
    io/Archive.cc \
    io/Compression.cc \
    io/File.cc \
    io/GridDescriptor.cc \
    io/Queue.cc \
    io/Stream.cc \
    io/TempFile.cc \
    math/Maps.cc \
    math/Proximity.cc \
    math/QuantizedUnitVec.cc \
    math/Transform.cc \
    Metadata.cc \
    MetaMap.cc \
    openvdb.cc \
    Platform.cc \
    points/AttributeArray.cc \
    points/AttributeArrayString.cc \
    points/AttributeGroup.cc \
    points/AttributeSet.cc \
    points/points.cc \
    points/StreamCompression.cc \
    util/Formats.cc \
    util/Util.cc \
#

UNITTEST_INCLUDE_NAMES := \
    unittest/util.h \
#

UNITTEST_SRC_NAMES := \
    unittest/main.cc \
    unittest/TestAttributeArray.cc \
    unittest/TestAttributeArrayString.cc \
    unittest/TestAttributeSet.cc \
    unittest/TestAttributeGroup.cc \
    unittest/TestBBox.cc \
    unittest/TestConjGradient.cc \
    unittest/TestCoord.cc \
    unittest/TestCpt.cc \
    unittest/TestCurl.cc \
    unittest/TestDense.cc \
    unittest/TestDenseSparseTools.cc \
    unittest/TestDiagnostics.cc \
    unittest/TestDivergence.cc \
    unittest/TestDoubleMetadata.cc \
    unittest/TestExceptions.cc \
    unittest/TestFile.cc \
    unittest/TestFloatMetadata.cc \
    unittest/TestGradient.cc \
    unittest/TestGrid.cc \
    unittest/TestGridBbox.cc \
    unittest/TestGridDescriptor.cc \
    unittest/TestGridIO.cc \
    unittest/TestGridTransformer.cc \
    unittest/TestIndexFilter.cc \
    unittest/TestIndexIterator.cc \
    unittest/TestInit.cc \
    unittest/TestInt32Metadata.cc \
    unittest/TestInt64Metadata.cc \
    unittest/TestInternalOrigin.cc \
    unittest/TestLaplacian.cc \
    unittest/TestLeaf.cc \
    unittest/TestLeafBool.cc \
    unittest/TestLeafManager.cc \
    unittest/TestLeafMask.cc \
    unittest/TestLeafIO.cc \
    unittest/TestLeafOrigin.cc \
    unittest/TestLevelSetRayIntersector.cc \
    unittest/TestLevelSetUtil.cc \
    unittest/TestLinearInterp.cc \
    unittest/TestMaps.cc \
    unittest/TestMat4Metadata.cc \
    unittest/TestMath.cc \
    unittest/TestMeanCurvature.cc \
    unittest/TestMeshToVolume.cc \
    unittest/TestMetadata.cc \
    unittest/TestMetadataIO.cc \
    unittest/TestMetaMap.cc \
    unittest/TestMultiResGrid.cc \
    unittest/TestName.cc \
    unittest/TestNodeIterator.cc \
    unittest/TestNodeManager.cc \
    unittest/TestNodeMask.cc \
    unittest/TestParticleAtlas.cc \
    unittest/TestParticlesToLevelSet.cc \
    unittest/TestPointAdvect.cc \
    unittest/TestPointAttribute.cc \
    unittest/TestPointConversion.cc \
    unittest/TestPointCount.cc \
    unittest/TestPointDataLeaf.cc \
    unittest/TestPointDelete.cc \
    unittest/TestPointGroup.cc \
    unittest/TestPointIndexGrid.cc \
    unittest/TestPointMask.cc \
    unittest/TestPointMove.cc \
    unittest/TestPointPartitioner.cc \
    unittest/TestPointSample.cc \
    unittest/TestPointScatter.cc \
    unittest/TestPointsToMask.cc \
    unittest/TestPoissonSolver.cc \
    unittest/TestPotentialFlow.cc \
    unittest/TestPrePostAPI.cc \
    unittest/TestQuadraticInterp.cc \
    unittest/TestQuantizedUnitVec.cc \
    unittest/TestQuat.cc \
    unittest/TestRay.cc \
    unittest/TestStats.cc \
    unittest/TestStream.cc \
    unittest/TestStreamCompression.cc \
    unittest/TestStringMetadata.cc \
    unittest/TestTools.cc \
    unittest/TestTopologyToLevelSet.cc \
    unittest/TestTransform.cc \
    unittest/TestTree.cc \
    unittest/TestTreeCombine.cc \
    unittest/TestTreeGetSetValues.cc \
    unittest/TestTreeIterators.cc \
    unittest/TestTreeVisitor.cc \
    unittest/TestUtil.cc \
    unittest/TestValueAccessor.cc \
    unittest/TestVec2Metadata.cc \
    unittest/TestVec3Metadata.cc \
    unittest/TestVolumeRayIntersector.cc \
    unittest/TestVolumeToMesh.cc \
    unittest/TestVolumeToSpheres.cc \
#

DOC_FILES := doc/doc.txt doc/faq.txt doc/changes.txt doc/codingstyle.txt doc/examplecode.txt doc/api_0_98_0.txt doc/math.txt doc/points.txt doc/python.txt
DOC_INDEX := doc/html/index.html
DOC_PDF := doc/latex/refman.pdf

LIBVIEWER_INCLUDE_NAMES := \
    viewer/Camera.h \
    viewer/ClipBox.h \
    viewer/Font.h \
    viewer/RenderModules.h \
    viewer/Viewer.h \
#
# Used for "install" target only
LIBVIEWER_PUBLIC_INCLUDE_NAMES := \
    viewer/Viewer.h \
#
LIBVIEWER_SRC_NAMES := \
    viewer/Camera.cc \
    viewer/ClipBox.cc \
    viewer/Font.cc \
    viewer/RenderModules.cc \
    viewer/Viewer.cc \
#
ifdef MBSD
    LIBVIEWER_FLAGS := -framework Cocoa -framework OpenGL -framework IOKit
else
    LIBVIEWER_FLAGS := -lGL -lGLU
endif


CMD_INCLUDE_NAMES := \
#

CMD_SRC_NAMES := \
    cmd/openvdb_print/main.cc \
    cmd/openvdb_render/main.cc \
    cmd/openvdb_view/main.cc \
#


PYTHON_INCLUDE_NAMES := \
    python/pyopenvdb.h \
    python/pyutil.h \
    python/pyAccessor.h \
    python/pyGrid.h \
#
# Used for "install" target only
PYTHON_PUBLIC_INCLUDE_NAMES := \
    python/pyopenvdb.h \
#
PYTHON_SRC_NAMES := \
    python/pyFloatGrid.cc \
    python/pyIntGrid.cc \
    python/pyMetadata.cc \
    python/pyOpenVDBModule.cc \
    python/pyTransform.cc \
    python/pyVec3Grid.cc \
#
PYCXXFLAGS := -fPIC -isystem python -isystem $(PYTHON_INCL_DIR) -isystem $(PYCONFIG_INCL_DIR)
ifneq (,$(strip $(NUMPY_INCL_DIR)))
PYCXXFLAGS += -isystem $(NUMPY_INCL_DIR) -DPY_OPENVDB_USE_NUMPY
endif
ifneq (no,$(strip $(PYTHON_WRAP_ALL_GRID_TYPES)))
PYCXXFLAGS += -DPY_OPENVDB_WRAP_ALL_GRID_TYPES
endif


HEADER_SUBDIRS := $(dir $(INCLUDE_NAMES))

ALL_INCLUDE_FILES := \
    $(INCLUDE_NAMES) \
    $(UNITTEST_INCLUDE_NAMES) \
    $(CMD_INCLUDE_NAMES) \
    $(LIBVIEWER_INCLUDE_NAMES) \
    $(PYTHON_INCLUDE_NAMES) \
#
SRC_FILES := \
    $(SRC_NAMES) \
    $(UNITTEST_SRC_NAMES) \
    $(CMD_SRC_NAMES) \
    $(LIBVIEWER_SRC_NAMES) \
    $(PYTHON_SRC_NAMES) \
#
ALL_SRC_FILES := $(SRC_FILES)

OBJ_NAMES := $(SRC_NAMES:.cc=.o)
UNITTEST_OBJ_NAMES := $(UNITTEST_SRC_NAMES:.cc=.o)
LIBVIEWER_OBJ_NAMES := $(LIBVIEWER_SRC_NAMES:.cc=.o)
PYTHON_OBJ_NAMES := $(PYTHON_SRC_NAMES:.cc=.o)

LIB_MAJOR_VERSION=$(shell grep 'define OPENVDB_LIBRARY_MAJOR_VERSION_NUMBER ' \
    version.h | sed 's/[^0-9]*//g')
LIB_MINOR_VERSION=$(shell grep 'define OPENVDB_LIBRARY_MINOR_VERSION_NUMBER ' \
    version.h | sed 's/[^0-9]*//g')
LIB_PATCH_VERSION=$(shell grep 'define OPENVDB_LIBRARY_PATCH_VERSION_NUMBER ' \
    version.h | sed 's/[^0-9]*//g')

LIB_VERSION=$(LIB_MAJOR_VERSION).$(LIB_MINOR_VERSION).$(LIB_PATCH_VERSION)
SO_VERSION=$(LIB_MAJOR_VERSION).$(LIB_MINOR_VERSION)

LIBOPENVDB_NAME=libopenvdb
LIBOPENVDB_STATIC := $(LIBOPENVDB_NAME).a
ifndef MBSD
LIBOPENVDB_SHARED_NAME := $(LIBOPENVDB_NAME).so
LIBOPENVDB_SHARED := $(LIBOPENVDB_NAME).so.$(LIB_VERSION)
LIBOPENVDB_SONAME := $(LIBOPENVDB_NAME).so.$(SO_VERSION)
LIBOPENVDB_SONAME_FLAGS := -Wl,-soname,$(LIBOPENVDB_SONAME)
else
LIBOPENVDB_SHARED_NAME := $(LIBOPENVDB_NAME).dylib
LIBOPENVDB_SHARED := $(LIBOPENVDB_NAME).$(LIB_VERSION).dylib
LIBOPENVDB_SONAME := $(LIBOPENVDB_NAME).$(SO_VERSION).dylib
LIBOPENVDB_SONAME_FLAGS := -Wl,-install_name,$(DESTDIR_LIB_DIR)/$(LIBOPENVDB_SONAME)
endif

# TODO: libopenvdb_viewer is currently built into vdb_view and is not installed separately.
LIBVIEWER_NAME=libopenvdb_viewer
LIBVIEWER_STATIC := $(LIBVIEWER_NAME).a
ifndef MBSD
LIBVIEWER_SHARED_NAME := $(LIBVIEWER_NAME).so
LIBVIEWER_SHARED := $(LIBVIEWER_NAME).so.$(LIB_VERSION)
LIBVIEWER_SONAME := $(LIBVIEWER_NAME).so.$(SO_VERSION)
LIBVIEWER_SONAME_FLAGS := -Wl,-soname,$(LIBVIEWER_SONAME)
else
LIBVIEWER_SHARED_NAME := $(LIBVIEWER_NAME).dylib
LIBVIEWER_SHARED := $(LIBVIEWER_NAME).$(LIB_VERSION).dylib
LIBVIEWER_SONAME := $(LIBVIEWER_NAME).$(SO_VERSION).dylib
LIBVIEWER_SONAME_FLAGS := -Wl,-install_name,$(DESTDIR_LIB_DIR)/$(LIBVIEWER_SONAME)
endif

PYTHON_MODULE_NAME=pyopenvdb
PYTHON_MODULE := $(PYTHON_MODULE_NAME).so
PYTHON_SONAME := $(PYTHON_MODULE_NAME).so.$(SO_VERSION)
ifndef MBSD
PYTHON_SONAME_FLAGS := -Wl,-soname,$(PYTHON_SONAME)
endif

ifeq (no,$(strip $(shared)))
    LIBOPENVDB := $(LIBOPENVDB_STATIC)
    LIBVIEWER := $(LIBVIEWER_STATIC)
else
    LIBOPENVDB := $(LIBOPENVDB_SHARED)
    LIBVIEWER := $(LIBVIEWER_SHARED)
    LIBOPENVDB_RPATH := -Wl,-rpath,$(DESTDIR_LIB_DIR)
endif # shared

DEPEND := dependencies

# Get the list of dependencies that are newer than the current target,
# but limit the list to at most three entries.
list_deps = $(if $(wordlist 4,5,$(?F)),$(firstword $(?F)) and others,$(wordlist 1,3,$(?F)))

ALL_PRODUCTS := \
    $(LIBOPENVDB) \
    vdb_test \
    vdb_lod \
    vdb_print \
    vdb_render \
    vdb_view \
    $(DEPEND) \
    $(LIBOPENVDB_SHARED_NAME) \
    $(LIBOPENVDB_SONAME) \
    $(PYTHON_MODULE) \
#

.SUFFIXES: .o .cc

.PHONY: all clean depend doc header_test install lib pdfdoc pydoc pytest python test viewerlib

.cc.o:
	@echo "Building $@ because of $(call list_deps)"
	$(CXX) -c $(CXXFLAGS) -fPIC -o $@ $<

all: lib python vdb_lod vdb_print vdb_render vdb_test depend

$(OBJ_NAMES): %.o: %.cc
	@echo "Building $@ because of $(call list_deps)"
	$(CXX) -c -DOPENVDB_PRIVATE $(CXXFLAGS) -fPIC -o $@ $<

ifneq (no,$(strip $(shared)))

# Build shared library
lib: $(LIBOPENVDB_SHARED_NAME) $(LIBOPENVDB_SONAME)

$(LIBOPENVDB_SHARED_NAME): $(LIBOPENVDB_SHARED)
	ln -f -s $< $@

$(LIBOPENVDB_SONAME): $(LIBOPENVDB_SHARED)
	ln -f -s $< $@

$(LIBOPENVDB_SHARED): $(OBJ_NAMES)
	@echo "Building $@ because of $(list_deps)"
	$(CXX) $(CXXFLAGS) -shared -o $@ $^ $(LIBS_RPATH) $(LIBOPENVDB_SONAME_FLAGS)

else

# Build static library
lib: $(LIBOPENVDB)

$(LIBOPENVDB_STATIC): $(OBJ_NAMES)
	@echo "Building $@ because of $(list_deps)"
	$(AR) cr $@ $^

endif # shared


$(DOC_INDEX): doxygen-config $(INCLUDE_NAMES) $(SRC_NAMES) $(DOC_FILES)
	@echo "Generating documentation because of $(list_deps)"
	echo 'OUTPUT_DIRECTORY=./doc' | cat doxygen-config - | $(DOXYGEN) - $(QUIET)

$(DOC_PDF): doxygen-config $(INCLUDE_NAMES) $(SRC_NAMES) $(DOC_FILES)
	@echo "Generating documentation because of $(list_deps)"
	echo -e 'OUTPUT_DIRECTORY=./doc\nGENERATE_LATEX=YES\nGENERATE_HTML=NO' \
	    | cat doxygen-config - | $(DOXYGEN) - $(QUIET) \
	    && cd ./doc/latex && make refman.pdf $(QUIET) \
	    && echo 'Created doc/latex/refman.pdf'

ifneq (,$(strip $(DOXYGEN)))
doc: $(DOC_INDEX)
pdfdoc: $(DOC_PDF)
else
doc:
	@echo "$@"': $$DOXYGEN is undefined'
pdfdoc:
	@echo "$@"': $$DOXYGEN is undefined'
endif

vdb_lod: $(LIBOPENVDB) cmd/openvdb_lod/main.cc
	@echo "Building $@ because of $(list_deps)"
	$(CXX) $(CXXFLAGS) -o $@ cmd/openvdb_lod/main.cc -I . \
	    $(LIBOPENVDB_RPATH) -L$(CURDIR) $(LIBOPENVDB) \
	    $(LIBS_RPATH) $(CONCURRENT_MALLOC_LIB)

vdb_print: $(LIBOPENVDB) cmd/openvdb_print/main.cc
	@echo "Building $@ because of $(list_deps)"
	$(CXX) $(CXXFLAGS) -o $@ cmd/openvdb_print/main.cc -I . \
	    $(LIBOPENVDB_RPATH) -L$(CURDIR) $(LIBOPENVDB) \
	    $(LIBS_RPATH) $(CONCURRENT_MALLOC_LIB)

vdb_render: $(LIBOPENVDB) cmd/openvdb_render/main.cc
	@echo "Building $@ because of $(list_deps)"
	$(CXX) $(CXXFLAGS) -o $@ cmd/openvdb_render/main.cc -I . \
	    -isystem $(EXR_INCL_DIR) -isystem $(ILMBASE_INCL_DIR) \
	    -Wl,-rpath,$(EXR_LIB_DIR) -L$(EXR_LIB_DIR) $(EXR_LIB) \
	    -Wl,-rpath,$(ILMBASE_LIB_DIR) -L$(ILMBASE_LIB_DIR) $(ILMBASE_LIB) \
	    $(LIBOPENVDB_RPATH) -L$(CURDIR) $(LIBOPENVDB) \
	    $(LIBS_RPATH) $(CONCURRENT_MALLOC_LIB)

# Create an openvdb_viewer/ symlink to the viewer/ subdirectory,
# to mirror the DWA directory structure.
openvdb_viewer:
	ln -f -s viewer openvdb_viewer

ifneq (yes,$(has_glfw))
vdb_view:
	@echo "$@"': GLFW is unavailable'
else
$(LIBVIEWER_INCLUDE_NAMES): openvdb_viewer

$(LIBVIEWER_OBJ_NAMES): $(LIBVIEWER_INCLUDE_NAMES)
$(LIBVIEWER_OBJ_NAMES): %.o: %.cc
	@echo "Building $@ because of $(list_deps)"
	$(CXX) -c $(CXXFLAGS) -I . -isystem $(GLFW_INCL_DIR) -DGL_GLEXT_PROTOTYPES=1 -fPIC -o $@ $<

vdb_view: $(LIBOPENVDB) $(LIBVIEWER_OBJ_NAMES) cmd/openvdb_view/main.cc
	@echo "Building $@ because of $(list_deps)"
	$(CXX) $(CXXFLAGS) -o $@ cmd/openvdb_view/main.cc $(LIBVIEWER_OBJ_NAMES) \
	    -I . $(LIBOPENVDB_RPATH) -L$(CURDIR) $(LIBOPENVDB) \
	    $(LIBVIEWER_FLAGS) $(LIBS_RPATH) $(BOOST_THREAD_LIB) $(CONCURRENT_MALLOC_LIB) \
	    -Wl,-rpath,$(GLFW_LIB_DIR) -L$(GLFW_LIB_DIR) $(GLFW_LIB)
endif


# Build the Python module
$(PYTHON_OBJ_NAMES): $(PYTHON_INCLUDE_NAMES)
$(PYTHON_OBJ_NAMES): %.o: %.cc
	@echo "Building $@ because of $(list_deps)"
	$(CXX) -c $(CXXFLAGS) -I . $(PYCXXFLAGS) -o $@ $<
$(PYTHON_MODULE): $(LIBOPENVDB) $(PYTHON_OBJ_NAMES)
	@echo "Building $@ because of $(list_deps)"
	$(CXX) $(CXXFLAGS) $(PYCXXFLAGS) -shared $(PYTHON_SONAME_FLAGS) -o $@ $(PYTHON_OBJ_NAMES) \
	    -Wl,-rpath,$(PYTHON_LIB_DIR) -L$(PYTHON_LIB_DIR) $(PYTHON_LIB) \
	    -Wl,-rpath,$(BOOST_PYTHON_LIB_DIR) -L$(BOOST_PYTHON_LIB_DIR) $(BOOST_PYTHON_LIB) \
	    $(LIBOPENVDB_RPATH) -L$(CURDIR) $(LIBOPENVDB) \
	    $(LIBS_RPATH) $(CONCURRENT_MALLOC_LIB)

ifeq (yes,$(has_python))
ifneq (,$(strip $(EPYDOC)))
pydoc: $(PYTHON_MODULE) $(LIBOPENVDB_SONAME)
	@echo "Generating Python module documentation because of $(list_deps)"
	pydocdir=doc/html/python; \
	mkdir -p $${pydocdir}; \
	echo "Created $${pydocdir}"; \
	export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(CURDIR); \
	export PYTHONPATH=${PYTHONPATH}:$(CURDIR); \
	    $(EPYDOC) --html -o $${pydocdir} $(PYTHON_MODULE_NAME) $(QUIET)
else
pydoc:
	@echo "$@"': $$EPYDOC is undefined'
endif

pytest: $(PYTHON_MODULE) $(LIBOPENVDB_SONAME)
	@echo "Testing Python module $(PYTHON_MODULE)"
	export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(CURDIR); \
	export PYTHONPATH=${PYTHONPATH}:$(CURDIR); \
	    python$(PYTHON_VERSION) ./python/test/TestOpenVDB.py $(QUIET_TEST)

python: $(PYTHON_MODULE)
else
python pytest pydoc:
	@echo "$@"': Python is unavailable'
endif


$(UNITTEST_OBJ_NAMES): %.o: %.cc
	@echo "Building $@ because of $(list_deps)"
	$(CXX) -c $(CXXFLAGS) -isystem $(CPPUNIT_INCL_DIR) -fPIC -o $@ $<

ifneq (,$(strip $(CPPUNIT_INCL_DIR)))
vdb_test: $(LIBOPENVDB) $(UNITTEST_OBJ_NAMES)
	@echo "Building $@ because of $(list_deps)"
	$(CXX) $(CXXFLAGS) -o $@ $(UNITTEST_OBJ_NAMES) \
	    $(LIBOPENVDB_RPATH) -L$(CURDIR) $(LIBOPENVDB) \
	    $(LIBS_RPATH) $(CONCURRENT_MALLOC_LIB) \
	    -Wl,-rpath,$(CPPUNIT_LIB_DIR) -L$(CPPUNIT_LIB_DIR) $(CPPUNIT_LIB)

test: lib vdb_test
	@echo "Testing $(LIBOPENVDB_NAME)"
	export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(CURDIR); ./vdb_test $(QUIET_TEST)
else
vdb_test:
	@echo "$@"': $$(CPPUNIT_INCL_DIR) is undefined'
test:
	@echo "$@"': $$(CPPUNIT_INCL_DIR) is undefined'
endif

install_lib: lib
	mkdir -p $(DESTDIR)/include/openvdb
	@echo "Created $(DESTDIR)/include/openvdb"
	pushd $(DESTDIR)/include/openvdb > /dev/null; \
	    mkdir -p $(HEADER_SUBDIRS); popd > /dev/null
	for f in $(INCLUDE_NAMES); \
	    do cp -f $$f $(DESTDIR)/include/openvdb/$$f; done
	@echo "Copied header files to $(DESTDIR)/include"
	@#
	mkdir -p $(DESTDIR_LIB_DIR)
	@echo "Created $(DESTDIR_LIB_DIR)"
	cp -f $(LIBOPENVDB) $(DESTDIR_LIB_DIR)
	pushd $(DESTDIR_LIB_DIR) > /dev/null; \
	    if [ -f $(LIBOPENVDB_SHARED) ]; then \
	        ln -f -s $(LIBOPENVDB_SHARED) $(LIBOPENVDB_SHARED_NAME); \
	        ln -f -s $(LIBOPENVDB_SHARED) $(LIBOPENVDB_SONAME); \
	    fi; \
	    popd > /dev/null
	@echo "Copied libopenvdb to $(DESTDIR_LIB_DIR)"

install: install_lib python vdb_lod vdb_print vdb_render vdb_view doc pydoc
	if [ -f $(LIBVIEWER) ]; \
	then \
	    mkdir -p $(DESTDIR)/include/openvdb_viewer; \
	    echo "Created $(DESTDIR)/include/openvdb_viewer"; \
	    cp -f $(LIBVIEWER_PUBLIC_INCLUDE_NAMES) $(DESTDIR)/include/openvdb_viewer/; \
	    @echo "Copied vdb_view header files to $(DESTDIR)/include" \
	    cp -f $(LIBVIEWER) $(DESTDIR_LIB_DIR); \
	    pushd $(DESTDIR_LIB_DIR) > /dev/null; \
	        if [ -f $(LIBVIEWER_SHARED) ]; then \
	            ln -f -s $(LIBVIEWER_SHARED) $(LIBVIEWER_SHARED_NAME); fi; \
	        popd > /dev/null; \
	    @echo "Copied libopenvdb_viewer to $(DESTDIR_LIB_DIR)"; \
	fi
	@#
	if [ -f $(PYTHON_MODULE) ]; \
	then \
	    installdir=$(DESTDIR)/python/include/python$(PYTHON_VERSION); \
	    mkdir -p $${installdir}; \
	    echo "Created $${installdir}"; \
	    cp -f $(PYTHON_PUBLIC_INCLUDE_NAMES) $${installdir}/; \
	    echo "Copied Python header files to $${installdir}"; \
	    installdir=$(DESTDIR)/python/lib/python$(PYTHON_VERSION); \
	    mkdir -p $${installdir}; \
	    echo "Created $${installdir}"; \
	    cp -f $(PYTHON_MODULE) $${installdir}/; \
	    pushd $${installdir} > /dev/null; \
	    ln -f -s $(PYTHON_MODULE) $(PYTHON_SONAME); \
	    popd > /dev/null; \
	    echo "Copied Python module to $${installdir}"; \
	fi
	@#
	mkdir -p $(DESTDIR)/bin
	@echo "Created $(DESTDIR)/bin/"
	cp -f vdb_lod $(DESTDIR)/bin
	@echo "Copied vdb_lod to $(DESTDIR)/bin/"
	cp -f vdb_print $(DESTDIR)/bin
	@echo "Copied vdb_print to $(DESTDIR)/bin/"
	cp -f vdb_render $(DESTDIR)/bin
	@echo "Copied vdb_render to $(DESTDIR)/bin/"
	if [ -f vdb_view ]; \
	then \
	    cp -f vdb_view $(DESTDIR)/bin; \
	    echo "Copied vdb_view to $(DESTDIR)/bin/"; \
	fi
	@#
	if [ -d doc/html ]; \
	then \
	    mkdir -p $(DESTDIR)/share/doc/openvdb; \
	    echo "Created $(DESTDIR)/share/doc/openvdb/"; \
	    cp -r -f doc/html $(DESTDIR)/share/doc/openvdb; \
	    echo "Copied documentation to $(DESTDIR)/share/doc/openvdb/"; \
	fi

# TODO: This accumulates all source file dependencies into a single file
# containing a rule for each *.o file.  Consider generating a separate
# dependency file for each *.o file instead.
$(DEPEND): $(ALL_INCLUDE_FILES) $(ALL_SRC_FILES) openvdb_viewer
	@echo "Generating dependencies because of $(list_deps)"
	$(RM) $(DEPEND)
	for f in $(SRC_NAMES) $(CMD_SRC_NAMES); \
	    do $(CXX) $(CXXFLAGS) -O0 \
	        -MM $$f -MT `echo $$f | sed 's%\.[^.]*%.o%'` >> $(DEPEND); \
	done
	if [ -d "$(CPPUNIT_INCL_DIR)" ]; \
	then \
	    for f in $(UNITTEST_SRC_NAMES); \
	        do $(CXX) $(CXXFLAGS) -O0 \
	            -MM $$f -MT `echo $$f | sed 's%\.[^.]*%.o%'` \
	            -isystem $(CPPUNIT_INCL_DIR) >> $(DEPEND); \
	    done; \
	fi

depend: $(DEPEND)

# Compile an implicit translation unit for each header to identify any indirect includes
HEADER_TEST_CXXFLAGS := $(CXXFLAGS)
HEADER_TEST_FILES := $(addprefix header_test-,$(INCLUDE_NAMES) $(CMD_INCLUDE_NAMES))
ifeq (yes,$(has_cppunit))
    HEADER_TEST_FILES += $(addprefix header_test-,$(UNITTEST_INCLUDE_NAMES))
    HEADER_TEST_CXXFLAGS += -isystem $(CPPUNIT_INCL_DIR)
endif
ifeq (yes,$(has_glfw))
    HEADER_TEST_FILES += $(addprefix header_test-,$(LIBVIEWER_INCLUDE_NAMES))
    HEADER_TEST_CXXFLAGS += -isystem $(GLFW_INCL_DIR) -DGL_GLEXT_PROTOTYPES=1
endif
ifeq (yes,$(has_python))
    HEADER_TEST_FILES += $(addprefix header_test-,$(PYTHON_INCLUDE_NAMES))
    HEADER_TEST_CXXFLAGS += $(PYCXXFLAGS)
endif
$(HEADER_TEST_FILES): header_test-%:
	echo "#include \"$*\"" | $(CXX) -c -x c++ $(HEADER_TEST_CXXFLAGS) -fPIC -o /dev/null -
echo_header_test:
	@echo "Checking for missing or indirectly included headers"
header_test: echo_header_test $(HEADER_TEST_FILES)

clean:
	$(RM) $(OBJ_NAMES) $(ALL_PRODUCTS) $(DEPEND)
	$(RM) $(LIBOPENVDB_STATIC)
	$(RM) $(LIBOPENVDB_SHARED)
	$(RM) $(LIBVIEWER_OBJ_NAMES)
	$(RM) $(PYTHON_OBJ_NAMES)
	$(RM) $(UNITTEST_OBJ_NAMES)
	$(RM) -r ./doc/html ./doc/latex

ifneq (,$(strip $(wildcard $(DEPEND))))
    include $(DEPEND)
endif

# Copyright (c) 2012-2018 DreamWorks Animation LLC
# All rights reserved. This software is distributed under the
# Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
