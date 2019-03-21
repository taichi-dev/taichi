///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012-2018 DreamWorks Animation LLC
//
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
//
// Redistributions of source code must retain the above copyright
// and license notice and the following restrictions and disclaimer.
//
// *     Neither the name of DreamWorks Animation nor the names of
// its contributors may be used to endorse or promote products derived
// from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// IN NO EVENT SHALL THE COPYRIGHT HOLDERS' AND CONTRIBUTORS' AGGREGATE
// LIABILITY FOR ALL CLAIMS REGARDLESS OF THEIR BASIS EXCEED US$250.00.
//
///////////////////////////////////////////////////////////////////////////

#include <vector>
#include <cppunit/extensions/HelperMacros.h>

#include <openvdb/openvdb.h>
#include <openvdb/Exceptions.h>
#include <openvdb/tools/LevelSetUtil.h>
#include <openvdb/tools/MeshToVolume.h>     // for createLevelSetBox()
#include <openvdb/tools/Composite.h>        // for csgDifference()

class TestLevelSetUtil: public CppUnit::TestCase
{
public:
    CPPUNIT_TEST_SUITE(TestLevelSetUtil);
    CPPUNIT_TEST(testSDFToFogVolume);
    CPPUNIT_TEST(testSDFInteriorMask);
    CPPUNIT_TEST(testExtractEnclosedRegion);
    CPPUNIT_TEST(testSegmentationTools);
    CPPUNIT_TEST_SUITE_END();

    void testSDFToFogVolume();
    void testSDFInteriorMask();
    void testExtractEnclosedRegion();
    void testSegmentationTools();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestLevelSetUtil);


////////////////////////////////////////

void
TestLevelSetUtil::testSDFToFogVolume()
{
    openvdb::FloatGrid::Ptr grid = openvdb::FloatGrid::create(10.0);

    grid->fill(openvdb::CoordBBox(openvdb::Coord(-100), openvdb::Coord(100)), 9.0);
    grid->fill(openvdb::CoordBBox(openvdb::Coord(-50), openvdb::Coord(50)), -9.0);


    openvdb::tools::sdfToFogVolume(*grid);

    CPPUNIT_ASSERT(grid->background() < 1e-7);

    openvdb::FloatGrid::ValueOnIter iter = grid->beginValueOn();
    for (; iter; ++iter) {
        CPPUNIT_ASSERT(iter.getValue() > 0.0);
        CPPUNIT_ASSERT(std::abs(iter.getValue() - 1.0) < 1e-7);
    }
}


void
TestLevelSetUtil::testSDFInteriorMask()
{
    typedef openvdb::FloatGrid          FloatGrid;
    typedef openvdb::BoolGrid           BoolGrid;
    typedef openvdb::Vec3s              Vec3s;
    typedef openvdb::math::BBox<Vec3s>  BBoxs;
    typedef openvdb::math::Transform    Transform;

    BBoxs bbox(Vec3s(0.0, 0.0, 0.0), Vec3s(1.0, 1.0, 1.0));

    Transform::Ptr transform = Transform::createLinearTransform(0.1);

    FloatGrid::Ptr sdfGrid = openvdb::tools::createLevelSetBox<FloatGrid>(bbox, *transform);

    BoolGrid::Ptr maskGrid = openvdb::tools::sdfInteriorMask(*sdfGrid);

    // test inside coord value
    openvdb::Coord ijk = transform->worldToIndexNodeCentered(openvdb::Vec3d(0.5, 0.5, 0.5));
    CPPUNIT_ASSERT(maskGrid->tree().getValue(ijk) == true);

    // test outside coord value
    ijk = transform->worldToIndexNodeCentered(openvdb::Vec3d(1.5, 1.5, 1.5));
    CPPUNIT_ASSERT(maskGrid->tree().getValue(ijk) == false);
}


void
TestLevelSetUtil::testExtractEnclosedRegion()
{
    typedef openvdb::FloatGrid          FloatGrid;
    typedef openvdb::BoolGrid           BoolGrid;
    typedef openvdb::Vec3s              Vec3s;
    typedef openvdb::math::BBox<Vec3s>  BBoxs;
    typedef openvdb::math::Transform    Transform;

    BBoxs regionA(Vec3s(0.0f, 0.0f, 0.0f), Vec3s(3.0f, 3.0f, 3.0f));
    BBoxs regionB(Vec3s(1.0f, 1.0f, 1.0f), Vec3s(2.0f, 2.0f, 2.0f));

    Transform::Ptr transform = Transform::createLinearTransform(0.1);

    FloatGrid::Ptr sdfGrid = openvdb::tools::createLevelSetBox<FloatGrid>(regionA, *transform);
    FloatGrid::Ptr sdfGridB = openvdb::tools::createLevelSetBox<FloatGrid>(regionB, *transform);

    openvdb::tools::csgDifference(*sdfGrid, *sdfGridB);

    BoolGrid::Ptr maskGrid = openvdb::tools::extractEnclosedRegion(*sdfGrid);

    // test inside ls region coord value
    openvdb::Coord ijk = transform->worldToIndexNodeCentered(openvdb::Vec3d(1.5, 1.5, 1.5));
    CPPUNIT_ASSERT(maskGrid->tree().getValue(ijk) == true);

    // test outside coord value
    ijk = transform->worldToIndexNodeCentered(openvdb::Vec3d(3.5, 3.5, 3.5));
    CPPUNIT_ASSERT(maskGrid->tree().getValue(ijk) == false);
}


void
TestLevelSetUtil::testSegmentationTools()
{
    typedef openvdb::FloatGrid          FloatGrid;
    typedef openvdb::Vec3s              Vec3s;
    typedef openvdb::math::BBox<Vec3s>  BBoxs;
    typedef openvdb::math::Transform    Transform;

    { // Test SDF segmentation

        // Create two sdf boxes with overlapping narrow-bands.
        BBoxs regionA(Vec3s(0.0f, 0.0f, 0.0f), Vec3s(2.0f, 2.0f, 2.0f));
        BBoxs regionB(Vec3s(2.5f, 0.0f, 0.0f), Vec3s(4.3f, 2.0f, 2.0f));

        Transform::Ptr transform = Transform::createLinearTransform(0.1);

        FloatGrid::Ptr sdfGrid = openvdb::tools::createLevelSetBox<FloatGrid>(regionA, *transform);
        FloatGrid::Ptr sdfGridB = openvdb::tools::createLevelSetBox<FloatGrid>(regionB, *transform);

        openvdb::tools::csgUnion(*sdfGrid, *sdfGridB);

        std::vector<FloatGrid::Ptr> segments;

        // This tool will not identify two separate segments when the narrow-bands overlap. 
        openvdb::tools::segmentActiveVoxels(*sdfGrid, segments);
        CPPUNIT_ASSERT(segments.size() == 1);

        segments.clear();

        // This tool should properly identify two separate segments
        openvdb::tools::segmentSDF(*sdfGrid, segments);
        CPPUNIT_ASSERT(segments.size() == 2);


        // test inside ls region coord value
        openvdb::Coord ijk = transform->worldToIndexNodeCentered(openvdb::Vec3d(1.5, 1.5, 1.5));
        CPPUNIT_ASSERT(segments[0]->tree().getValue(ijk) < 0.0f);

        // test outside coord value
        ijk = transform->worldToIndexNodeCentered(openvdb::Vec3d(3.5, 3.5, 3.5));
        CPPUNIT_ASSERT(segments[0]->tree().getValue(ijk) > 0.0f);
    }

    { // Test fog volume with active tiles

        openvdb::FloatGrid::Ptr grid = openvdb::FloatGrid::create(0.0);

        grid->fill(openvdb::CoordBBox(openvdb::Coord(0), openvdb::Coord(50)), 1.0);
        grid->fill(openvdb::CoordBBox(openvdb::Coord(60), openvdb::Coord(100)), 1.0);

        CPPUNIT_ASSERT(grid->tree().hasActiveTiles() == true);

        std::vector<FloatGrid::Ptr> segments;
        openvdb::tools::segmentActiveVoxels(*grid, segments);
        CPPUNIT_ASSERT(segments.size() == 2);
    }
}


// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
