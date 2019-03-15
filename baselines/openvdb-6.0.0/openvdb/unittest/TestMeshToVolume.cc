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

#include <openvdb/tools/MeshToVolume.h>
#include <openvdb/util/Util.h>


class TestMeshToVolume: public CppUnit::TestCase
{
public:
    CPPUNIT_TEST_SUITE(TestMeshToVolume);
    CPPUNIT_TEST(testUtils);
    CPPUNIT_TEST(testConversion);
    CPPUNIT_TEST(testCreateLevelSetBox);
    CPPUNIT_TEST_SUITE_END();

    void testUtils();
    void testConversion();
    void testCreateLevelSetBox();

};

CPPUNIT_TEST_SUITE_REGISTRATION(TestMeshToVolume);


////////////////////////////////////////


void
TestMeshToVolume::testUtils()
{
    /// Test nearestCoord
    openvdb::Vec3d xyz(0.7, 2.2, -2.7);
    openvdb::Coord ijk = openvdb::util::nearestCoord(xyz);
    CPPUNIT_ASSERT(ijk[0] == 0 && ijk[1] == 2 && ijk[2] == -3);

    xyz = openvdb::Vec3d(-22.1, 4.6, 202.34);
    ijk = openvdb::util::nearestCoord(xyz);
    CPPUNIT_ASSERT(ijk[0] == -23 && ijk[1] == 4 && ijk[2] == 202);

    /// Test the coordinate offset table for neghbouring voxels
    openvdb::Coord sum(0, 0, 0);

    unsigned int pX = 0, pY = 0, pZ = 0, mX = 0, mY = 0, mZ = 0;

    for (unsigned int i = 0; i < 26; ++i) {
        ijk = openvdb::util::COORD_OFFSETS[i];
        sum += ijk;

        if (ijk[0] == 1)       ++pX;
        else if (ijk[0] == -1) ++mX;

        if (ijk[1] == 1)       ++pY;
        else if (ijk[1] == -1) ++mY;

        if (ijk[2] == 1)       ++pZ;
        else if (ijk[2] == -1) ++mZ;
    }

    CPPUNIT_ASSERT(sum == openvdb::Coord(0, 0, 0));

    CPPUNIT_ASSERT( pX == 9);
    CPPUNIT_ASSERT( pY == 9);
    CPPUNIT_ASSERT( pZ == 9);
    CPPUNIT_ASSERT( mX == 9);
    CPPUNIT_ASSERT( mY == 9);
    CPPUNIT_ASSERT( mZ == 9);
}

void
TestMeshToVolume::testConversion()
{
    using namespace openvdb;

    std::vector<Vec3s> points;
    std::vector<Vec4I> quads;

    // cube vertices
    points.push_back(Vec3s(2, 2, 2)); // 0       6--------7
    points.push_back(Vec3s(5, 2, 2)); // 1      /|       /|
    points.push_back(Vec3s(2, 5, 2)); // 2     2--------3 |
    points.push_back(Vec3s(5, 5, 2)); // 3     | |      | |
    points.push_back(Vec3s(2, 2, 5)); // 4     | 4------|-5
    points.push_back(Vec3s(5, 2, 5)); // 5     |/       |/
    points.push_back(Vec3s(2, 5, 5)); // 6     0--------1
    points.push_back(Vec3s(5, 5, 5)); // 7

    // cube faces
    quads.push_back(Vec4I(0, 1, 3, 2)); // front
    quads.push_back(Vec4I(5, 4, 6, 7)); // back
    quads.push_back(Vec4I(0, 2, 6, 4)); // left
    quads.push_back(Vec4I(1, 5, 7, 3)); // right
    quads.push_back(Vec4I(2, 3, 7, 6)); // top
    quads.push_back(Vec4I(0, 4, 5, 1)); // bottom

    math::Transform::Ptr xform = math::Transform::createLinearTransform();

    tools::QuadAndTriangleDataAdapter<Vec3s, Vec4I> mesh(points, quads);

    FloatGrid::Ptr grid = tools::meshToVolume<FloatGrid>(mesh, *xform);

    CPPUNIT_ASSERT(grid.get() != NULL);
    CPPUNIT_ASSERT_EQUAL(int(GRID_LEVEL_SET), int(grid->getGridClass()));
    CPPUNIT_ASSERT_EQUAL(1, int(grid->baseTree().leafCount()));

    grid = tools::meshToLevelSet<FloatGrid>(*xform, points, quads);

    CPPUNIT_ASSERT(grid.get() != NULL);
    CPPUNIT_ASSERT_EQUAL(int(GRID_LEVEL_SET), int(grid->getGridClass()));
    CPPUNIT_ASSERT_EQUAL(1, int(grid->baseTree().leafCount()));
}


void
TestMeshToVolume::testCreateLevelSetBox()
{
    typedef openvdb::FloatGrid          FloatGrid;
    typedef openvdb::Vec3s              Vec3s;
    typedef openvdb::math::BBox<Vec3s>  BBoxs;
    typedef openvdb::math::Transform    Transform;

    BBoxs bbox(Vec3s(0.0, 0.0, 0.0), Vec3s(1.0, 1.0, 1.0));

    Transform::Ptr transform = Transform::createLinearTransform(0.1);

    FloatGrid::Ptr grid = openvdb::tools::createLevelSetBox<FloatGrid>(bbox, *transform);

    double gridBackground = grid->background();
    double expectedBackground = transform->voxelSize().x() * double(openvdb::LEVEL_SET_HALF_WIDTH);

    CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedBackground, gridBackground, 1e-6);

    CPPUNIT_ASSERT(grid->tree().leafCount() > 0);

    // test inside coord value
    openvdb::Coord ijk = transform->worldToIndexNodeCentered(openvdb::Vec3d(0.5, 0.5, 0.5));
    CPPUNIT_ASSERT(grid->tree().getValue(ijk) < 0.0f);

    // test outside coord value
    ijk = transform->worldToIndexNodeCentered(openvdb::Vec3d(1.5, 1.5, 1.5));
    CPPUNIT_ASSERT(grid->tree().getValue(ijk) > 0.0f);
}


// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
