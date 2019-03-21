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

#include <cppunit/extensions/HelperMacros.h>
#include <openvdb/openvdb.h>
#include <openvdb/math/Maps.h> // for math::NonlinearFrustumMap
#include <openvdb/tools/Clip.h>


// See also TestGrid::testClipping()
class TestClip: public CppUnit::TestFixture
{
public:
    static const openvdb::CoordBBox kCubeBBox, kInnerBBox;

    TestClip(): mCube{
        []() {
            auto cube = openvdb::FloatGrid{0.0f};
            cube.fill(kCubeBBox, /*value=*/5.0f, /*active=*/true);
            return cube;
        }()}
    {}

    void setUp() override;
    void tearDown() override;

    CPPUNIT_TEST_SUITE(TestClip);
    CPPUNIT_TEST(testBBox);
    CPPUNIT_TEST(testFrustum);
    CPPUNIT_TEST(testMaskGrid);
    CPPUNIT_TEST(testBoolMask);
    CPPUNIT_TEST(testInvertedBoolMask);
    CPPUNIT_TEST(testNonBoolMask);
    CPPUNIT_TEST(testInvertedNonBoolMask);
    CPPUNIT_TEST_SUITE_END();

    void testBBox();
    void testFrustum();
    void testMaskGrid();
    void testBoolMask();
    void testInvertedBoolMask();
    void testNonBoolMask();
    void testInvertedNonBoolMask();

private:
    void validate(const openvdb::FloatGrid&);

    const openvdb::FloatGrid mCube;
};

const openvdb::CoordBBox
    // The volume to be clipped is a 21 x 21 x 21 solid cube.
    TestClip::kCubeBBox{openvdb::Coord{-10}, openvdb::Coord{10}},
    // The clipping mask is a 1 x 1 x 13 segment extending along the Z axis inside the cube.
    TestClip::kInnerBBox{openvdb::Coord{4, 4, -6}, openvdb::Coord{4, 4, 6}};

CPPUNIT_TEST_SUITE_REGISTRATION(TestClip);


////////////////////////////////////////


void
TestClip::setUp()
{
    openvdb::initialize();
}

void
TestClip::tearDown()
{
    openvdb::uninitialize();
}


void
TestClip::validate(const openvdb::FloatGrid& clipped)
{
    using namespace openvdb;

    const CoordBBox bbox = clipped.evalActiveVoxelBoundingBox();
    CPPUNIT_ASSERT_EQUAL(kInnerBBox.min().x(), bbox.min().x());
    CPPUNIT_ASSERT_EQUAL(kInnerBBox.min().y(), bbox.min().y());
    CPPUNIT_ASSERT_EQUAL(kInnerBBox.min().z(), bbox.min().z());
    CPPUNIT_ASSERT_EQUAL(kInnerBBox.max().x(), bbox.max().x());
    CPPUNIT_ASSERT_EQUAL(kInnerBBox.max().y(), bbox.max().y());
    CPPUNIT_ASSERT_EQUAL(kInnerBBox.max().z(), bbox.max().z());
    CPPUNIT_ASSERT_EQUAL(6 + 6 + 1, int(clipped.activeVoxelCount()));
    CPPUNIT_ASSERT_EQUAL(2, int(clipped.constTree().leafCount()));

    FloatGrid::ConstAccessor acc = clipped.getConstAccessor();
    const float bg = clipped.background();
    Coord xyz;
    int &x = xyz[0], &y = xyz[1], &z = xyz[2];
    for (x = kCubeBBox.min().x(); x <= kCubeBBox.max().x(); ++x) {
        for (y = kCubeBBox.min().y(); y <= kCubeBBox.max().y(); ++y) {
            for (z = kCubeBBox.min().z(); z <= kCubeBBox.max().z(); ++z) {
                if (x == 4 && y == 4 && z >= -6 && z <= 6) {
                    CPPUNIT_ASSERT_EQUAL(5.f, acc.getValue(Coord(4, 4, z)));
                } else {
                    CPPUNIT_ASSERT_EQUAL(bg, acc.getValue(Coord(x, y, z)));
                }
            }
        }
    }
}


////////////////////////////////////////


// Test clipping against a bounding box.
void
TestClip::testBBox()
{
    using namespace openvdb;
    BBoxd clipBox(Vec3d(4.0, 4.0, -6.0), Vec3d(4.9, 4.9, 6.0));
    FloatGrid::Ptr clipped = tools::clip(mCube, clipBox);
    validate(*clipped);
}


// Test clipping against a camera frustum.
void
TestClip::testFrustum()
{
    using namespace openvdb;

    const auto d = double(kCubeBBox.max().z());
    const math::NonlinearFrustumMap frustum{
        /*position=*/Vec3d{0.0, 0.0, 5.0 * d},
        /*direction=*/Vec3d{0.0, 0.0, -1.0},
        /*up=*/Vec3d{0.0, d / 2.0, 0.0},
        /*aspect=*/1.0,
        /*near=*/4.0 * d + 1.0,
        /*depth=*/kCubeBBox.dim().z() - 2.0,
        /*x_count=*/100,
        /*z_count=*/100};
    const auto frustumIndexBBox = frustum.getBBox();

    {
        auto clipped = tools::clip(mCube, frustum);

        const auto bbox = clipped->evalActiveVoxelBoundingBox();
        const auto cubeDim = kCubeBBox.dim();
        CPPUNIT_ASSERT_EQUAL(kCubeBBox.min().z() + 1, bbox.min().z());
        CPPUNIT_ASSERT_EQUAL(kCubeBBox.max().z() - 1, bbox.max().z());
        CPPUNIT_ASSERT(int(bbox.volume()) < int(cubeDim.x() * cubeDim.y() * (cubeDim.z() - 2)));

        // Note: mCube index space corresponds to world space.
        for (auto it = clipped->beginValueOn(); it; ++it) {
            const auto xyz = frustum.applyInverseMap(it.getCoord().asVec3d());
            CPPUNIT_ASSERT(frustumIndexBBox.isInside(xyz));
        }
    }
    {
        auto tile = openvdb::FloatGrid{0.0f};
        tile.tree().addTile(/*level=*/2, Coord{0}, /*value=*/5.0f, /*active=*/true);

        auto clipped = tools::clip(tile, frustum);
        CPPUNIT_ASSERT(!clipped->empty());
        for (auto it = clipped->beginValueOn(); it; ++it) {
            const auto xyz = frustum.applyInverseMap(it.getCoord().asVec3d());
            CPPUNIT_ASSERT(frustumIndexBBox.isInside(xyz));
        }

        clipped = tools::clip(tile, frustum, /*keepInterior=*/false);
        CPPUNIT_ASSERT(!clipped->empty());
        for (auto it = clipped->beginValueOn(); it; ++it) {
            const auto xyz = frustum.applyInverseMap(it.getCoord().asVec3d());
            CPPUNIT_ASSERT(!frustumIndexBBox.isInside(xyz));
        }
    }
}


// Test clipping against a MaskGrid.
void
TestClip::testMaskGrid()
{
    using namespace openvdb;
    MaskGrid mask(false);
    mask.fill(kInnerBBox, true, true);
    FloatGrid::Ptr clipped = tools::clip(mCube, mask);
    validate(*clipped);
}


// Test clipping against a boolean mask grid.
void
TestClip::testBoolMask()
{
    using namespace openvdb;
    BoolGrid mask(false);
    mask.fill(kInnerBBox, true, true);
    FloatGrid::Ptr clipped = tools::clip(mCube, mask);
    validate(*clipped);
}


// Test clipping against a boolean mask grid with mask inversion.
void
TestClip::testInvertedBoolMask()
{
    using namespace openvdb;
    // Construct a mask grid that is the "inverse" of the mask used in the other tests.
    // (This is not a true inverse, since the mask's active voxel bounds are finite.)
    BoolGrid mask(false);
    mask.fill(kCubeBBox, true, true);
    mask.fill(kInnerBBox, false, false);
    // Clipping against the "inverted" mask with mask inversion enabled
    // should give the same results as clipping normally against the normal mask.
    FloatGrid::Ptr clipped = tools::clip(mCube, mask, /*keepInterior=*/false);
    clipped->pruneGrid();
    validate(*clipped);
}


// Test clipping against a non-boolean mask grid.
void
TestClip::testNonBoolMask()
{
    using namespace openvdb;
    Int32Grid mask(0);
    mask.fill(kInnerBBox, -5, true);
    FloatGrid::Ptr clipped = tools::clip(mCube, mask);
    validate(*clipped);
}


// Test clipping against a non-boolean mask grid with mask inversion.
void
TestClip::testInvertedNonBoolMask()
{
    using namespace openvdb;
    // Construct a mask grid that is the "inverse" of the mask used in the other tests.
    // (This is not a true inverse, since the mask's active voxel bounds are finite.)
    Grid<UInt32Tree> mask(0);
    auto paddedCubeBBox = kCubeBBox;
    paddedCubeBBox.expand(2);
    mask.fill(paddedCubeBBox, 99, true);
    mask.fill(kInnerBBox, 0, false);
    // Clipping against the "inverted" mask with mask inversion enabled
    // should give the same results as clipping normally against the normal mask.
    FloatGrid::Ptr clipped = tools::clip(mCube, mask, /*keepInterior=*/false);
    clipped->pruneGrid();
    validate(*clipped);
}

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
