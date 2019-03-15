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
#include <openvdb/points/PointDataGrid.h>
#include <openvdb/points/PointConversion.h>
#include <openvdb/points/PointMask.h>
#include <algorithm>
#include <string>
#include <vector>

using namespace openvdb;
using namespace openvdb::points;

class TestPointMask: public CppUnit::TestCase
{
public:

    void setUp() override { openvdb::initialize(); }
    void tearDown() override { openvdb::uninitialize(); }

    CPPUNIT_TEST_SUITE(TestPointMask);
    CPPUNIT_TEST(testMask);
    CPPUNIT_TEST(testMaskDeformer);
    CPPUNIT_TEST_SUITE_END();

    void testMask();
    void testMaskDeformer();

}; // class TestPointMask


void
TestPointMask::testMask()
{
    std::vector<Vec3s> positions =  {
                                        {1, 1, 1},
                                        {1, 5, 1},
                                        {2, 1, 1},
                                        {2, 2, 1},
                                    };

    const PointAttributeVector<Vec3s> pointList(positions);

    const float voxelSize = 0.1f;
    openvdb::math::Transform::Ptr transform(
        openvdb::math::Transform::createLinearTransform(voxelSize));

    tools::PointIndexGrid::Ptr pointIndexGrid =
        tools::createPointIndexGrid<tools::PointIndexGrid>(pointList, *transform);

    PointDataGrid::Ptr points =
            createPointDataGrid<NullCodec, PointDataGrid>(*pointIndexGrid,
                                                          pointList, *transform);

    { // simple topology copy
        auto mask = convertPointsToMask(*points);

        CPPUNIT_ASSERT_EQUAL(points->tree().activeVoxelCount(), Index64(4));
        CPPUNIT_ASSERT_EQUAL(mask->tree().activeVoxelCount(), Index64(4));
    }

    { // mask grid instead of bool grid
        auto mask = convertPointsToMask<PointDataGrid, MaskGrid>(*points);

        CPPUNIT_ASSERT_EQUAL(points->tree().activeVoxelCount(), Index64(4));
        CPPUNIT_ASSERT_EQUAL(mask->tree().activeVoxelCount(), Index64(4));
    }

    { // identical transform
        auto mask = convertPointsToMask(*points, *transform);

        CPPUNIT_ASSERT_EQUAL(points->tree().activeVoxelCount(), Index64(4));
        CPPUNIT_ASSERT_EQUAL(mask->tree().activeVoxelCount(), Index64(4));
    }

    // assign point 3 to new group "test"

    appendGroup(points->tree(), "test");

    std::vector<short> groups{0,0,1,0};

    setGroup(points->tree(), pointIndexGrid->tree(), groups, "test");

    std::vector<std::string> includeGroups{"test"};
    std::vector<std::string> excludeGroups;

    { // convert in turn "test" and not "test"
        MultiGroupFilter filter(includeGroups, excludeGroups,
            points->tree().cbeginLeaf()->attributeSet());
        auto mask = convertPointsToMask(*points, filter);

        CPPUNIT_ASSERT_EQUAL(points->tree().activeVoxelCount(), Index64(4));
        CPPUNIT_ASSERT_EQUAL(mask->tree().activeVoxelCount(), Index64(1));

        MultiGroupFilter filter2(excludeGroups, includeGroups,
            points->tree().cbeginLeaf()->attributeSet());
        mask = convertPointsToMask(*points, filter2);

        CPPUNIT_ASSERT_EQUAL(mask->tree().activeVoxelCount(), Index64(3));
    }

    { // use a much larger voxel size that splits the points into two regions
        const float newVoxelSize(2);
        openvdb::math::Transform::Ptr newTransform(
            openvdb::math::Transform::createLinearTransform(newVoxelSize));

        auto mask = convertPointsToMask(*points, *newTransform);

        CPPUNIT_ASSERT_EQUAL(mask->tree().activeVoxelCount(), Index64(2));

        MultiGroupFilter filter(includeGroups, excludeGroups,
            points->tree().cbeginLeaf()->attributeSet());
        mask = convertPointsToMask(*points, *newTransform, filter);

        CPPUNIT_ASSERT_EQUAL(mask->tree().activeVoxelCount(), Index64(1));

        MultiGroupFilter filter2(excludeGroups, includeGroups,
            points->tree().cbeginLeaf()->attributeSet());
        mask = convertPointsToMask(*points, *newTransform, filter2);

        CPPUNIT_ASSERT_EQUAL(mask->tree().activeVoxelCount(), Index64(2));
    }
}


struct StaticVoxelDeformer
{
    StaticVoxelDeformer(const Vec3d& position)
        : mPosition(position) { }

    template <typename LeafT>
    void reset(LeafT& /*leaf*/, size_t /*idx*/) { }

    template <typename IterT>
    void apply(Vec3d& position, IterT&) const { position = mPosition; }

private:
    Vec3d mPosition;
};

template <bool WorldSpace = true>
struct YOffsetDeformer
{
    YOffsetDeformer(const Vec3d& offset) : mOffset(offset) { }

    template <typename LeafT>
    void reset(LeafT& /*leaf*/, size_t /*idx*/) { }

    template <typename IterT>
    void apply(Vec3d& position, IterT&) const { position += mOffset; }

    Vec3d mOffset;
};

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace points {

// configure both voxel deformers to be applied in index-space

template<>
struct DeformerTraits<StaticVoxelDeformer> {
    static const bool IndexSpace = true;
};

template<>
struct DeformerTraits<YOffsetDeformer<false>> {
    static const bool IndexSpace = true;
};

} // namespace points
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb


void
TestPointMask::testMaskDeformer()
{
    // This test validates internal functionality that is used in various applications, such as
    // building masks and producing count grids. Note that by convention, methods that live
    // in an "internal" namespace are typically not promoted as part of the public API
    // and thus do not receive the same level of rigour in avoiding breaking API changes.

    std::vector<Vec3s> positions =  {
                                        {1, 1, 1},
                                        {1, 5, 1},
                                        {2, 1, 1},
                                        {2, 2, 1},
                                    };

    const PointAttributeVector<Vec3s> pointList(positions);

    const float voxelSize = 0.1f;
    openvdb::math::Transform::Ptr transform(
        openvdb::math::Transform::createLinearTransform(voxelSize));

    tools::PointIndexGrid::Ptr pointIndexGrid =
        tools::createPointIndexGrid<tools::PointIndexGrid>(pointList, *transform);

    PointDataGrid::Ptr points =
            createPointDataGrid<NullCodec, PointDataGrid>(*pointIndexGrid,
                                                          pointList, *transform);

    // assign point 3 to new group "test"

    appendGroup(points->tree(), "test");

    std::vector<short> groups{0,0,1,0};

    setGroup(points->tree(), pointIndexGrid->tree(), groups, "test");

    NullFilter nullFilter;

    { // null deformer
        NullDeformer deformer;

        auto mask = point_mask_internal::convertPointsToScalar<MaskGrid>(
            *points, *transform, nullFilter, deformer);

        auto mask2 = convertPointsToMask(*points);

        CPPUNIT_ASSERT_EQUAL(points->tree().activeVoxelCount(), Index64(4));
        CPPUNIT_ASSERT_EQUAL(mask->tree().activeVoxelCount(), Index64(4));
        CPPUNIT_ASSERT(mask->tree().hasSameTopology(mask2->tree()));
        CPPUNIT_ASSERT(mask->tree().hasSameTopology(points->tree()));
    }

    { // static voxel deformer
        // collapse all points into a random voxel at (9, 13, 106)
        StaticVoxelDeformer deformer(Vec3d(9, 13, 106));

        auto mask = point_mask_internal::convertPointsToScalar<MaskGrid>(
            *points, *transform, nullFilter, deformer);

        CPPUNIT_ASSERT_EQUAL(mask->tree().activeVoxelCount(), Index64(1));
        CPPUNIT_ASSERT(!mask->tree().cbeginLeaf()->isValueOn(Coord(9, 13, 105)));
        CPPUNIT_ASSERT(mask->tree().cbeginLeaf()->isValueOn(Coord(9, 13, 106)));
    }

    { // +y offset deformer
        Vec3d offset(0, 41.7, 0);
        YOffsetDeformer</*world-space*/false> deformer(offset);

        auto mask = point_mask_internal::convertPointsToScalar<MaskGrid>(
            *points, *transform, nullFilter, deformer);

        // (repeat with deformer configured as world-space)
        YOffsetDeformer</*world-space*/true> deformerWS(offset * voxelSize);

        auto maskWS = point_mask_internal::convertPointsToScalar<MaskGrid>(
            *points, *transform, nullFilter, deformerWS);

        CPPUNIT_ASSERT_EQUAL(mask->tree().activeVoxelCount(), Index64(4));
        CPPUNIT_ASSERT_EQUAL(maskWS->tree().activeVoxelCount(), Index64(4));

        std::vector<Coord> maskVoxels;
        std::vector<Coord> maskVoxelsWS;
        std::vector<Coord> pointVoxels;

        for (auto leaf = mask->tree().cbeginLeaf(); leaf; ++leaf) {
            for (auto iter = leaf->cbeginValueOn(); iter; ++iter) {
                maskVoxels.emplace_back(iter.getCoord());
            }
        }

        for (auto leaf = maskWS->tree().cbeginLeaf(); leaf; ++leaf) {
            for (auto iter = leaf->cbeginValueOn(); iter; ++iter) {
                maskVoxelsWS.emplace_back(iter.getCoord());
            }
        }

        for (auto leaf = points->tree().cbeginLeaf(); leaf; ++leaf) {
            for (auto iter = leaf->cbeginValueOn(); iter; ++iter) {
                pointVoxels.emplace_back(iter.getCoord());
            }
        }

        std::sort(maskVoxels.begin(), maskVoxels.end());
        std::sort(maskVoxelsWS.begin(), maskVoxelsWS.end());
        std::sort(pointVoxels.begin(), pointVoxels.end());

        CPPUNIT_ASSERT_EQUAL(maskVoxels.size(), size_t(4));
        CPPUNIT_ASSERT_EQUAL(maskVoxelsWS.size(), size_t(4));
        CPPUNIT_ASSERT_EQUAL(pointVoxels.size(), size_t(4));

        for (int i = 0; i < int(pointVoxels.size()); i++) {
            Coord newCoord(pointVoxels[i]);
            newCoord.x() = static_cast<Int32>(newCoord.x() + offset.x());
            newCoord.y() = static_cast<Int32>(math::Round(newCoord.y() + offset.y()));
            newCoord.z() = static_cast<Int32>(newCoord.z() + offset.z());
            CPPUNIT_ASSERT_EQUAL(maskVoxels[i], newCoord);
            CPPUNIT_ASSERT_EQUAL(maskVoxelsWS[i], newCoord);
        }

        // use a different transform to verify deformers and transforms can be used together

        const float newVoxelSize = 0.02f;
        openvdb::math::Transform::Ptr newTransform(
            openvdb::math::Transform::createLinearTransform(newVoxelSize));

        auto mask2 = point_mask_internal::convertPointsToScalar<MaskGrid>(
            *points, *newTransform, nullFilter, deformer);

        CPPUNIT_ASSERT_EQUAL(mask2->tree().activeVoxelCount(), Index64(4));

        std::vector<Coord> maskVoxels2;

        for (auto leaf = mask2->tree().cbeginLeaf(); leaf; ++leaf) {
            for (auto iter = leaf->cbeginValueOn(); iter; ++iter) {
                maskVoxels2.emplace_back(iter.getCoord());
            }
        }

        std::sort(maskVoxels2.begin(), maskVoxels2.end());

        for (int i = 0; i < int(maskVoxels.size()); i++) {
            Coord newCoord(pointVoxels[i]);
            newCoord.x() = static_cast<Int32>((newCoord.x() + offset.x()) * 5);
            newCoord.y() = static_cast<Int32>(math::Round((newCoord.y() + offset.y()) * 5));
            newCoord.z() = static_cast<Int32>((newCoord.z() + offset.z()) * 5);
            CPPUNIT_ASSERT_EQUAL(maskVoxels2[i], newCoord);
        }

        // only use points in group "test"

        std::vector<std::string> includeGroups{"test"};
        std::vector<std::string> excludeGroups;
        MultiGroupFilter filter(includeGroups, excludeGroups,
            points->tree().cbeginLeaf()->attributeSet());

        auto mask3 = point_mask_internal::convertPointsToScalar<MaskGrid>(
            *points, *transform, filter, deformer);

        CPPUNIT_ASSERT_EQUAL(mask3->tree().activeVoxelCount(), Index64(1));

        for (auto leaf = mask3->tree().cbeginLeaf(); leaf; ++leaf) {
            for (auto iter = leaf->cbeginValueOn(); iter; ++iter) {
                Coord newCoord(pointVoxels[2]);
                newCoord.x() = static_cast<Int32>(newCoord.x() + offset.x());
                newCoord.y() = static_cast<Int32>(math::Round(newCoord.y() + offset.y()));
                newCoord.z() = static_cast<Int32>(newCoord.z() + offset.z());
                CPPUNIT_ASSERT_EQUAL(iter.getCoord(), newCoord);
            }
        }
    }
}


CPPUNIT_TEST_SUITE_REGISTRATION(TestPointMask);

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
