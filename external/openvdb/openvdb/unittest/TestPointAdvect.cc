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
#include "util.h"
#include <openvdb/points/PointAttribute.h>
#include <openvdb/points/PointDataGrid.h>
#include <openvdb/points/PointConversion.h>
#include <openvdb/points/PointScatter.h>
#include <openvdb/points/PointAdvect.h>
#include <openvdb/tools/LevelSetSphere.h>
#include <openvdb/tools/Composite.h> // csgDifference
#include <openvdb/tools/MeshToVolume.h> // createLevelSetBox
#include <openvdb/openvdb.h>
#include <openvdb/Types.h>
#include <string>
#include <vector>

using namespace openvdb;
using namespace openvdb::points;

class TestPointAdvect: public CppUnit::TestCase
{
public:

    void setUp() override { openvdb::initialize(); }
    void tearDown() override { openvdb::uninitialize(); }

    CPPUNIT_TEST_SUITE(TestPointAdvect);
    CPPUNIT_TEST(testAdvect);
    CPPUNIT_TEST(testZalesaksDisk);
    CPPUNIT_TEST_SUITE_END();

    void testAdvect();
    void testZalesaksDisk();
}; // class TestPointAdvect

CPPUNIT_TEST_SUITE_REGISTRATION(TestPointAdvect);


////////////////////////////////////////


void
TestPointAdvect::testAdvect()
{
    // generate four points

    const float voxelSize = 1.0f;
    std::vector<Vec3s> positions =  {
                                    {5, 2, 3},
                                    {2, 4, 1},
                                    {50, 5, 1},
                                    {3, 20, 1},
                                };

    const PointAttributeVector<Vec3s> pointList(positions);

    math::Transform::Ptr pointTransform(math::Transform::createLinearTransform(voxelSize));

    auto pointIndexGrid = tools::createPointIndexGrid<tools::PointIndexGrid>(
        pointList, *pointTransform);

    auto points = createPointDataGrid<NullCodec, PointDataGrid>(
        *pointIndexGrid, pointList, *pointTransform);

    std::vector<int> id;
    id.push_back(0);
    id.push_back(1);
    id.push_back(2);
    id.push_back(3);

    auto idAttributeType = TypedAttributeArray<int>::attributeType();
    appendAttribute(points->tree(), "id", idAttributeType);

    // create a wrapper around the id vector
    PointAttributeVector<int> idWrapper(id);

    populateAttribute<PointDataTree, tools::PointIndexTree, PointAttributeVector<int>>(
            points->tree(), pointIndexGrid->tree(), "id", idWrapper);

    // create "test" group which only contains third point

    appendGroup(points->tree(), "test");
    std::vector<short> groups(positions.size(), 0);
    groups[2] = 1;
    setGroup(points->tree(), pointIndexGrid->tree(), groups, "test");

    // create "test2" group which contains second and third point

    appendGroup(points->tree(), "test2");
    groups[1] = 1;
    setGroup(points->tree(), pointIndexGrid->tree(), groups, "test2");

    const Vec3s tolerance(1e-3f);

    // advect by velocity using all integration orders

    for (Index integrationOrder = 0; integrationOrder < 5; integrationOrder++) {
        Vec3s velocityBackground(1.0, 2.0, 3.0);
        double timeStep = 1.0;
        int steps = 1;
        auto velocity = Vec3SGrid::create(velocityBackground); // grid with background value only

        auto pointsToAdvect = points->deepCopy();
        const auto& transform = pointsToAdvect->transform();

        advectPoints(*pointsToAdvect, *velocity, integrationOrder, timeStep, steps);

        for (auto leaf = pointsToAdvect->tree().beginLeaf(); leaf; ++leaf) {
            AttributeHandle<Vec3s> positionHandle(leaf->constAttributeArray("P"));
            AttributeHandle<int> idHandle(leaf->constAttributeArray("id"));
            for (auto iter = leaf->beginIndexOn(); iter; ++iter) {
                int theId = idHandle.get(*iter);
                Vec3s position = transform.indexToWorld(
                    positionHandle.get(*iter) + iter.getCoord().asVec3d());
                Vec3s expectedPosition(positions[theId]);
                if (integrationOrder > 0)   expectedPosition += velocityBackground;
                CPPUNIT_ASSERT(math::isApproxEqual(position, expectedPosition, tolerance));
            }
        }
    }

    // invalid advection scheme

    auto zeroVelocityGrid = Vec3SGrid::create(Vec3s(0));
    CPPUNIT_ASSERT_THROW(advectPoints(*points, *zeroVelocityGrid, 5, 1.0, 1), ValueError);

    { // advect varying dt and steps
        Vec3s velocityBackground(1.0, 2.0, 3.0);
        Index integrationOrder = 4;
        double timeStep = 0.1;
        int steps = 100;
        auto velocity = Vec3SGrid::create(velocityBackground); // grid with background value only

        auto pointsToAdvect = points->deepCopy();
        const auto& transform = pointsToAdvect->transform();

        advectPoints(*pointsToAdvect, *velocity, integrationOrder, timeStep, steps);

        for (auto leaf = pointsToAdvect->tree().beginLeaf(); leaf; ++leaf) {
            AttributeHandle<Vec3s> positionHandle(leaf->constAttributeArray("P"));
            AttributeHandle<int> idHandle(leaf->constAttributeArray("id"));
            for (auto iter = leaf->beginIndexOn(); iter; ++iter) {
                int theId = idHandle.get(*iter);
                Vec3s position = transform.indexToWorld(
                    positionHandle.get(*iter) + iter.getCoord().asVec3d());
                Vec3s expectedPosition(positions[theId] + velocityBackground * 10);
                CPPUNIT_ASSERT(math::isApproxEqual(position, expectedPosition, tolerance));
            }
        }
    }

    { // perform filtered advection
        Vec3s velocityBackground(1.0, 2.0, 3.0);
        Index integrationOrder = 4;
        double timeStep = 1.0;
        int steps = 1;
        auto velocity = Vec3SGrid::create(velocityBackground); // grid with background value only

        std::vector<std::string> advectIncludeGroups;
        std::vector<std::string> advectExcludeGroups;
        std::vector<std::string> includeGroups;
        std::vector<std::string> excludeGroups;

        { // only advect points in "test" group
            advectIncludeGroups.push_back("test");

            auto leaf = points->tree().cbeginLeaf();
            MultiGroupFilter advectFilter(
                advectIncludeGroups, advectExcludeGroups, leaf->attributeSet());
            MultiGroupFilter filter(includeGroups, excludeGroups, leaf->attributeSet());

            auto pointsToAdvect = points->deepCopy();
            const auto& transform = pointsToAdvect->transform();

            advectPoints(*pointsToAdvect, *velocity, integrationOrder, timeStep, steps,
                advectFilter, filter);

            CPPUNIT_ASSERT_EQUAL(Index64(4), pointCount(pointsToAdvect->tree()));

            for (auto leafIter = pointsToAdvect->tree().beginLeaf(); leafIter; ++leafIter) {
                AttributeHandle<Vec3s> positionHandle(leafIter->constAttributeArray("P"));
                AttributeHandle<int> idHandle(leafIter->constAttributeArray("id"));
                for (auto iter = leafIter->beginIndexOn(); iter; ++iter) {
                    int theId = idHandle.get(*iter);
                    Vec3s position = transform.indexToWorld(
                        positionHandle.get(*iter) + iter.getCoord().asVec3d());
                    Vec3s expectedPosition(positions[theId]);
                    if (theId == 2)    expectedPosition += velocityBackground;
                    CPPUNIT_ASSERT(math::isApproxEqual(position, expectedPosition, tolerance));
                }
            }

            advectIncludeGroups.clear();
        }

        { // only keep points in "test" group
            includeGroups.push_back("test");

            auto leaf = points->tree().cbeginLeaf();
            MultiGroupFilter advectFilter(
                advectIncludeGroups, advectExcludeGroups, leaf->attributeSet());
            MultiGroupFilter filter(includeGroups, excludeGroups, leaf->attributeSet());

            auto pointsToAdvect = points->deepCopy();
            const auto& transform = pointsToAdvect->transform();

            advectPoints(*pointsToAdvect, *velocity, integrationOrder, timeStep, steps,
                advectFilter, filter);

            CPPUNIT_ASSERT_EQUAL(Index64(1), pointCount(pointsToAdvect->tree()));

            for (auto leafIter = pointsToAdvect->tree().beginLeaf(); leafIter; ++leafIter) {
                AttributeHandle<Vec3s> positionHandle(leafIter->constAttributeArray("P"));
                AttributeHandle<int> idHandle(leafIter->constAttributeArray("id"));
                for (auto iter = leafIter->beginIndexOn(); iter; ++iter) {
                    int theId = idHandle.get(*iter);
                    Vec3s position = transform.indexToWorld(
                        positionHandle.get(*iter) + iter.getCoord().asVec3d());
                    Vec3s expectedPosition(positions[theId]);
                    expectedPosition += velocityBackground;
                    CPPUNIT_ASSERT(math::isApproxEqual(position, expectedPosition, tolerance));
                }
            }

            includeGroups.clear();
        }

        { // only advect points in "test2" group, delete points in "test" group
            advectIncludeGroups.push_back("test2");
            excludeGroups.push_back("test");

            auto leaf = points->tree().cbeginLeaf();
            MultiGroupFilter advectFilter(
                advectIncludeGroups, advectExcludeGroups, leaf->attributeSet());
            MultiGroupFilter filter(includeGroups, excludeGroups, leaf->attributeSet());

            auto pointsToAdvect = points->deepCopy();
            const auto& transform = pointsToAdvect->transform();

            advectPoints(*pointsToAdvect, *velocity, integrationOrder, timeStep, steps,
                advectFilter, filter);

            CPPUNIT_ASSERT_EQUAL(Index64(3), pointCount(pointsToAdvect->tree()));

            for (auto leafIter = pointsToAdvect->tree().beginLeaf(); leafIter; ++leafIter) {
                AttributeHandle<Vec3s> positionHandle(leafIter->constAttributeArray("P"));
                AttributeHandle<int> idHandle(leafIter->constAttributeArray("id"));
                for (auto iter = leafIter->beginIndexOn(); iter; ++iter) {
                    int theId = idHandle.get(*iter);
                    Vec3s position = transform.indexToWorld(
                        positionHandle.get(*iter) + iter.getCoord().asVec3d());
                    Vec3s expectedPosition(positions[theId]);
                    if (theId == 1)    expectedPosition += velocityBackground;
                    CPPUNIT_ASSERT(math::isApproxEqual(position, expectedPosition, tolerance));
                }
            }

            advectIncludeGroups.clear();
            excludeGroups.clear();
        }

        { // advect all points, caching disabled
            auto pointsToAdvect = points->deepCopy();
            const auto& transform = pointsToAdvect->transform();

            auto leaf = points->tree().cbeginLeaf();
            MultiGroupFilter advectFilter(
                advectIncludeGroups, advectExcludeGroups, leaf->attributeSet());
            MultiGroupFilter filter(includeGroups, excludeGroups, leaf->attributeSet());

            advectPoints(*pointsToAdvect, *velocity, integrationOrder, timeStep, steps,
                advectFilter, filter, false);

            CPPUNIT_ASSERT_EQUAL(Index64(4), pointCount(pointsToAdvect->tree()));

            for (auto leafIter = pointsToAdvect->tree().beginLeaf(); leafIter; ++leafIter) {
                AttributeHandle<Vec3s> positionHandle(leafIter->constAttributeArray("P"));
                AttributeHandle<int> idHandle(leafIter->constAttributeArray("id"));
                for (auto iter = leafIter->beginIndexOn(); iter; ++iter) {
                    int theId = idHandle.get(*iter);
                    Vec3s position = transform.indexToWorld(
                        positionHandle.get(*iter) + iter.getCoord().asVec3d());
                    Vec3s expectedPosition(positions[theId]);
                    expectedPosition += velocityBackground;
                    CPPUNIT_ASSERT(math::isApproxEqual(position, expectedPosition, tolerance));
                }
            }
        }

        { // only advect points in "test2" group, delete points in "test" group, caching disabled
            advectIncludeGroups.push_back("test2");
            excludeGroups.push_back("test");

            auto leaf = points->tree().cbeginLeaf();
            MultiGroupFilter advectFilter(
                advectIncludeGroups, advectExcludeGroups, leaf->attributeSet());
            MultiGroupFilter filter(includeGroups, excludeGroups, leaf->attributeSet());

            auto pointsToAdvect = points->deepCopy();
            const auto& transform = pointsToAdvect->transform();

            advectPoints(*pointsToAdvect, *velocity, integrationOrder, timeStep, steps,
                advectFilter, filter, false);

            CPPUNIT_ASSERT_EQUAL(Index64(3), pointCount(pointsToAdvect->tree()));

            for (auto leafIter = pointsToAdvect->tree().beginLeaf(); leafIter; ++leafIter) {
                AttributeHandle<Vec3s> positionHandle(leafIter->constAttributeArray("P"));
                AttributeHandle<int> idHandle(leafIter->constAttributeArray("id"));
                for (auto iter = leafIter->beginIndexOn(); iter; ++iter) {
                    int theId = idHandle.get(*iter);
                    Vec3s position = transform.indexToWorld(
                        positionHandle.get(*iter) + iter.getCoord().asVec3d());
                    Vec3s expectedPosition(positions[theId]);
                    if (theId == 1)    expectedPosition += velocityBackground;
                    CPPUNIT_ASSERT(math::isApproxEqual(position, expectedPosition, tolerance));
                }
            }

            advectIncludeGroups.clear();
            excludeGroups.clear();
        }
    }
}


void
TestPointAdvect::testZalesaksDisk()
{
    // advect a notched sphere known as Zalesak's disk in a rotational velocity field

    // build the level set sphere

    Vec3s center(0, 0, 0);
    float radius = 10;
    float voxelSize = 0.2f;

    auto zalesak = tools::createLevelSetSphere<FloatGrid>(radius, center, voxelSize);

    // create box for notch using width and depth relative to radius

    const math::Transform::Ptr xform = math::Transform::createLinearTransform(voxelSize);

    Vec3f min(center);
    Vec3f max(center);
    float notchWidth = 0.5f;
    float notchDepth = 1.5f;

    min.x() -= (radius * notchWidth) / 2;
    min.y() -= (radius * (notchDepth - 1));
    min.z() -= radius * 1.1f;

    max.x() += (radius * notchWidth) / 2;
    max.y() += radius * 1.1f;
    max.z() += radius * 1.1f;

    math::BBox<Vec3f> bbox(min, max);

    auto notch = tools::createLevelSetBox<FloatGrid>(bbox, *xform);

    // subtract notch from the sphere

    tools::csgDifference(*zalesak, *notch);

    // scatter points inside the sphere

    auto points = points::denseUniformPointScatter(*zalesak, /*pointsPerVoxel=*/8);

    // append an integer "id" attribute

    auto idAttributeType = TypedAttributeArray<int>::attributeType();
    appendAttribute(points->tree(), "id", idAttributeType);

    // populate it in serial based on iteration order

    int id = 0;
    for (auto leaf = points->tree().beginLeaf(); leaf; ++leaf) {
        AttributeWriteHandle<int> handle(leaf->attributeArray("id"));
        for (auto iter = leaf->beginIndexOn(); iter; ++iter) {
            handle.set(*iter, id++);
        }
    }

    // copy grid into new grid for advecting

    auto pointsToAdvect = points->deepCopy();

    // populate a velocity grid that rotates in X

    auto velocity = Vec3SGrid::create(Vec3s(0));
    velocity->setTransform(xform);

    CoordBBox activeBbox(zalesak->evalActiveVoxelBoundingBox());
    activeBbox.expand(5);

    velocity->denseFill(activeBbox, Vec3s(0));

    for (auto leaf = velocity->tree().beginLeaf(); leaf; ++leaf) {
        for (auto iter = leaf->beginValueOn(); iter; ++iter) {
            Vec3s position = xform->indexToWorld(iter.getCoord().asVec3d());
            Vec3s vel = (position.cross(Vec3s(0, 0, 1)) * 2 * M_PI) / 10;
            iter.setValue(vel);
        }
    }

    // extract original positions

    const Index count = Index(pointCount(points->constTree()));

    std::vector<Vec3f> preAdvectPositions(count, Vec3f(0));

    for (auto leaf = points->constTree().cbeginLeaf(); leaf; ++leaf) {
        AttributeHandle<int> idHandle(leaf->constAttributeArray("id"));
        AttributeHandle<Vec3f> posHandle(leaf->constAttributeArray("P"));
        for (auto iter = leaf->beginIndexOn(); iter; ++iter) {
            Vec3f position = posHandle.get(*iter) + iter.getCoord().asVec3d();
            preAdvectPositions[idHandle.get(*iter)] = Vec3f(xform->indexToWorld(position));
        }
    }

    // advect points a half revolution

    points::advectPoints(*pointsToAdvect, *velocity, Index(4), 1.0, 5);

    // extract new positions

    std::vector<Vec3f> postAdvectPositions(count, Vec3f(0));

    for (auto leaf = pointsToAdvect->constTree().cbeginLeaf(); leaf; ++leaf) {
        AttributeHandle<int> idHandle(leaf->constAttributeArray("id"));
        AttributeHandle<Vec3f> posHandle(leaf->constAttributeArray("P"));
        for (auto iter = leaf->beginIndexOn(); iter; ++iter) {
            Vec3f position = posHandle.get(*iter) + iter.getCoord().asVec3d();
            postAdvectPositions[idHandle.get(*iter)] = Vec3f(xform->indexToWorld(position));
        }
    }

    for (Index i = 0; i < count; i++) {
        CPPUNIT_ASSERT(!math::isApproxEqual(
            preAdvectPositions[i], postAdvectPositions[i], Vec3f(0.1)));
    }

    // advect points another half revolution

    points::advectPoints(*pointsToAdvect, *velocity, Index(4), 1.0, 5);

    for (auto leaf = pointsToAdvect->constTree().cbeginLeaf(); leaf; ++leaf) {
        AttributeHandle<int> idHandle(leaf->constAttributeArray("id"));
        AttributeHandle<Vec3f> posHandle(leaf->constAttributeArray("P"));
        for (auto iter = leaf->beginIndexOn(); iter; ++iter) {
            Vec3f position = posHandle.get(*iter) + iter.getCoord().asVec3d();
            postAdvectPositions[idHandle.get(*iter)] = Vec3f(xform->indexToWorld(position));
        }
    }

    for (Index i = 0; i < count; i++) {
        CPPUNIT_ASSERT(math::isApproxEqual(
            preAdvectPositions[i], postAdvectPositions[i], Vec3f(0.1)));
    }
}

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
