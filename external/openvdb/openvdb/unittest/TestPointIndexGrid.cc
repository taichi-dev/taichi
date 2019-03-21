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
#include <openvdb/tools/PointIndexGrid.h>

#include <vector>
#include <algorithm>
#include <cmath>
#include "util.h" // for genPoints


struct TestPointIndexGrid: public CppUnit::TestCase
{
    CPPUNIT_TEST_SUITE(TestPointIndexGrid);
    CPPUNIT_TEST(testPointIndexGrid);
    CPPUNIT_TEST(testPointIndexFilter);
    CPPUNIT_TEST(testWorldSpaceSearchAndUpdate);
    CPPUNIT_TEST_SUITE_END();

    void testPointIndexGrid();
    void testPointIndexFilter();
    void testWorldSpaceSearchAndUpdate();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestPointIndexGrid);

////////////////////////////////////////

namespace {

class PointList
{
public:
    typedef openvdb::Vec3R  PosType;

    PointList(const std::vector<PosType>& points)
        : mPoints(&points)
    {
    }

    size_t size() const {
        return mPoints->size();
    }

    void getPos(size_t n, PosType& xyz) const {
        xyz = (*mPoints)[n];
    }

protected:
    std::vector<PosType> const * const mPoints;
}; // PointList


template<typename T>
bool hasDuplicates(const std::vector<T>& items)
{
    std::vector<T> vec(items);
    std::sort(vec.begin(), vec.end());

    size_t duplicates = 0;
    for (size_t n = 1, N = vec.size(); n < N; ++n) {
        if (vec[n] == vec[n-1]) ++duplicates;
    }
    return duplicates != 0;
}


template<typename T>
struct WeightedAverageAccumulator {
    typedef T ValueType;
    WeightedAverageAccumulator(T const * const array, const T radius)
        : mValues(array), mInvRadius(1.0/radius), mWeightSum(0.0), mValueSum(0.0) {}

    void reset() { mWeightSum = mValueSum = T(0.0); }

    void operator()(const T distSqr, const size_t pointIndex) {
        const T weight = T(1.0) - openvdb::math::Sqrt(distSqr) * mInvRadius;
        mWeightSum += weight;
        mValueSum += weight * mValues[pointIndex];
    }

    T result() const { return mWeightSum > T(0.0) ? mValueSum / mWeightSum : T(0.0); }

private:
    T const * const mValues;
    const T mInvRadius;
    T mWeightSum, mValueSum;
}; // struct WeightedAverageAccumulator

} // namespace



////////////////////////////////////////


void
TestPointIndexGrid::testPointIndexGrid()
{
    const float voxelSize = 0.01f;
    const openvdb::math::Transform::Ptr transform =
            openvdb::math::Transform::createLinearTransform(voxelSize);

    // generate points

    std::vector<openvdb::Vec3R> points;
    unittest_util::genPoints(40000, points);

    PointList pointList(points);


    // construct data structure
    typedef openvdb::tools::PointIndexGrid PointIndexGrid;

    PointIndexGrid::Ptr pointGridPtr =
        openvdb::tools::createPointIndexGrid<PointIndexGrid>(pointList, *transform);

    openvdb::CoordBBox bbox;
    pointGridPtr->tree().evalActiveVoxelBoundingBox(bbox);

    // coord bbox search

    typedef PointIndexGrid::ConstAccessor ConstAccessor;
    typedef openvdb::tools::PointIndexIterator<> PointIndexIterator;

    ConstAccessor acc = pointGridPtr->getConstAccessor();
    PointIndexIterator it(bbox, acc);

    CPPUNIT_ASSERT(it.test());
    CPPUNIT_ASSERT_EQUAL(points.size(), it.size());

    // fractional bbox search

    openvdb::BBoxd region(bbox.min().asVec3d(), bbox.max().asVec3d());

    // points are bucketed in a cell-centered fashion, we need to pad the
    // coordinate range to get the same search region in the fractional bbox.
    region.expand(voxelSize * 0.5);

    it.searchAndUpdate(region, acc, pointList, *transform);

    CPPUNIT_ASSERT(it.test());
    CPPUNIT_ASSERT_EQUAL(points.size(), it.size());

    {
        std::vector<uint32_t> vec;
        vec.reserve(it.size());
        for (; it; ++it) {
            vec.push_back(*it);
        }

        CPPUNIT_ASSERT_EQUAL(vec.size(), it.size());
        CPPUNIT_ASSERT(!hasDuplicates(vec));
    }

    // radial search
    openvdb::Vec3d center = region.getCenter();
    double radius = region.extents().x() * 0.5;
    it.searchAndUpdate(center, radius, acc, pointList, *transform);

    CPPUNIT_ASSERT(it.test());
    CPPUNIT_ASSERT_EQUAL(points.size(), it.size());

    {
        std::vector<uint32_t> vec;
        vec.reserve(it.size());
        for (; it; ++it) {
            vec.push_back(*it);
        }

        CPPUNIT_ASSERT_EQUAL(vec.size(), it.size());
        CPPUNIT_ASSERT(!hasDuplicates(vec));
    }


    center = region.min();
    it.searchAndUpdate(center, radius, acc, pointList, *transform);

    CPPUNIT_ASSERT(it.test());

    {
        std::vector<uint32_t> vec;
        vec.reserve(it.size());
        for (; it; ++it) {
            vec.push_back(*it);
        }

        CPPUNIT_ASSERT_EQUAL(vec.size(), it.size());
        CPPUNIT_ASSERT(!hasDuplicates(vec));

        // check that no points where missed.

        std::vector<unsigned char> indexMask(points.size(), 0);
        for (size_t n = 0, N = vec.size(); n < N; ++n) {
            indexMask[vec[n]] = 1;
        }

        const double r2 = radius * radius;
        openvdb::Vec3R v;
        for (size_t n = 0, N = indexMask.size(); n < N; ++n) {
            v = center - transform->worldToIndex(points[n]);
            if (indexMask[n] == 0) {
                CPPUNIT_ASSERT(!(v.lengthSqr() < r2));
            } else {
                CPPUNIT_ASSERT(v.lengthSqr() < r2);
            }
        }
    }


    // Check partitioning

    CPPUNIT_ASSERT(openvdb::tools::isValidPartition(pointList, *pointGridPtr));

    points[10000].x() += 1.5; // manually modify a few points.
    points[20000].x() += 1.5;
    points[30000].x() += 1.5;

    CPPUNIT_ASSERT(!openvdb::tools::isValidPartition(pointList, *pointGridPtr));

    PointIndexGrid::Ptr pointGrid2Ptr =
        openvdb::tools::getValidPointIndexGrid<PointIndexGrid>(pointList, pointGridPtr);

    CPPUNIT_ASSERT(openvdb::tools::isValidPartition(pointList, *pointGrid2Ptr));
}


void
TestPointIndexGrid::testPointIndexFilter()
{
    // generate points
    const float voxelSize = 0.01f;
    const size_t pointCount = 10000;
    const openvdb::math::Transform::Ptr transform =
            openvdb::math::Transform::createLinearTransform(voxelSize);

    std::vector<openvdb::Vec3d> points;
    unittest_util::genPoints(pointCount, points);

    PointList pointList(points);

    // construct data structure
    typedef openvdb::tools::PointIndexGrid PointIndexGrid;

    PointIndexGrid::Ptr pointGridPtr =
        openvdb::tools::createPointIndexGrid<PointIndexGrid>(pointList, *transform);


    std::vector<double> pointDensity(pointCount, 1.0);

    openvdb::tools::PointIndexFilter<PointList>
        filter(pointList, pointGridPtr->tree(), pointGridPtr->transform());

    const double radius = 3.0 * voxelSize;

    WeightedAverageAccumulator<double>
        accumulator(&pointDensity.front(), radius);

    double sum = 0.0;
    for (size_t n = 0, N = points.size(); n < N; ++n) {
        accumulator.reset();
        filter.searchAndApply(points[n], radius, accumulator);
        sum += accumulator.result();
    }

    CPPUNIT_ASSERT_DOUBLES_EQUAL(sum, double(points.size()), 1e-6);
}


void
TestPointIndexGrid::testWorldSpaceSearchAndUpdate()
{
    // Create random particles in a cube.
    openvdb::math::Rand01<> rnd(0);

    const size_t N = 1000000;
    std::vector<openvdb::Vec3d> pos;
    pos.reserve(N);

    // Create a box to query points.
    openvdb::BBoxd wsBBox(openvdb::Vec3d(0.25), openvdb::Vec3d(0.75));

    std::set<size_t> indexListA;

    for (size_t i = 0; i < N; ++i) {
        openvdb::Vec3d p(rnd(), rnd(), rnd());
        pos.push_back(p);

        if (wsBBox.isInside(p)) {
            indexListA.insert(i);
        }
    }

    // Create a point index grid
    const double dx = 0.025;
    openvdb::math::Transform::Ptr transform = openvdb::math::Transform::createLinearTransform(dx);

    PointList pointArray(pos);
    openvdb::tools::PointIndexGrid::Ptr pointIndexGrid
        = openvdb::tools::createPointIndexGrid<openvdb::tools::PointIndexGrid, PointList>(pointArray, *transform);

    // Search for points within the box.
    openvdb::tools::PointIndexGrid::ConstAccessor acc = pointIndexGrid->getConstAccessor();

    openvdb::tools::PointIndexIterator<openvdb::tools::PointIndexTree> pointIndexIter;
    pointIndexIter.worldSpaceSearchAndUpdate<PointList>(wsBBox, acc, pointArray, pointIndexGrid->transform());

    std::set<size_t> indexListB;
    for (; pointIndexIter; ++pointIndexIter) {
        indexListB.insert(*pointIndexIter);
    }

    CPPUNIT_ASSERT_EQUAL(indexListA.size(), indexListB.size());
}


// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
