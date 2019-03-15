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
#include <openvdb/tools/PointPartitioner.h>

#include <vector>


class TestPointPartitioner: public CppUnit::TestCase
{
public:
    CPPUNIT_TEST_SUITE(TestPointPartitioner);
    CPPUNIT_TEST(testPartitioner);
    CPPUNIT_TEST_SUITE_END();

    void testPartitioner();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestPointPartitioner);

////////////////////////////////////////

namespace {

struct PointList {
    typedef openvdb::Vec3s PosType;

    PointList(const std::vector<PosType>& points) : mPoints(&points) {}

    size_t size() const { return mPoints->size(); }

    void getPos(size_t n, PosType& xyz) const { xyz = (*mPoints)[n]; }

protected:
    std::vector<PosType> const * const mPoints;
}; // PointList

} // namespace

////////////////////////////////////////


void
TestPointPartitioner::testPartitioner()
{
    const size_t pointCount = 10000;
    const float voxelSize = 0.1f;

    std::vector<openvdb::Vec3s> points(pointCount, openvdb::Vec3s(0.f));
    for (size_t n = 1; n < pointCount; ++n) {
        points[n].x() = points[n-1].x() + voxelSize;
    }

    PointList pointList(points);

    const openvdb::math::Transform::Ptr transform =
            openvdb::math::Transform::createLinearTransform(voxelSize);

    typedef openvdb::tools::UInt32PointPartitioner PointPartitioner;

    PointPartitioner::Ptr partitioner =
            PointPartitioner::create(pointList, *transform);

    CPPUNIT_ASSERT(!partitioner->empty());

    // The default interpretation should be cell-centered.
    CPPUNIT_ASSERT(partitioner->usingCellCenteredTransform());

    const size_t expectedPageCount = pointCount / (1u << PointPartitioner::LOG2DIM);

    CPPUNIT_ASSERT_EQUAL(expectedPageCount, partitioner->size());
    CPPUNIT_ASSERT_EQUAL(openvdb::Coord(0), partitioner->origin(0));

    PointPartitioner::IndexIterator it = partitioner->indices(0);

    CPPUNIT_ASSERT(it.test());
    CPPUNIT_ASSERT_EQUAL(it.size(), size_t(1 << PointPartitioner::LOG2DIM));

    PointPartitioner::IndexIterator itB = partitioner->indices(0);

    CPPUNIT_ASSERT_EQUAL(++it, ++itB);
    CPPUNIT_ASSERT(it != ++itB);

    std::vector<PointPartitioner::IndexType> indices;

    for (it.reset(); it; ++it) {
        indices.push_back(*it);
    }

    CPPUNIT_ASSERT_EQUAL(it.size(), indices.size());

    size_t idx = 0;
    for (itB.reset(); itB; ++itB) {
        CPPUNIT_ASSERT_EQUAL(indices[idx++], *itB);
    }
}


// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
