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
#include <openvdb/Exceptions.h>
#include <openvdb/tree/LeafNode.h>
#include <openvdb/math/Math.h>// for math::Random01(), math::Pow3()

class TestLeaf: public CppUnit::TestCase
{
public:
    CPPUNIT_TEST_SUITE(TestLeaf);
    CPPUNIT_TEST(testBuffer);
    CPPUNIT_TEST(testGetValue);
    CPPUNIT_TEST(testSetValue);
    CPPUNIT_TEST(testIsValueSet);
    CPPUNIT_TEST(testProbeValue);
    CPPUNIT_TEST(testIterators);
    CPPUNIT_TEST(testEquivalence);
    CPPUNIT_TEST(testGetOrigin);
    CPPUNIT_TEST(testIteratorGetCoord);
    CPPUNIT_TEST(testNegativeIndexing);
    CPPUNIT_TEST(testIsConstant);
    CPPUNIT_TEST(testMedian);
    CPPUNIT_TEST(testFill);
    CPPUNIT_TEST_SUITE_END();

    void testBuffer();
    void testGetValue();
    void testSetValue();
    void testIsValueSet();
    void testProbeValue();
    void testIterators();
    void testEquivalence();
    void testGetOrigin();
    void testIteratorGetCoord();
    void testNegativeIndexing();
    void testIsConstant();
    void testMedian();
    void testFill();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestLeaf);

typedef openvdb::tree::LeafNode<int, 3> LeafType;
typedef LeafType::Buffer                BufferType;
using openvdb::Index;

void
TestLeaf::testBuffer()
{
    {// access
        BufferType buf;

        for (Index i = 0; i < BufferType::size(); ++i) {
            buf.mData[i] = i;
            CPPUNIT_ASSERT(buf[i] == buf.mData[i]);
        }
        for (Index i = 0; i < BufferType::size(); ++i) {
            buf[i] = i;
            CPPUNIT_ASSERT_EQUAL(int(i), buf[i]);
        }
    }

    {// swap
        BufferType buf0, buf1, buf2;

        int *buf0Data = buf0.mData;
        int *buf1Data = buf1.mData;

        for (Index i = 0; i < BufferType::size(); ++i) {
            buf0[i] = i;
            buf1[i] = i * 2;
        }

        buf0.swap(buf1);

        CPPUNIT_ASSERT(buf0.mData == buf1Data);
        CPPUNIT_ASSERT(buf1.mData == buf0Data);

        buf1.swap(buf0);

        CPPUNIT_ASSERT(buf0.mData == buf0Data);
        CPPUNIT_ASSERT(buf1.mData == buf1Data);

        buf0.swap(buf2);

        CPPUNIT_ASSERT(buf2.mData == buf0Data);

        buf2.swap(buf0);

        CPPUNIT_ASSERT(buf0.mData == buf0Data);
    }

}

void
TestLeaf::testGetValue()
{
    LeafType leaf(openvdb::Coord(0, 0, 0));

    leaf.mBuffer[0] = 2;
    leaf.mBuffer[1] = 3;
    leaf.mBuffer[2] = 4;
    leaf.mBuffer[65] = 10;

    CPPUNIT_ASSERT_EQUAL(2, leaf.getValue(openvdb::Coord(0, 0, 0)));
    CPPUNIT_ASSERT_EQUAL(3, leaf.getValue(openvdb::Coord(0, 0, 1)));
    CPPUNIT_ASSERT_EQUAL(4, leaf.getValue(openvdb::Coord(0, 0, 2)));

    CPPUNIT_ASSERT_EQUAL(10, leaf.getValue(openvdb::Coord(1, 0, 1)));
}

void
TestLeaf::testSetValue()
{
    LeafType leaf(openvdb::Coord(0, 0, 0), 3);

    openvdb::Coord xyz(0, 0, 0);
    leaf.setValueOn(xyz, 10);
    CPPUNIT_ASSERT_EQUAL(10, leaf.getValue(xyz));

    xyz.reset(7, 7, 7);
    leaf.setValueOn(xyz, 7);
    CPPUNIT_ASSERT_EQUAL(7, leaf.getValue(xyz));
    leaf.setValueOnly(xyz, 10);
    CPPUNIT_ASSERT_EQUAL(10, leaf.getValue(xyz));

    xyz.reset(2, 3, 6);
    leaf.setValueOn(xyz, 236);
    CPPUNIT_ASSERT_EQUAL(236, leaf.getValue(xyz));

    leaf.setValueOff(xyz, 1);
    CPPUNIT_ASSERT_EQUAL(1, leaf.getValue(xyz));
    CPPUNIT_ASSERT(!leaf.isValueOn(xyz));
}

void
TestLeaf::testIsValueSet()
{
    LeafType leaf(openvdb::Coord(0, 0, 0));
    leaf.setValueOn(openvdb::Coord(1, 5, 7), 10);

    CPPUNIT_ASSERT(leaf.isValueOn(openvdb::Coord(1, 5, 7)));

    CPPUNIT_ASSERT(!leaf.isValueOn(openvdb::Coord(0, 5, 7)));
    CPPUNIT_ASSERT(!leaf.isValueOn(openvdb::Coord(1, 6, 7)));
    CPPUNIT_ASSERT(!leaf.isValueOn(openvdb::Coord(0, 5, 6)));
}

void
TestLeaf::testProbeValue()
{
    LeafType leaf(openvdb::Coord(0, 0, 0));
    leaf.setValueOn(openvdb::Coord(1, 6, 5), 10);

    LeafType::ValueType val;
    CPPUNIT_ASSERT(leaf.probeValue(openvdb::Coord(1, 6, 5), val));
    CPPUNIT_ASSERT(!leaf.probeValue(openvdb::Coord(1, 6, 4), val));
}

void
TestLeaf::testIterators()
{
    LeafType leaf(openvdb::Coord(0, 0, 0), 2);
    leaf.setValueOn(openvdb::Coord(1, 2, 3), -3);
    leaf.setValueOn(openvdb::Coord(5, 2, 3),  4);
    LeafType::ValueType sum = 0;
    for (LeafType::ValueOnIter iter = leaf.beginValueOn(); iter; ++iter) sum += *iter;
    CPPUNIT_ASSERT_EQUAL((-3 + 4), sum);
}

void
TestLeaf::testEquivalence()
{
    LeafType leaf( openvdb::Coord(0, 0, 0), 2);
    LeafType leaf2(openvdb::Coord(0, 0, 0), 3);

    CPPUNIT_ASSERT(leaf != leaf2);

    for(openvdb::Index32 i = 0; i < LeafType::size(); ++i) {
        leaf.setValueOnly(i, i);
        leaf2.setValueOnly(i, i);
    }
    CPPUNIT_ASSERT(leaf == leaf2);

    // set some values.
    leaf.setValueOn(openvdb::Coord(0, 0, 0), 1);
    leaf.setValueOn(openvdb::Coord(0, 1, 0), 1);
    leaf.setValueOn(openvdb::Coord(1, 1, 0), 1);
    leaf.setValueOn(openvdb::Coord(1, 1, 2), 1);

    leaf2.setValueOn(openvdb::Coord(0, 0, 0), 1);
    leaf2.setValueOn(openvdb::Coord(0, 1, 0), 1);
    leaf2.setValueOn(openvdb::Coord(1, 1, 0), 1);
    leaf2.setValueOn(openvdb::Coord(1, 1, 2), 1);

    CPPUNIT_ASSERT(leaf == leaf2);

    leaf2.setValueOn(openvdb::Coord(0, 0, 1), 1);

    CPPUNIT_ASSERT(leaf != leaf2);

    leaf2.setValueOff(openvdb::Coord(0, 0, 1), 1);

    CPPUNIT_ASSERT(leaf == leaf2);
}

void
TestLeaf::testGetOrigin()
{
    {
        LeafType leaf(openvdb::Coord(1, 0, 0), 1);
        CPPUNIT_ASSERT_EQUAL(openvdb::Coord(0, 0, 0), leaf.origin());
    }
    {
        LeafType leaf(openvdb::Coord(0, 0, 0), 1);
        CPPUNIT_ASSERT_EQUAL(openvdb::Coord(0, 0, 0), leaf.origin());
    }
    {
        LeafType leaf(openvdb::Coord(8, 0, 0), 1);
        CPPUNIT_ASSERT_EQUAL(openvdb::Coord(8, 0, 0), leaf.origin());
    }
    {
        LeafType leaf(openvdb::Coord(8, 1, 0), 1);
        CPPUNIT_ASSERT_EQUAL(openvdb::Coord(8, 0, 0), leaf.origin());
    }
    {
        LeafType leaf(openvdb::Coord(1024, 1, 3), 1);
        CPPUNIT_ASSERT_EQUAL(openvdb::Coord(128*8, 0, 0), leaf.origin());
    }
    {
        LeafType leaf(openvdb::Coord(1023, 1, 3), 1);
        CPPUNIT_ASSERT_EQUAL(openvdb::Coord(127*8, 0, 0), leaf.origin());
    }
    {
        LeafType leaf(openvdb::Coord(512, 512, 512), 1);
        CPPUNIT_ASSERT_EQUAL(openvdb::Coord(512, 512, 512), leaf.origin());
    }
    {
        LeafType leaf(openvdb::Coord(2, 52, 515), 1);
        CPPUNIT_ASSERT_EQUAL(openvdb::Coord(0, 48, 512), leaf.origin());
    }
}

void
TestLeaf::testIteratorGetCoord()
{
    using namespace openvdb;

    LeafType leaf(openvdb::Coord(8, 8, 0), 2);

    CPPUNIT_ASSERT_EQUAL(Coord(8, 8, 0), leaf.origin());

    leaf.setValueOn(Coord(1, 2, 3), -3);
    leaf.setValueOn(Coord(5, 2, 3),  4);

    LeafType::ValueOnIter iter = leaf.beginValueOn();
    Coord xyz = iter.getCoord();
    CPPUNIT_ASSERT_EQUAL(Coord(9, 10, 3), xyz);

    ++iter;
    xyz = iter.getCoord();
    CPPUNIT_ASSERT_EQUAL(Coord(13, 10, 3), xyz);
}

void
TestLeaf::testNegativeIndexing()
{
    using namespace openvdb;

    LeafType leaf(openvdb::Coord(-9, -2, -8), 1);

    CPPUNIT_ASSERT_EQUAL(Coord(-16, -8, -8), leaf.origin());

    leaf.setValueOn(Coord(1, 2, 3), -3);
    leaf.setValueOn(Coord(5, 2, 3),  4);

    CPPUNIT_ASSERT_EQUAL(-3, leaf.getValue(Coord(1, 2, 3)));
    CPPUNIT_ASSERT_EQUAL(4, leaf.getValue(Coord(5, 2, 3)));

    LeafType::ValueOnIter iter = leaf.beginValueOn();
    Coord xyz = iter.getCoord();
    CPPUNIT_ASSERT_EQUAL(Coord(-15, -6, -5), xyz);

    ++iter;
    xyz = iter.getCoord();
    CPPUNIT_ASSERT_EQUAL(Coord(-11, -6, -5), xyz);
}

void
TestLeaf::testIsConstant()
{
    using namespace openvdb;
    const Coord origin(-9, -2, -8);

    {// check old version (v3.0 and older) with float
        // Acceptable range: first-value +/- tolerance
        const float val = 1.0f, tol = 0.01f;
        tree::LeafNode<float, 3> leaf(origin, val, true);
        float v = 0.0f;
        bool stat = false;
        CPPUNIT_ASSERT(leaf.isConstant(v, stat, tol));
        CPPUNIT_ASSERT(stat);
        CPPUNIT_ASSERT_EQUAL(val, v);

        leaf.setValueOff(0);
        CPPUNIT_ASSERT(!leaf.isConstant(v, stat, tol));

        leaf.setValueOn(0);
        CPPUNIT_ASSERT(leaf.isConstant(v, stat, tol));

        leaf.setValueOn(0, val + 0.99f*tol);
        CPPUNIT_ASSERT(leaf.isConstant(v, stat, tol));
        CPPUNIT_ASSERT(stat);
        CPPUNIT_ASSERT_EQUAL(val + 0.99f*tol, v);

        leaf.setValueOn(0, val + 1.01f*tol);
        CPPUNIT_ASSERT(!leaf.isConstant(v, stat, tol));
    }
    {// check old version (v3.0 and older) with double
        // Acceptable range: first-value +/- tolerance
        const double val = 1.0, tol = 0.00001;
        tree::LeafNode<double, 3> leaf(origin, val, true);
        double v = 0.0;
        bool stat = false;
        CPPUNIT_ASSERT(leaf.isConstant(v, stat, tol));
        CPPUNIT_ASSERT(stat);
        CPPUNIT_ASSERT_EQUAL(val, v);

        leaf.setValueOff(0);
        CPPUNIT_ASSERT(!leaf.isConstant(v, stat, tol));

        leaf.setValueOn(0);
        CPPUNIT_ASSERT(leaf.isConstant(v, stat, tol));

        leaf.setValueOn(0, val + 0.99*tol);
        CPPUNIT_ASSERT(leaf.isConstant(v, stat, tol));
        CPPUNIT_ASSERT(stat);
        CPPUNIT_ASSERT_EQUAL(val + 0.99*tol, v);

        leaf.setValueOn(0, val + 1.01*tol);
        CPPUNIT_ASSERT(!leaf.isConstant(v, stat, tol));
    }
    {// check newer version (v3.2 and newer) with float
        // Acceptable range: max - min <= tolerance
        const float val = 1.0, tol = 0.01f;
        tree::LeafNode<float, 3> leaf(origin, val, true);
        float vmin = 0.0f, vmax = 0.0f;
        bool stat = false;

        CPPUNIT_ASSERT(leaf.isConstant(vmin, vmax, stat, tol));
        CPPUNIT_ASSERT(stat);
        CPPUNIT_ASSERT_EQUAL(val, vmin);
        CPPUNIT_ASSERT_EQUAL(val, vmax);

        leaf.setValueOff(0);
        CPPUNIT_ASSERT(!leaf.isConstant(vmin, vmax, stat, tol));

        leaf.setValueOn(0);
        CPPUNIT_ASSERT(leaf.isConstant(vmin, vmax, stat, tol));

        leaf.setValueOn(0, val + tol);
        CPPUNIT_ASSERT(leaf.isConstant(vmin, vmax, stat, tol));
        CPPUNIT_ASSERT_EQUAL(val, vmin);
        CPPUNIT_ASSERT_EQUAL(val + tol, vmax);

        leaf.setValueOn(0, val + 1.01f*tol);
        CPPUNIT_ASSERT(!leaf.isConstant(vmin, vmax, stat, tol));
    }
    {// check newer version (v3.2 and newer) with double
        // Acceptable range: (max- min) <= tolerance
        const double val = 1.0, tol = 0.000001;
        tree::LeafNode<double, 3> leaf(origin, val, true);
        double vmin = 0.0, vmax = 0.0;
        bool stat = false;
        CPPUNIT_ASSERT(leaf.isConstant(vmin, vmax, stat, tol));
        CPPUNIT_ASSERT(stat);
        CPPUNIT_ASSERT_EQUAL(val, vmin);
        CPPUNIT_ASSERT_EQUAL(val, vmax);

        leaf.setValueOff(0);
        CPPUNIT_ASSERT(!leaf.isConstant(vmin, vmax, stat, tol));

        leaf.setValueOn(0);
        CPPUNIT_ASSERT(leaf.isConstant(vmin, vmax, stat, tol));

        leaf.setValueOn(0, val + tol);
        CPPUNIT_ASSERT(leaf.isConstant(vmin, vmax, stat, tol));
        CPPUNIT_ASSERT_EQUAL(val, vmin);
        CPPUNIT_ASSERT_EQUAL(val + tol, vmax);

        leaf.setValueOn(0, val + 1.01*tol);
        CPPUNIT_ASSERT(!leaf.isConstant(vmin, vmax, stat, tol));
    }
    {// check newer version (v3.2 and newer) with float and random values
        typedef tree::LeafNode<float,3> LeafNodeT;
        const float val = 1.0, tol = 1.0f;
        LeafNodeT leaf(origin, val, true);
        float min = 2.0f, max = -min;
        math::Random01 r(145);// random values in the range [0,1]
        for (Index i=0; i<LeafNodeT::NUM_VALUES; ++i) {
            const float v = float(r());
            if (v < min) min = v;
            if (v > max) max = v;
            leaf.setValueOnly(i, v);
        }
        float vmin = 0.0f, vmax = 0.0f;
        bool stat = false;
        CPPUNIT_ASSERT(leaf.isConstant(vmin, vmax, stat, tol));
        CPPUNIT_ASSERT(stat);
        CPPUNIT_ASSERT(math::isApproxEqual(min, vmin));
        CPPUNIT_ASSERT(math::isApproxEqual(max, vmax));
    }
}

void
TestLeaf::testMedian()
{
    using namespace openvdb;
    const Coord origin(-9, -2, -8);
    std::vector<float> v{5, 6, 4, 3, 2, 6, 7, 9, 3};
    tree::LeafNode<float, 3> leaf(origin, 1.0f, false);

    float val = 0.0f;
    CPPUNIT_ASSERT_EQUAL(Index(0), leaf.medianOn(val));
    CPPUNIT_ASSERT_EQUAL(0.0f, val);
    CPPUNIT_ASSERT_EQUAL(leaf.numValues(), leaf.medianOff(val));
    CPPUNIT_ASSERT_EQUAL(1.0f, val);
    CPPUNIT_ASSERT_EQUAL(1.0f, leaf.medianAll());

    leaf.setValue(Coord(0,0,0), v[0]);
    CPPUNIT_ASSERT_EQUAL(Index(1), leaf.medianOn(val));
    CPPUNIT_ASSERT_EQUAL(v[0], val);
    CPPUNIT_ASSERT_EQUAL(leaf.numValues()-1, leaf.medianOff(val));
    CPPUNIT_ASSERT_EQUAL(1.0f, val);
    CPPUNIT_ASSERT_EQUAL(1.0f, leaf.medianAll());

    leaf.setValue(Coord(0,0,1), v[1]);
    CPPUNIT_ASSERT_EQUAL(Index(2), leaf.medianOn(val));
    CPPUNIT_ASSERT_EQUAL(v[0], val);
    CPPUNIT_ASSERT_EQUAL(leaf.numValues()-2, leaf.medianOff(val));
    CPPUNIT_ASSERT_EQUAL(1.0f, val);
    CPPUNIT_ASSERT_EQUAL(1.0f, leaf.medianAll());

    leaf.setValue(Coord(0,2,1), v[2]);
    CPPUNIT_ASSERT_EQUAL(Index(3), leaf.medianOn(val));
    CPPUNIT_ASSERT_EQUAL(v[0], val);
    CPPUNIT_ASSERT_EQUAL(leaf.numValues()-3, leaf.medianOff(val));
    CPPUNIT_ASSERT_EQUAL(1.0f, val);
    CPPUNIT_ASSERT_EQUAL(1.0f, leaf.medianAll());

    leaf.setValue(Coord(1,2,1), v[3]);
    CPPUNIT_ASSERT_EQUAL(Index(4), leaf.medianOn(val));
    CPPUNIT_ASSERT_EQUAL(v[2], val);
    CPPUNIT_ASSERT_EQUAL(leaf.numValues()-4, leaf.medianOff(val));
    CPPUNIT_ASSERT_EQUAL(1.0f, val);
    CPPUNIT_ASSERT_EQUAL(1.0f, leaf.medianAll());

    leaf.setValue(Coord(1,2,3), v[4]);
    CPPUNIT_ASSERT_EQUAL(Index(5), leaf.medianOn(val));
    CPPUNIT_ASSERT_EQUAL(v[2], val);
    CPPUNIT_ASSERT_EQUAL(leaf.numValues()-5, leaf.medianOff(val));
    CPPUNIT_ASSERT_EQUAL(1.0f, val);
    CPPUNIT_ASSERT_EQUAL(1.0f, leaf.medianAll());

    leaf.setValue(Coord(2,2,1), v[5]);
    CPPUNIT_ASSERT_EQUAL(Index(6), leaf.medianOn(val));
    CPPUNIT_ASSERT_EQUAL(v[2], val);
    CPPUNIT_ASSERT_EQUAL(leaf.numValues()-6, leaf.medianOff(val));
    CPPUNIT_ASSERT_EQUAL(1.0f, val);
    CPPUNIT_ASSERT_EQUAL(1.0f, leaf.medianAll());

    leaf.setValue(Coord(2,4,1), v[6]);
    CPPUNIT_ASSERT_EQUAL(Index(7), leaf.medianOn(val));
    CPPUNIT_ASSERT_EQUAL(v[0], val);
    CPPUNIT_ASSERT_EQUAL(leaf.numValues()-7, leaf.medianOff(val));
    CPPUNIT_ASSERT_EQUAL(1.0f, val);
    CPPUNIT_ASSERT_EQUAL(1.0f, leaf.medianAll());

    leaf.setValue(Coord(2,6,1), v[7]);
    CPPUNIT_ASSERT_EQUAL(Index(8), leaf.medianOn(val));
    CPPUNIT_ASSERT_EQUAL(v[0], val);
    CPPUNIT_ASSERT_EQUAL(leaf.numValues()-8, leaf.medianOff(val));
    CPPUNIT_ASSERT_EQUAL(1.0f, val);
    CPPUNIT_ASSERT_EQUAL(1.0f, leaf.medianAll());

    leaf.setValue(Coord(7,2,1), v[8]);
    CPPUNIT_ASSERT_EQUAL(Index(9), leaf.medianOn(val));
    CPPUNIT_ASSERT_EQUAL(v[0], val);
    CPPUNIT_ASSERT_EQUAL(leaf.numValues()-9, leaf.medianOff(val));
    CPPUNIT_ASSERT_EQUAL(1.0f, val);
    CPPUNIT_ASSERT_EQUAL(1.0f, leaf.medianAll());

    leaf.fill(2.0f, true);

    CPPUNIT_ASSERT_EQUAL(leaf.numValues(), leaf.medianOn(val));
    CPPUNIT_ASSERT_EQUAL(2.0f, val);
    CPPUNIT_ASSERT_EQUAL(Index(0), leaf.medianOff(val));
    CPPUNIT_ASSERT_EQUAL(2.0f, val);
    CPPUNIT_ASSERT_EQUAL(2.0f, leaf.medianAll());
}

void
TestLeaf::testFill()
{
    using namespace openvdb;
    const Coord origin(-9, -2, -8);

    const float bg = 0.0f, fg = 1.0f;
    tree::LeafNode<float, 3> leaf(origin, bg, false);

    const int bboxDim = 1 + int(leaf.dim() >> 1);
    auto bbox = CoordBBox::createCube(leaf.origin(), bboxDim);
    CPPUNIT_ASSERT_EQUAL(math::Pow3(bboxDim), int(bbox.volume()));

    bbox = leaf.getNodeBoundingBox();
    leaf.fill(bbox, bg, false);
    CPPUNIT_ASSERT(leaf.isEmpty());
    leaf.fill(bbox, fg, true);
    CPPUNIT_ASSERT(leaf.isDense());

    leaf.fill(bbox, bg, false);
    CPPUNIT_ASSERT(leaf.isEmpty());

    // Fill a region that is larger than the node but that doesn't completely enclose it.
    bbox.max() = bbox.min() + (bbox.dim() >> 1);
    bbox.expand(bbox.min() - Coord{10});
    leaf.fill(bbox, fg, true);

    // Verify that fill() correctly clips the fill region to the node.
    auto clippedBBox = leaf.getNodeBoundingBox();
    clippedBBox.intersect(bbox);
    CPPUNIT_ASSERT_EQUAL(int(clippedBBox.volume()), int(leaf.onVoxelCount()));
}

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
