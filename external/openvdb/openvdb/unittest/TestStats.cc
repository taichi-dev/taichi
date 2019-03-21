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

#include <openvdb/openvdb.h>
#include <openvdb/math/Operators.h> // for ISGradient
#include <openvdb/math/Stats.h>
#include <openvdb/tools/Statistics.h>
#include <cppunit/extensions/HelperMacros.h>

#define ASSERT_DOUBLES_EXACTLY_EQUAL(expected, actual) \
    CPPUNIT_ASSERT_DOUBLES_EQUAL((expected), (actual), /*tolerance=*/0.0);


class TestStats: public CppUnit::TestCase
{
public:
    CPPUNIT_TEST_SUITE(TestStats);
    CPPUNIT_TEST(testMinMax);
    CPPUNIT_TEST(testExtrema);
    CPPUNIT_TEST(testStats);
    CPPUNIT_TEST(testHistogram);
    CPPUNIT_TEST(testGridExtrema);
    CPPUNIT_TEST(testGridStats);
    CPPUNIT_TEST(testGridHistogram);
    CPPUNIT_TEST(testGridOperatorStats);
    CPPUNIT_TEST_SUITE_END();

    void testMinMax();
    void testExtrema();
    void testStats();
    void testHistogram();
    void testGridExtrema();
    void testGridStats();
    void testGridHistogram();
    void testGridOperatorStats();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestStats);

void
TestStats::testMinMax()
{
    {// test Coord which uses lexicographic less than
        openvdb::math::MinMax<openvdb::Coord> s(openvdb::Coord::max(), openvdb::Coord::min());
        //openvdb::math::MinMax<openvdb::Coord> s;// will not compile since Coord is not a POD type
        CPPUNIT_ASSERT_EQUAL(openvdb::Coord::max(), s.min());
        CPPUNIT_ASSERT_EQUAL(openvdb::Coord::min(), s.max());
        s.add( openvdb::Coord(1,2,3) );
        CPPUNIT_ASSERT_EQUAL(openvdb::Coord(1,2,3), s.min());
        CPPUNIT_ASSERT_EQUAL(openvdb::Coord(1,2,3), s.max());
        s.add( openvdb::Coord(0,2,3) );
        CPPUNIT_ASSERT_EQUAL(openvdb::Coord(0,2,3), s.min());
        CPPUNIT_ASSERT_EQUAL(openvdb::Coord(1,2,3), s.max());
        s.add( openvdb::Coord(1,2,4) );
        CPPUNIT_ASSERT_EQUAL(openvdb::Coord(0,2,3), s.min());
        CPPUNIT_ASSERT_EQUAL(openvdb::Coord(1,2,4), s.max());
    }
    {// test double
        openvdb::math::MinMax<double> s;
        CPPUNIT_ASSERT_EQUAL( std::numeric_limits<double>::max(), s.min());
        CPPUNIT_ASSERT_EQUAL(-std::numeric_limits<double>::max(), s.max());
        s.add( 1.0 );
        CPPUNIT_ASSERT_EQUAL(1.0, s.min());
        CPPUNIT_ASSERT_EQUAL(1.0, s.max());
        s.add( 2.5 );
        CPPUNIT_ASSERT_EQUAL(1.0, s.min());
        CPPUNIT_ASSERT_EQUAL(2.5, s.max());
        s.add( -0.5 );
        CPPUNIT_ASSERT_EQUAL(-0.5, s.min());
        CPPUNIT_ASSERT_EQUAL( 2.5, s.max());
    }
    {// test int
        openvdb::math::MinMax<int> s;
        CPPUNIT_ASSERT_EQUAL(std::numeric_limits<int>::max(), s.min());
        CPPUNIT_ASSERT_EQUAL(std::numeric_limits<int>::min(), s.max());
        s.add( 1 );
        CPPUNIT_ASSERT_EQUAL(1, s.min());
        CPPUNIT_ASSERT_EQUAL(1, s.max());
        s.add( 2 );
        CPPUNIT_ASSERT_EQUAL(1, s.min());
        CPPUNIT_ASSERT_EQUAL(2, s.max());
        s.add( -5 );
        CPPUNIT_ASSERT_EQUAL(-5, s.min());
        CPPUNIT_ASSERT_EQUAL( 2, s.max());
    }
    {// test unsigned
        openvdb::math::MinMax<uint32_t> s;
        CPPUNIT_ASSERT_EQUAL(std::numeric_limits<uint32_t>::max(), s.min());
        CPPUNIT_ASSERT_EQUAL(uint32_t(0), s.max());
        s.add( 1 );
        CPPUNIT_ASSERT_EQUAL(uint32_t(1), s.min());
        CPPUNIT_ASSERT_EQUAL(uint32_t(1), s.max());
        s.add( 2 );
        CPPUNIT_ASSERT_EQUAL(uint32_t(1), s.min());
        CPPUNIT_ASSERT_EQUAL(uint32_t(2), s.max());
        s.add( 0 );
        CPPUNIT_ASSERT_EQUAL( uint32_t(0), s.min());
        CPPUNIT_ASSERT_EQUAL( uint32_t(2), s.max());
    }
}


void
TestStats::testExtrema()
{
    {// trivial test
        openvdb::math::Extrema s;
        s.add(0);
        s.add(1);
        CPPUNIT_ASSERT_EQUAL(2, int(s.size()));
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, s.min(), 0.000001);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, s.max(), 0.000001);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, s.range(), 0.000001);
        //s.print("test");
    }
    {// non-trivial test
        openvdb::math::Extrema s;
        const int data[5]={600, 470, 170, 430, 300};
        for (int i=0; i<5; ++i) s.add(data[i]);
        CPPUNIT_ASSERT_EQUAL(5, int(s.size()));
        CPPUNIT_ASSERT_DOUBLES_EQUAL(data[2], s.min(), 0.000001);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(data[0], s.max(), 0.000001);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(data[0]-data[2], s.range(), 0.000001);
        //s.print("test");
    }
    {// non-trivial test of Extrema::add(Extrema)
        openvdb::math::Extrema s, t;
        const int data[5]={600, 470, 170, 430, 300};
        for (int i=0; i<3; ++i) s.add(data[i]);
        for (int i=3; i<5; ++i) t.add(data[i]);
        s.add(t);
        CPPUNIT_ASSERT_EQUAL(5, int(s.size()));
        CPPUNIT_ASSERT_DOUBLES_EQUAL(data[2], s.min(), 0.000001);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(data[0], s.max(), 0.000001);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(data[0]-data[2], s.range(), 0.000001);
        //s.print("test");
    }
    {// Trivial test of Extrema::add(value, n)
        openvdb::math::Extrema s;
        const double val = 3.45;
        const uint64_t n = 57;
        s.add(val, 57);
        CPPUNIT_ASSERT_EQUAL(n, s.size());
        CPPUNIT_ASSERT_DOUBLES_EQUAL(val, s.min(), 0.000001);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(val, s.max(), 0.000001);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, s.range(), 0.000001);
    }
    {// Test 1 of Extrema::add(value), Extrema::add(value, n) and Extrema::add(Extrema)
        openvdb::math::Extrema s, t;
        const double val1 = 1.0, val2 = 3.0;
        const uint64_t n1 = 1, n2 =1;
        s.add(val1,  n1);
        CPPUNIT_ASSERT_EQUAL(uint64_t(n1), s.size());
        CPPUNIT_ASSERT_DOUBLES_EQUAL(val1, s.min(),      0.000001);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(val1, s.max(),      0.000001);
        for (uint64_t i=0; i<n2; ++i) t.add(val2);
        s.add(t);
        CPPUNIT_ASSERT_EQUAL(uint64_t(n2), t.size());
        CPPUNIT_ASSERT_DOUBLES_EQUAL(val2, t.min(),      0.000001);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(val2, t.max(),      0.000001);

        CPPUNIT_ASSERT_EQUAL(uint64_t(n1+n2), s.size());
        CPPUNIT_ASSERT_DOUBLES_EQUAL(val1,    s.min(),  0.000001);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(val2,    s.max(),  0.000001);
    }
    {// Non-trivial test of Extrema::add(value, n)
        openvdb::math::Extrema s;
        s.add(3.45,  6);
        s.add(1.39,  2);
        s.add(2.56, 13);
        s.add(0.03);
        openvdb::math::Extrema t;
        for (int i=0; i< 6; ++i) t.add(3.45);
        for (int i=0; i< 2; ++i) t.add(1.39);
        for (int i=0; i<13; ++i) t.add(2.56);
        t.add(0.03);
        CPPUNIT_ASSERT_EQUAL(s.size(), t.size());
        CPPUNIT_ASSERT_DOUBLES_EQUAL(s.min(), t.min(),  0.000001);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(s.max(), t.max(),  0.000001);
    }
}

void
TestStats::testStats()
{
    {// trivial test
        openvdb::math::Stats s;
        s.add(0);
        s.add(1);
        CPPUNIT_ASSERT_EQUAL(2, int(s.size()));
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, s.min(), 0.000001);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, s.max(), 0.000001);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.5, s.mean(), 0.000001);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.25, s.variance(), 0.000001);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.5, s.stdDev(), 0.000001);
        //s.print("test");
    }
    {// non-trivial test
        openvdb::math::Stats s;
        const int data[5]={600, 470, 170, 430, 300};
        for (int i=0; i<5; ++i) s.add(data[i]);
        double sum = 0.0;
        for (int i=0; i<5; ++i) sum += data[i];
        const double mean = sum/5.0;
        sum = 0.0;
        for (int i=0; i<5; ++i) sum += (data[i]-mean)*(data[i]-mean);
        const double var = sum/5.0;
        CPPUNIT_ASSERT_EQUAL(5, int(s.size()));
        CPPUNIT_ASSERT_DOUBLES_EQUAL(data[2], s.min(), 0.000001);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(data[0], s.max(), 0.000001);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(mean, s.mean(), 0.000001);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(var, s.variance(), 0.000001);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(sqrt(var), s.stdDev(),  0.000001);
        //s.print("test");
    }
    {// non-trivial test of Stats::add(Stats)
        openvdb::math::Stats s, t;
        const int data[5]={600, 470, 170, 430, 300};
        for (int i=0; i<3; ++i) s.add(data[i]);
        for (int i=3; i<5; ++i) t.add(data[i]);
        s.add(t);
        double sum = 0.0;
        for (int i=0; i<5; ++i) sum += data[i];
        const double mean = sum/5.0;
        sum = 0.0;
        for (int i=0; i<5; ++i) sum += (data[i]-mean)*(data[i]-mean);
        const double var = sum/5.0;
        CPPUNIT_ASSERT_EQUAL(5, int(s.size()));
        CPPUNIT_ASSERT_DOUBLES_EQUAL(data[2], s.min(), 0.000001);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(data[0], s.max(), 0.000001);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(mean, s.mean(), 0.000001);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(var, s.variance(), 0.000001);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(sqrt(var), s.stdDev(),  0.000001);
        //s.print("test");
    }
    {// Trivial test of Stats::add(value, n)
        openvdb::math::Stats s;
        const double val = 3.45;
        const uint64_t n = 57;
        s.add(val, 57);
        CPPUNIT_ASSERT_EQUAL(n, s.size());
        CPPUNIT_ASSERT_DOUBLES_EQUAL(val, s.min(), 0.000001);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(val, s.max(), 0.000001);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(val, s.mean(), 0.000001);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, s.variance(), 0.000001);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, s.stdDev(),  0.000001);
    }
    {// Test 1 of Stats::add(value), Stats::add(value, n) and Stats::add(Stats)
        openvdb::math::Stats s, t;
        const double val1 = 1.0, val2 = 3.0, sum = val1 + val2;
        const uint64_t n1 = 1, n2 =1;
        s.add(val1,  n1);
        CPPUNIT_ASSERT_EQUAL(uint64_t(n1), s.size());
        CPPUNIT_ASSERT_DOUBLES_EQUAL(val1, s.min(),      0.000001);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(val1, s.max(),      0.000001);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(val1, s.mean(),     0.000001);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0,  s.variance(), 0.000001);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0,  s.stdDev(),   0.000001);
        for (uint64_t i=0; i<n2; ++i) t.add(val2);
        s.add(t);
        CPPUNIT_ASSERT_EQUAL(uint64_t(n2), t.size());
        CPPUNIT_ASSERT_DOUBLES_EQUAL(val2, t.min(),      0.000001);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(val2, t.max(),      0.000001);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(val2, t.mean(),     0.000001);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0,  t.variance(), 0.000001);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0,  t.stdDev(),   0.000001);
        CPPUNIT_ASSERT_EQUAL(uint64_t(n1+n2), s.size());
        CPPUNIT_ASSERT_DOUBLES_EQUAL(val1,    s.min(),  0.000001);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(val2,    s.max(),  0.000001);
        const double mean = sum/double(n1+n2);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(mean,    s.mean(), 0.000001);
        double var = 0.0;
        for (uint64_t i=0; i<n1; ++i) var += openvdb::math::Pow2(val1-mean);
        for (uint64_t i=0; i<n2; ++i) var += openvdb::math::Pow2(val2-mean);
        var /= double(n1+n2);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(var, s.variance(), 0.000001);
    }
    {// Test 2 of Stats::add(value), Stats::add(value, n) and Stats::add(Stats)
        openvdb::math::Stats s, t;
        const double val1 = 1.0, val2 = 3.0, sum = val1 + val2;
        const uint64_t n1 = 1, n2 =1;
        for (uint64_t i=0; i<n1; ++i) s.add(val1);
        CPPUNIT_ASSERT_EQUAL(uint64_t(n1), s.size());
        CPPUNIT_ASSERT_DOUBLES_EQUAL(val1, s.min(),      0.000001);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(val1, s.max(),      0.000001);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(val1, s.mean(),     0.000001);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0,  s.variance(), 0.000001);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0,  s.stdDev(),   0.000001);
        t.add(val2,  n2);
        CPPUNIT_ASSERT_EQUAL(uint64_t(n2), t.size());
        CPPUNIT_ASSERT_DOUBLES_EQUAL(val2, t.min(),      0.000001);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(val2, t.max(),      0.000001);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(val2, t.mean(),     0.000001);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0,  t.variance(), 0.000001);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0,  t.stdDev(),   0.000001);
        s.add(t);
        CPPUNIT_ASSERT_EQUAL(uint64_t(n1+n2), s.size());
        CPPUNIT_ASSERT_DOUBLES_EQUAL(val1,    s.min(),  0.000001);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(val2,    s.max(),  0.000001);
        const double mean = sum/double(n1+n2);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(mean,    s.mean(), 0.000001);
        double var = 0.0;
        for (uint64_t i=0; i<n1; ++i) var += openvdb::math::Pow2(val1-mean);
        for (uint64_t i=0; i<n2; ++i) var += openvdb::math::Pow2(val2-mean);
        var /= double(n1+n2);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(var, s.variance(), 0.000001);
    }
    {// Non-trivial test of Stats::add(value, n) and Stats::add(Stats)
        openvdb::math::Stats s;
        s.add(3.45,  6);
        s.add(1.39,  2);
        s.add(2.56, 13);
        s.add(0.03);
        openvdb::math::Stats t;
        for (int i=0; i< 6; ++i) t.add(3.45);
        for (int i=0; i< 2; ++i) t.add(1.39);
        for (int i=0; i<13; ++i) t.add(2.56);
        t.add(0.03);
        CPPUNIT_ASSERT_EQUAL(s.size(), t.size());
        CPPUNIT_ASSERT_DOUBLES_EQUAL(s.min(), t.min(),  0.000001);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(s.max(), t.max(),  0.000001);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(s.mean(),t.mean(), 0.000001);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(s.variance(), t.variance(), 0.000001);
    }
    {// Non-trivial test of Stats::add(value, n)
        openvdb::math::Stats s;
        s.add(3.45,  6);
        s.add(1.39,  2);
        s.add(2.56, 13);
        s.add(0.03);
        openvdb::math::Stats t;
        for (int i=0; i< 6; ++i) t.add(3.45);
        for (int i=0; i< 2; ++i) t.add(1.39);
        for (int i=0; i<13; ++i) t.add(2.56);
        t.add(0.03);
        CPPUNIT_ASSERT_EQUAL(s.size(), t.size());
        CPPUNIT_ASSERT_DOUBLES_EQUAL(s.min(), t.min(),  0.000001);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(s.max(), t.max(),  0.000001);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(s.mean(),t.mean(), 0.000001);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(s.variance(), t.variance(), 0.000001);
    }

    //std::cerr << "\nCompleted TestStats::testStats!\n" << std::endl;
}

void
TestStats::testHistogram()
{
     {// Histogram test
        openvdb::math::Stats s;
        const int data[5]={600, 470, 170, 430, 300};
        for (int i=0; i<5; ++i) s.add(data[i]);
        openvdb::math::Histogram h(s, 10);
        for (int i=0; i<5; ++i) CPPUNIT_ASSERT(h.add(data[i]));
        int bin[10]={0};
        for (int i=0; i<5; ++i) {
            for (int j=0; j<10; ++j) if (data[i] >= h.min(j) && data[i] < h.max(j)) bin[j]++;
        }
        for (int i=0; i<5; ++i)  CPPUNIT_ASSERT_EQUAL(bin[i],int(h.count(i)));
        //h.print("test");
    }
    {//Test print of Histogram
        openvdb::math::Stats s;
        const int N=500000;
        for (int i=0; i<N; ++i) s.add(N/2-i);
        //s.print("print-test");
        openvdb::math::Histogram h(s, 25);
        for (int i=0; i<N; ++i) CPPUNIT_ASSERT(h.add(N/2-i));
        //h.print("print-test");
    }
}

namespace {

struct GradOp
{
    typedef openvdb::FloatGrid GridT;

    GridT::ConstAccessor acc;

    GradOp(const GridT& grid): acc(grid.getConstAccessor()) {}

    template <typename StatsT>
    void operator()(const GridT::ValueOnCIter& it, StatsT& stats) const
    {
        typedef openvdb::math::ISGradient<openvdb::math::FD_1ST> GradT;
        if (it.isVoxelValue()) {
            stats.add(GradT::result(acc, it.getCoord()).length());
        } else {
            openvdb::CoordBBox bbox = it.getBoundingBox();
            openvdb::Coord xyz;
            int &x = xyz[0], &y = xyz[1], &z = xyz[2];
            for (x = bbox.min()[0]; x <= bbox.max()[0]; ++x) {
                for (y = bbox.min()[1]; y <= bbox.max()[1]; ++y) {
                    for (z = bbox.min()[2]; z <= bbox.max()[2]; ++z) {
                        stats.add(GradT::result(acc, xyz).length());
                    }
                }
            }
        }
    }
};

} // unnamed namespace

void
TestStats::testGridExtrema()
{
    using namespace openvdb;

    const int DIM = 109;
    {
        const float background = 0.0;
        FloatGrid grid(background);
        {
            // Compute active value statistics for a grid with a single active voxel.
            grid.tree().setValue(Coord(0), /*value=*/42.0);
            math::Extrema ex = tools::extrema(grid.cbeginValueOn());

            CPPUNIT_ASSERT_DOUBLES_EQUAL(42.0, ex.min(),  /*tolerance=*/0.0);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(42.0, ex.max(),  /*tolerance=*/0.0);

            // Compute inactive value statistics for a grid with only background voxels.
            grid.tree().setValueOff(Coord(0), background);
            ex = tools::extrema(grid.cbeginValueOff());

            CPPUNIT_ASSERT_DOUBLES_EQUAL(background, ex.min(),  /*tolerance=*/0.0);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(background, ex.max(),  /*tolerance=*/0.0);
        }

        // Compute active value statistics for a grid with two active voxel populations
        // of the same size but two different values.
        grid.fill(CoordBBox::createCube(Coord(0), DIM), /*value=*/1.0);
        grid.fill(CoordBBox::createCube(Coord(-300), DIM), /*value=*/-3.0);

        CPPUNIT_ASSERT_EQUAL(Index64(2 * DIM * DIM * DIM), grid.activeVoxelCount());

        for (int threaded = 0; threaded <= 1; ++threaded) {
            math::Extrema ex = tools::extrema(grid.cbeginValueOn(), threaded);

            CPPUNIT_ASSERT_DOUBLES_EQUAL(double(-3.0), ex.min(),  /*tolerance=*/0.0);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(double(1.0),  ex.max(),  /*tolerance=*/0.0);
        }

        // Compute active value statistics for just the positive values.
        for (int threaded = 0; threaded <= 1; ++threaded) {
            struct Local {
                static void addIfPositive(const FloatGrid::ValueOnCIter& it, math::Extrema& ex)
                {
                    const float f = *it;
                    if (f > 0.0) {
                        if (it.isVoxelValue()) ex.add(f);
                        else ex.add(f, it.getVoxelCount());
                    }
                }
            };
            math::Extrema ex =
                tools::extrema(grid.cbeginValueOn(), &Local::addIfPositive, threaded);

            CPPUNIT_ASSERT_DOUBLES_EQUAL(double(1.0), ex.min(),  /*tolerance=*/0.0);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(double(1.0), ex.max(),  /*tolerance=*/0.0);
        }

        // Compute active value statistics for the first-order gradient.
        for (int threaded = 0; threaded <= 1; ++threaded) {
            // First, using a custom ValueOp...
            math::Extrema ex = tools::extrema(grid.cbeginValueOn(), GradOp(grid), threaded);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(double(0.0), ex.min(), /*tolerance=*/0.0);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(
                double(9.0 + 9.0 + 9.0), ex.max() * ex.max(), /*tol=*/1.0e-3);
                // max gradient is (dx, dy, dz) = (-3 - 0, -3 - 0, -3 - 0)

            // ...then using tools::opStatistics().
            typedef math::ISOpMagnitude<math::ISGradient<math::FD_1ST> > MathOp;
            ex = tools::opExtrema(grid.cbeginValueOn(), MathOp(), threaded);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(double(0.0), ex.min(), /*tolerance=*/0.0);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(
                double(9.0 + 9.0 + 9.0), ex.max() * ex.max(), /*tolerance=*/1.0e-3);
                // max gradient is (dx, dy, dz) = (-3 - 0, -3 - 0, -3 - 0)
        }
    }
    {
        const Vec3s background(0.0);
        Vec3SGrid grid(background);

        // Compute active vector magnitude statistics for a vector-valued grid
        // with two active voxel populations of the same size but two different values.
        grid.fill(CoordBBox::createCube(Coord(0), DIM),    Vec3s(3.0, 0.0, 4.0)); // length = 5
        grid.fill(CoordBBox::createCube(Coord(-300), DIM), Vec3s(1.0, 2.0, 2.0)); // length = 3

        CPPUNIT_ASSERT_EQUAL(Index64(2 * DIM * DIM * DIM), grid.activeVoxelCount());

        for (int threaded = 0; threaded <= 1; ++threaded) {
            math::Extrema ex = tools::extrema(grid.cbeginValueOn(), threaded);

            CPPUNIT_ASSERT_DOUBLES_EQUAL(double(3.0), ex.min(),  /*tolerance=*/0.0);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(double(5.0), ex.max(),  /*tolerance=*/0.0);
        }
    }
}

void
TestStats::testGridStats()
{
    using namespace openvdb;

    const int DIM = 109;
    {
        const float background = 0.0;
        FloatGrid grid(background);
        {
            // Compute active value statistics for a grid with a single active voxel.
            grid.tree().setValue(Coord(0), /*value=*/42.0);
            math::Stats stats = tools::statistics(grid.cbeginValueOn());

            CPPUNIT_ASSERT_DOUBLES_EQUAL(42.0, stats.min(),  /*tolerance=*/0.0);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(42.0, stats.max(),  /*tolerance=*/0.0);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(42.0, stats.mean(), /*tolerance=*/1.0e-8);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0,  stats.variance(), /*tolerance=*/1.0e-8);

            // Compute inactive value statistics for a grid with only background voxels.
            grid.tree().setValueOff(Coord(0), background);
            stats = tools::statistics(grid.cbeginValueOff());

            CPPUNIT_ASSERT_DOUBLES_EQUAL(background, stats.min(),  /*tolerance=*/0.0);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(background, stats.max(),  /*tolerance=*/0.0);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(background, stats.mean(), /*tolerance=*/1.0e-8);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0,        stats.variance(), /*tolerance=*/1.0e-8);
        }

        // Compute active value statistics for a grid with two active voxel populations
        // of the same size but two different values.
        grid.fill(CoordBBox::createCube(Coord(0), DIM), /*value=*/1.0);
        grid.fill(CoordBBox::createCube(Coord(-300), DIM), /*value=*/-3.0);

        CPPUNIT_ASSERT_EQUAL(Index64(2 * DIM * DIM * DIM), grid.activeVoxelCount());

        for (int threaded = 0; threaded <= 1; ++threaded) {
            math::Stats stats = tools::statistics(grid.cbeginValueOn(), threaded);

            CPPUNIT_ASSERT_DOUBLES_EQUAL(double(-3.0), stats.min(),  /*tolerance=*/0.0);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(double(1.0),  stats.max(),  /*tolerance=*/0.0);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(double(-1.0), stats.mean(), /*tolerance=*/1.0e-8);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(double(4.0),  stats.variance(), /*tolerance=*/1.0e-8);
        }

        // Compute active value statistics for just the positive values.
        for (int threaded = 0; threaded <= 1; ++threaded) {
            struct Local {
                static void addIfPositive(const FloatGrid::ValueOnCIter& it, math::Stats& stats)
                {
                    const float f = *it;
                    if (f > 0.0) {
                        if (it.isVoxelValue()) stats.add(f);
                        else stats.add(f, it.getVoxelCount());
                    }
                }
            };
            math::Stats stats =
                tools::statistics(grid.cbeginValueOn(), &Local::addIfPositive, threaded);

            CPPUNIT_ASSERT_DOUBLES_EQUAL(double(1.0), stats.min(),  /*tolerance=*/0.0);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(double(1.0), stats.max(),  /*tolerance=*/0.0);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(double(1.0), stats.mean(), /*tolerance=*/1.0e-8);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(double(0.0), stats.variance(), /*tolerance=*/1.0e-8);
        }

        // Compute active value statistics for the first-order gradient.
        for (int threaded = 0; threaded <= 1; ++threaded) {
            // First, using a custom ValueOp...
            math::Stats stats = tools::statistics(grid.cbeginValueOn(), GradOp(grid), threaded);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(double(0.0), stats.min(), /*tolerance=*/0.0);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(
                double(9.0 + 9.0 + 9.0), stats.max() * stats.max(), /*tol=*/1.0e-3);
                // max gradient is (dx, dy, dz) = (-3 - 0, -3 - 0, -3 - 0)

            // ...then using tools::opStatistics().
            typedef math::ISOpMagnitude<math::ISGradient<math::FD_1ST> > MathOp;
            stats = tools::opStatistics(grid.cbeginValueOn(), MathOp(), threaded);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(double(0.0), stats.min(), /*tolerance=*/0.0);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(
                double(9.0 + 9.0 + 9.0), stats.max() * stats.max(), /*tolerance=*/1.0e-3);
                // max gradient is (dx, dy, dz) = (-3 - 0, -3 - 0, -3 - 0)
        }
    }
    {
        const Vec3s background(0.0);
        Vec3SGrid grid(background);

        // Compute active vector magnitude statistics for a vector-valued grid
        // with two active voxel populations of the same size but two different values.
        grid.fill(CoordBBox::createCube(Coord(0), DIM),    Vec3s(3.0, 0.0, 4.0)); // length = 5
        grid.fill(CoordBBox::createCube(Coord(-300), DIM), Vec3s(1.0, 2.0, 2.0)); // length = 3

        CPPUNIT_ASSERT_EQUAL(Index64(2 * DIM * DIM * DIM), grid.activeVoxelCount());

        for (int threaded = 0; threaded <= 1; ++threaded) {
            math::Stats stats = tools::statistics(grid.cbeginValueOn(), threaded);

            CPPUNIT_ASSERT_DOUBLES_EQUAL(double(3.0), stats.min(),  /*tolerance=*/0.0);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(double(5.0), stats.max(),  /*tolerance=*/0.0);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(double(4.0), stats.mean(), /*tolerance=*/1.0e-8);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(double(1.0),  stats.variance(), /*tolerance=*/1.0e-8);
        }
    }
}


namespace {

template<typename OpT, typename GridT>
inline void
doTestGridOperatorStats(const GridT& grid, const OpT& op)
{
    openvdb::math::Stats serialStats =
        openvdb::tools::opStatistics(grid.cbeginValueOn(), op, /*threaded=*/false);

    openvdb::math::Stats parallelStats =
        openvdb::tools::opStatistics(grid.cbeginValueOn(), op, /*threaded=*/true);

    // Verify that the results from threaded and serial runs are equivalent.
    CPPUNIT_ASSERT_EQUAL(serialStats.size(), parallelStats.size());
    ASSERT_DOUBLES_EXACTLY_EQUAL(serialStats.min(), parallelStats.min());
    ASSERT_DOUBLES_EXACTLY_EQUAL(serialStats.max(), parallelStats.max());
    CPPUNIT_ASSERT_DOUBLES_EQUAL(serialStats.mean(), parallelStats.mean(), /*tolerance=*/1.0e-6);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(serialStats.variance(), parallelStats.variance(), 1.0e-6);
}

}

void
TestStats::testGridOperatorStats()
{
    using namespace openvdb;

    typedef math::UniformScaleMap MapType;
    MapType map;

    const int DIM = 109;
    {
        // Test operations on a scalar grid.
        const float background = 0.0;
        FloatGrid grid(background);
        grid.fill(CoordBBox::createCube(Coord(0), DIM), /*value=*/1.0);
        grid.fill(CoordBBox::createCube(Coord(-300), DIM), /*value=*/-3.0);

        {   // Magnitude of gradient computed via first-order differencing
            typedef math::MapAdapter<MapType,
                math::OpMagnitude<math::Gradient<MapType, math::FD_1ST>, MapType>, double> OpT;
            doTestGridOperatorStats(grid, OpT(map));
        }
        {   // Magnitude of index-space gradient computed via first-order differencing
            typedef math::ISOpMagnitude<math::ISGradient<math::FD_1ST> > OpT;
            doTestGridOperatorStats(grid, OpT());
        }
        {   // Laplacian of index-space gradient computed via second-order central differencing
            typedef math::ISLaplacian<math::CD_SECOND> OpT;
            doTestGridOperatorStats(grid, OpT());
        }
    }
    {
        // Test operations on a vector grid.
        const Vec3s background(0.0);
        Vec3SGrid grid(background);
        grid.fill(CoordBBox::createCube(Coord(0), DIM),    Vec3s(3.0, 0.0, 4.0)); // length = 5
        grid.fill(CoordBBox::createCube(Coord(-300), DIM), Vec3s(1.0, 2.0, 2.0)); // length = 3

        {   // Divergence computed via first-order differencing
            typedef math::MapAdapter<MapType,
                math::Divergence<MapType, math::FD_1ST>, double> OpT;
            doTestGridOperatorStats(grid, OpT(map));
        }
        {   // Magnitude of curl computed via first-order differencing
            typedef math::MapAdapter<MapType,
                math::OpMagnitude<math::Curl<MapType, math::FD_1ST>, MapType>, double> OpT;
            doTestGridOperatorStats(grid, OpT(map));
        }
        {   // Magnitude of index-space curl computed via first-order differencing
            typedef math::ISOpMagnitude<math::ISCurl<math::FD_1ST> > OpT;
            doTestGridOperatorStats(grid, OpT());
        }
    }
}


void
TestStats::testGridHistogram()
{
    using namespace openvdb;

    const int DIM = 109;
    {
        const float background = 0.0;
        FloatGrid grid(background);
        {
            const double value = 42.0;

            // Compute a histogram of the active values of a grid with a single active voxel.
            grid.tree().setValue(Coord(0), value);
            math::Histogram hist = tools::histogram(grid.cbeginValueOn(),
                /*min=*/0.0, /*max=*/100.0);

            for (int i = 0, N = int(hist.numBins()); i < N; ++i) {
                uint64_t expected = ((hist.min(i) <= value && value <= hist.max(i)) ? 1 : 0);
                CPPUNIT_ASSERT_EQUAL(expected, hist.count(i));
            }
        }

        // Compute a histogram of the active values of a grid with two
        // active voxel populations of the same size but two different values.
        grid.fill(CoordBBox::createCube(Coord(0), DIM), /*value=*/1.0);
        grid.fill(CoordBBox::createCube(Coord(-300), DIM), /*value=*/3.0);

        CPPUNIT_ASSERT_EQUAL(uint64_t(2 * DIM * DIM * DIM), grid.activeVoxelCount());

        for (int threaded = 0; threaded <= 1; ++threaded) {
            math::Histogram hist = tools::histogram(grid.cbeginValueOn(),
                /*min=*/0.0, /*max=*/10.0, /*numBins=*/9, threaded);

            CPPUNIT_ASSERT_EQUAL(Index64(2 * DIM * DIM * DIM), hist.size());
            for (int i = 0, N = int(hist.numBins()); i < N; ++i) {
                if (i == 0 || i == 2) {
                    CPPUNIT_ASSERT_EQUAL(uint64_t(DIM * DIM * DIM), hist.count(i));
                } else {
                    CPPUNIT_ASSERT_EQUAL(uint64_t(0), hist.count(i));
                }
            }
        }
    }
    {
        const Vec3s background(0.0);
        Vec3SGrid grid(background);

        // Compute a histogram of vector magnitudes of the active values of a
        // vector-valued grid with two active voxel populations of the same size
        // but two different values.
        grid.fill(CoordBBox::createCube(Coord(0), DIM),    Vec3s(3.0, 0.0, 4.0)); // length = 5
        grid.fill(CoordBBox::createCube(Coord(-300), DIM), Vec3s(1.0, 2.0, 2.0)); // length = 3

        CPPUNIT_ASSERT_EQUAL(Index64(2 * DIM * DIM * DIM), grid.activeVoxelCount());

        for (int threaded = 0; threaded <= 1; ++threaded) {
            math::Histogram hist = tools::histogram(grid.cbeginValueOn(),
                /*min=*/0.0, /*max=*/10.0, /*numBins=*/9, threaded);

            CPPUNIT_ASSERT_EQUAL(Index64(2 * DIM * DIM * DIM), hist.size());
            for (int i = 0, N = int(hist.numBins()); i < N; ++i) {
                if (i == 2 || i == 4) {
                    CPPUNIT_ASSERT_EQUAL(uint64_t(DIM * DIM * DIM), hist.count(i));
                } else {
                    CPPUNIT_ASSERT_EQUAL(uint64_t(0), hist.count(i));
                }
            }
        }
    }
}

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
