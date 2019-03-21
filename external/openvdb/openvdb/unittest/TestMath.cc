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
#include <openvdb/math/Math.h>
#include <openvdb/Types.h>
#include <type_traits>
#include <vector>


class TestMath: public CppUnit::TestCase
{
public:
    CPPUNIT_TEST_SUITE(TestMath);
    CPPUNIT_TEST(testAll);
    CPPUNIT_TEST(testRandomInt);
    CPPUNIT_TEST(testRandom01);
    CPPUNIT_TEST(testMinMaxIndex);
    CPPUNIT_TEST_SUITE_END();

    void testAll();
    void testRandomInt();
    void testRandom01();
    void testMinMaxIndex();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestMath);


// This suite of tests obviously needs to be expanded!
void
TestMath::testAll()
{
    using namespace openvdb;

    {// Sign
        CPPUNIT_ASSERT_EQUAL(math::Sign( 3   ), 1);
        CPPUNIT_ASSERT_EQUAL(math::Sign(-1.0 ),-1);
        CPPUNIT_ASSERT_EQUAL(math::Sign( 0.0f), 0);
    }
    {// SignChange
        CPPUNIT_ASSERT( math::SignChange( -1, 1));
        CPPUNIT_ASSERT(!math::SignChange( 0.0f, 0.5f));
        CPPUNIT_ASSERT( math::SignChange( 0.0f,-0.5f));
        CPPUNIT_ASSERT( math::SignChange(-0.1, 0.0001));

    }
    {// isApproxZero
        CPPUNIT_ASSERT( math::isApproxZero( 0.0f));
        CPPUNIT_ASSERT(!math::isApproxZero( 9.0e-6f));
        CPPUNIT_ASSERT(!math::isApproxZero(-9.0e-6f));
        CPPUNIT_ASSERT( math::isApproxZero( 9.0e-9f));
        CPPUNIT_ASSERT( math::isApproxZero(-9.0e-9f));
        CPPUNIT_ASSERT( math::isApproxZero( 0.01, 0.1));
    }
    {// Cbrt
        const double a = math::Cbrt(3.0);
        CPPUNIT_ASSERT(math::isApproxEqual(a*a*a, 3.0, 1e-6));
    }
    {// isNegative
        CPPUNIT_ASSERT(!std::is_signed<unsigned int>::value);
        CPPUNIT_ASSERT(std::is_signed<int>::value);
        CPPUNIT_ASSERT(!std::is_signed<bool>::value);
        //CPPUNIT_ASSERT(std::is_signed<double>::value);//fails!
        //CPPUNIT_ASSERT(std::is_signed<float>::value);//fails!

        CPPUNIT_ASSERT( math::isNegative(-1.0f));
        CPPUNIT_ASSERT(!math::isNegative( 1.0f));
        CPPUNIT_ASSERT( math::isNegative(-1.0));
        CPPUNIT_ASSERT(!math::isNegative( 1.0));
        CPPUNIT_ASSERT(!math::isNegative(true));
        CPPUNIT_ASSERT(!math::isNegative(false));
        CPPUNIT_ASSERT(!math::isNegative(1u));
        CPPUNIT_ASSERT( math::isNegative(-1));
        CPPUNIT_ASSERT(!math::isNegative( 1));
    }
    {// zeroVal
        CPPUNIT_ASSERT_EQUAL(zeroVal<bool>(), false);
        CPPUNIT_ASSERT_EQUAL(zeroVal<int>(), int(0));
        CPPUNIT_ASSERT_EQUAL(zeroVal<float>(), 0.0f);
        CPPUNIT_ASSERT_EQUAL(zeroVal<double>(), 0.0);
        CPPUNIT_ASSERT_EQUAL(zeroVal<Vec3i>(), Vec3i(0,0,0));
        CPPUNIT_ASSERT_EQUAL(zeroVal<Vec3s>(), Vec3s(0,0,0));
        CPPUNIT_ASSERT_EQUAL(zeroVal<Vec3d>(), Vec3d(0,0,0));
        CPPUNIT_ASSERT_EQUAL(zeroVal<Quats>(), Quats::zero());
        CPPUNIT_ASSERT_EQUAL(zeroVal<Quatd>(), Quatd::zero());
        CPPUNIT_ASSERT_EQUAL(zeroVal<Mat3s>(), Mat3s::zero());
        CPPUNIT_ASSERT_EQUAL(zeroVal<Mat3d>(), Mat3d::zero());
        CPPUNIT_ASSERT_EQUAL(zeroVal<Mat4s>(), Mat4s::zero());
        CPPUNIT_ASSERT_EQUAL(zeroVal<Mat4d>(), Mat4d::zero());
    }
}


void
TestMath::testRandomInt()
{
    using openvdb::math::RandomInt;

    int imin = -3, imax = 11;
    RandomInt rnd(/*seed=*/42, imin, imax);

    // Generate a sequence of random integers and verify that they all fall
    // in the interval [imin, imax].
    std::vector<int> seq(100);
    for (int i = 0; i < 100; ++i) {
        seq[i] = rnd();
        CPPUNIT_ASSERT(seq[i] >= imin);
        CPPUNIT_ASSERT(seq[i] <= imax);
    }

    // Verify that generators with the same seed produce the same sequence.
    rnd = RandomInt(42, imin, imax);
    for (int i = 0; i < 100; ++i) {
        int r = rnd();
        CPPUNIT_ASSERT_EQUAL(seq[i], r);
    }

    // Verify that generators with different seeds produce different sequences.
    rnd = RandomInt(101, imin, imax);
    std::vector<int> newSeq(100);
    for (int i = 0; i < 100; ++i) newSeq[i] = rnd();
    CPPUNIT_ASSERT(newSeq != seq);

    // Temporarily change the range.
    imin = -5; imax = 6;
    for (int i = 0; i < 100; ++i) {
        int r = rnd(imin, imax);
        CPPUNIT_ASSERT(r >= imin);
        CPPUNIT_ASSERT(r <= imax);
    }
    // Verify that the range change was temporary.
    imin = -3; imax = 11;
    for (int i = 0; i < 100; ++i) {
        int r = rnd();
        CPPUNIT_ASSERT(r >= imin);
        CPPUNIT_ASSERT(r <= imax);
    }

    // Permanently change the range.
    imin = -5; imax = 6;
    rnd.setRange(imin, imax);
    for (int i = 0; i < 100; ++i) {
        int r = rnd();
        CPPUNIT_ASSERT(r >= imin);
        CPPUNIT_ASSERT(r <= imax);
    }

    // Verify that it is OK to specify imin > imax (they are automatically swapped).
    imin = 5; imax = -6;
    rnd.setRange(imin, imax);

    rnd = RandomInt(42, imin, imax);
}


void
TestMath::testRandom01()
{
    using openvdb::math::Random01;
    using openvdb::math::isApproxEqual;

    Random01 rnd(/*seed=*/42);

    // Generate a sequence of random numbers and verify that they all fall
    // in the interval [0, 1).
    std::vector<Random01::ValueType> seq(100);
    for (int i = 0; i < 100; ++i) {
        seq[i] = rnd();
        CPPUNIT_ASSERT(seq[i] >= 0.0);
        CPPUNIT_ASSERT(seq[i] < 1.0);
    }

    // Verify that generators with the same seed produce the same sequence.
    rnd = Random01(42);
    for (int i = 0; i < 100; ++i) {
        CPPUNIT_ASSERT_DOUBLES_EQUAL(seq[i], rnd(), /*tolerance=*/1.0e-6);
    }

    // Verify that generators with different seeds produce different sequences.
    rnd = Random01(101);
    bool allEqual = true;
    for (int i = 0; allEqual && i < 100; ++i) {
        if (!isApproxEqual(rnd(), seq[i])) allEqual = false;
    }
    CPPUNIT_ASSERT(!allEqual);
}

void
TestMath::testMinMaxIndex()
{
    const openvdb::Vec3R a(-1, 2, 0);
    CPPUNIT_ASSERT_EQUAL(size_t(0), openvdb::math::MinIndex(a));
    CPPUNIT_ASSERT_EQUAL(size_t(1), openvdb::math::MaxIndex(a));
    const openvdb::Vec3R b(-1, -2, 0);
    CPPUNIT_ASSERT_EQUAL(size_t(1), openvdb::math::MinIndex(b));
    CPPUNIT_ASSERT_EQUAL(size_t(2), openvdb::math::MaxIndex(b));
    const openvdb::Vec3R c(5, 2, 1);
    CPPUNIT_ASSERT_EQUAL(size_t(2), openvdb::math::MinIndex(c));
    CPPUNIT_ASSERT_EQUAL(size_t(0), openvdb::math::MaxIndex(c));
    const openvdb::Vec3R d(0, 0, 1);
    CPPUNIT_ASSERT_EQUAL(size_t(1), openvdb::math::MinIndex(d));
    CPPUNIT_ASSERT_EQUAL(size_t(2), openvdb::math::MaxIndex(d));
    const openvdb::Vec3R e(1, 0, 0);
    CPPUNIT_ASSERT_EQUAL(size_t(2), openvdb::math::MinIndex(e));
    CPPUNIT_ASSERT_EQUAL(size_t(0), openvdb::math::MaxIndex(e));
    const openvdb::Vec3R f(0, 1, 0);
    CPPUNIT_ASSERT_EQUAL(size_t(2), openvdb::math::MinIndex(f));
    CPPUNIT_ASSERT_EQUAL(size_t(1), openvdb::math::MaxIndex(f));
    const openvdb::Vec3R g(1, 1, 0);
    CPPUNIT_ASSERT_EQUAL(size_t(2), openvdb::math::MinIndex(g));
    CPPUNIT_ASSERT_EQUAL(size_t(1), openvdb::math::MaxIndex(g));
    const openvdb::Vec3R h(1, 0, 1);
    CPPUNIT_ASSERT_EQUAL(size_t(1), openvdb::math::MinIndex(h));
    CPPUNIT_ASSERT_EQUAL(size_t(2), openvdb::math::MaxIndex(h));
    const openvdb::Vec3R i(0, 1, 1);
    CPPUNIT_ASSERT_EQUAL(size_t(0), openvdb::math::MinIndex(i));
    CPPUNIT_ASSERT_EQUAL(size_t(2), openvdb::math::MaxIndex(i));
    const openvdb::Vec3R j(1, 1, 1);
    CPPUNIT_ASSERT_EQUAL(size_t(2), openvdb::math::MinIndex(j));
    CPPUNIT_ASSERT_EQUAL(size_t(2), openvdb::math::MaxIndex(j));
}

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
