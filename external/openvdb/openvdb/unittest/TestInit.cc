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
#include <openvdb/openvdb.h>


class TestInit: public CppUnit::TestCase
{
public:
    CPPUNIT_TEST_SUITE(TestInit);
    CPPUNIT_TEST(test);
    CPPUNIT_TEST_SUITE_END();

    void test();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestInit);


void
TestInit::test()
{
    using namespace openvdb;

    initialize();

    // data types
    CPPUNIT_ASSERT(DoubleMetadata::isRegisteredType());
    CPPUNIT_ASSERT(FloatMetadata::isRegisteredType());
    CPPUNIT_ASSERT(Int32Metadata::isRegisteredType());
    CPPUNIT_ASSERT(Int64Metadata::isRegisteredType());
    CPPUNIT_ASSERT(StringMetadata::isRegisteredType());
    CPPUNIT_ASSERT(Vec2IMetadata::isRegisteredType());
    CPPUNIT_ASSERT(Vec2SMetadata::isRegisteredType());
    CPPUNIT_ASSERT(Vec2DMetadata::isRegisteredType());
    CPPUNIT_ASSERT(Vec3IMetadata::isRegisteredType());
    CPPUNIT_ASSERT(Vec3SMetadata::isRegisteredType());
    CPPUNIT_ASSERT(Vec3DMetadata::isRegisteredType());

    // map types
    CPPUNIT_ASSERT(math::AffineMap::isRegistered());
    CPPUNIT_ASSERT(math::UnitaryMap::isRegistered());
    CPPUNIT_ASSERT(math::ScaleMap::isRegistered());
    CPPUNIT_ASSERT(math::TranslationMap::isRegistered());
    CPPUNIT_ASSERT(math::ScaleTranslateMap::isRegistered());
    CPPUNIT_ASSERT(math::NonlinearFrustumMap::isRegistered());
    
    // grid types
    CPPUNIT_ASSERT(BoolGrid::isRegistered());
    CPPUNIT_ASSERT(FloatGrid::isRegistered());
    CPPUNIT_ASSERT(DoubleGrid::isRegistered());
    CPPUNIT_ASSERT(Int32Grid::isRegistered());
    CPPUNIT_ASSERT(Int64Grid::isRegistered());
    CPPUNIT_ASSERT(StringGrid::isRegistered());
    CPPUNIT_ASSERT(Vec3IGrid::isRegistered());
    CPPUNIT_ASSERT(Vec3SGrid::isRegistered());
    CPPUNIT_ASSERT(Vec3DGrid::isRegistered());

    uninitialize();

    CPPUNIT_ASSERT(!DoubleMetadata::isRegisteredType());
    CPPUNIT_ASSERT(!FloatMetadata::isRegisteredType());
    CPPUNIT_ASSERT(!Int32Metadata::isRegisteredType());
    CPPUNIT_ASSERT(!Int64Metadata::isRegisteredType());
    CPPUNIT_ASSERT(!StringMetadata::isRegisteredType());
    CPPUNIT_ASSERT(!Vec2IMetadata::isRegisteredType());
    CPPUNIT_ASSERT(!Vec2SMetadata::isRegisteredType());
    CPPUNIT_ASSERT(!Vec2DMetadata::isRegisteredType());
    CPPUNIT_ASSERT(!Vec3IMetadata::isRegisteredType());
    CPPUNIT_ASSERT(!Vec3SMetadata::isRegisteredType());
    CPPUNIT_ASSERT(!Vec3DMetadata::isRegisteredType());

    CPPUNIT_ASSERT(!math::AffineMap::isRegistered());
    CPPUNIT_ASSERT(!math::UnitaryMap::isRegistered());
    CPPUNIT_ASSERT(!math::ScaleMap::isRegistered());
    CPPUNIT_ASSERT(!math::TranslationMap::isRegistered());
    CPPUNIT_ASSERT(!math::ScaleTranslateMap::isRegistered());
    CPPUNIT_ASSERT(!math::NonlinearFrustumMap::isRegistered());

    CPPUNIT_ASSERT(!BoolGrid::isRegistered());
    CPPUNIT_ASSERT(!FloatGrid::isRegistered());
    CPPUNIT_ASSERT(!DoubleGrid::isRegistered());
    CPPUNIT_ASSERT(!Int32Grid::isRegistered());
    CPPUNIT_ASSERT(!Int64Grid::isRegistered());
    CPPUNIT_ASSERT(!StringGrid::isRegistered());
    CPPUNIT_ASSERT(!Vec3IGrid::isRegistered());
    CPPUNIT_ASSERT(!Vec3SGrid::isRegistered());
    CPPUNIT_ASSERT(!Vec3DGrid::isRegistered());
}

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
