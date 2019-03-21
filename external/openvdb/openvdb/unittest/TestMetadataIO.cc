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
#include <openvdb/Metadata.h>
#include <openvdb/Types.h>
#include <iostream>
#include <sstream>

// CPPUNIT_TEST_SUITE() invokes CPPUNIT_TESTNAMER_DECL() to generate a suite name
// from the FixtureType.  But if FixtureType is a templated type, the generated name
// can become long and messy.  This macro overrides the normal naming logic,
// instead invoking FixtureType::testSuiteName(), which should be a static member
// function that returns a std::string containing the suite name for the specific
// template instantiation.
#undef CPPUNIT_TESTNAMER_DECL
#define CPPUNIT_TESTNAMER_DECL( variableName, FixtureType ) \
    CPPUNIT_NS::TestNamer variableName( FixtureType::testSuiteName() )

template<typename T>
class TestMetadataIO: public CppUnit::TestCase
{
public:
    static std::string testSuiteName()
    {
        std::string name = openvdb::typeNameAsString<T>();
        if (!name.empty()) name[0] = static_cast<char>(::toupper(name[0]));
        return "TestMetadataIO" + name;
    }

    CPPUNIT_TEST_SUITE(TestMetadataIO);
    CPPUNIT_TEST(test);
    CPPUNIT_TEST(testMultiple);
    CPPUNIT_TEST_SUITE_END();

    void test();
    void testMultiple();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestMetadataIO<int>);
CPPUNIT_TEST_SUITE_REGISTRATION(TestMetadataIO<int64_t>);
CPPUNIT_TEST_SUITE_REGISTRATION(TestMetadataIO<float>);
CPPUNIT_TEST_SUITE_REGISTRATION(TestMetadataIO<double>);
CPPUNIT_TEST_SUITE_REGISTRATION(TestMetadataIO<std::string>);
CPPUNIT_TEST_SUITE_REGISTRATION(TestMetadataIO<openvdb::Vec3R>);
CPPUNIT_TEST_SUITE_REGISTRATION(TestMetadataIO<openvdb::Vec2i>);


template<typename T>
void
TestMetadataIO<T>::test()
{
    using namespace openvdb;

    TypedMetadata<T> m(1);

    std::ostringstream ostr(std::ios_base::binary);

    m.write(ostr);

    std::istringstream istr(ostr.str(), std::ios_base::binary);

    TypedMetadata<T> tm;
    tm.read(istr);

    CPPUNIT_ASSERT_DOUBLES_EQUAL(T(1),tm.value(),0);
    //CPPUNIT_ASSERT(tm.value() == T(1));
}


template<typename T>
void
TestMetadataIO<T>::testMultiple()
{
    using namespace openvdb;

    TypedMetadata<T> m(1);
    TypedMetadata<T> m2(2);

    std::ostringstream ostr(std::ios_base::binary);

    m.write(ostr);
    m2.write(ostr);

    std::istringstream istr(ostr.str(), std::ios_base::binary);

    TypedMetadata<T> tm, tm2;
    tm.read(istr);
    tm2.read(istr);

    CPPUNIT_ASSERT_DOUBLES_EQUAL(T(1),tm.value(),0);
    //CPPUNIT_ASSERT(tm.value() == T(1));
    CPPUNIT_ASSERT_DOUBLES_EQUAL(T(2),tm2.value(),0);
    //CPPUNIT_ASSERT(tm2.value() == T(2));
}


template<>
void
TestMetadataIO<std::string>::test()
{
    using namespace openvdb;

    TypedMetadata<std::string> m("test");

    std::ostringstream ostr(std::ios_base::binary);

    m.write(ostr);

    std::istringstream istr(ostr.str(), std::ios_base::binary);

    TypedMetadata<std::string> tm;
    tm.read(istr);

    CPPUNIT_ASSERT(tm.value() == "test");
}


template<>
void
TestMetadataIO<std::string>::testMultiple()
{
    using namespace openvdb;

    TypedMetadata<std::string> m("test");
    TypedMetadata<std::string> m2("test2");

    std::ostringstream ostr(std::ios_base::binary);

    m.write(ostr);
    m2.write(ostr);

    std::istringstream istr(ostr.str(), std::ios_base::binary);

    TypedMetadata<std::string> tm, tm2;
    tm.read(istr);
    tm2.read(istr);

    CPPUNIT_ASSERT(tm.value() == "test");
    CPPUNIT_ASSERT(tm2.value() == "test2");
}


template<>
void
TestMetadataIO<openvdb::Vec3R>::test()
{
    using namespace openvdb;

    TypedMetadata<Vec3R> m(Vec3R(1, 2, 3));

    std::ostringstream ostr(std::ios_base::binary);

    m.write(ostr);

    std::istringstream istr(ostr.str(), std::ios_base::binary);

    TypedMetadata<Vec3R> tm;
    tm.read(istr);

    CPPUNIT_ASSERT(tm.value() == Vec3R(1, 2, 3));
}


template<>
void
TestMetadataIO<openvdb::Vec3R>::testMultiple()
{
    using namespace openvdb;

    TypedMetadata<Vec3R> m(Vec3R(1, 2, 3));
    TypedMetadata<Vec3R> m2(Vec3R(4, 5, 6));

    std::ostringstream ostr(std::ios_base::binary);

    m.write(ostr);
    m2.write(ostr);

    std::istringstream istr(ostr.str(), std::ios_base::binary);

    TypedMetadata<Vec3R> tm, tm2;
    tm.read(istr);
    tm2.read(istr);

    CPPUNIT_ASSERT(tm.value() == Vec3R(1, 2, 3));
    CPPUNIT_ASSERT(tm2.value() == Vec3R(4, 5, 6));
}


template<>
void
TestMetadataIO<openvdb::Vec2i>::test()
{
    using namespace openvdb;

    TypedMetadata<Vec2i> m(Vec2i(1, 2));

    std::ostringstream ostr(std::ios_base::binary);

    m.write(ostr);

    std::istringstream istr(ostr.str(), std::ios_base::binary);

    TypedMetadata<Vec2i> tm;
    tm.read(istr);

    CPPUNIT_ASSERT(tm.value() == Vec2i(1, 2));
}


template<>
void
TestMetadataIO<openvdb::Vec2i>::testMultiple()
{
    using namespace openvdb;

    TypedMetadata<Vec2i> m(Vec2i(1, 2));
    TypedMetadata<Vec2i> m2(Vec2i(3, 4));

    std::ostringstream ostr(std::ios_base::binary);

    m.write(ostr);
    m2.write(ostr);

    std::istringstream istr(ostr.str(), std::ios_base::binary);

    TypedMetadata<Vec2i> tm, tm2;
    tm.read(istr);
    tm2.read(istr);

    CPPUNIT_ASSERT(tm.value() == Vec2i(1, 2));
    CPPUNIT_ASSERT(tm2.value() == Vec2i(3, 4));
}

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
