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
#include <openvdb/Types.h>
#include <cctype> // for toupper()
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
class TestLeafIO: public CppUnit::TestCase
{
public:
    static std::string testSuiteName()
    {
        std::string name = openvdb::typeNameAsString<T>();
        if (!name.empty()) name[0] = static_cast<char>(::toupper(name[0]));
        return "TestLeafIO" + name;
    }

    CPPUNIT_TEST_SUITE(TestLeafIO);
    CPPUNIT_TEST(testBuffer);
    CPPUNIT_TEST_SUITE_END();

    void testBuffer();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestLeafIO<int>);
CPPUNIT_TEST_SUITE_REGISTRATION(TestLeafIO<float>);
CPPUNIT_TEST_SUITE_REGISTRATION(TestLeafIO<double>);
CPPUNIT_TEST_SUITE_REGISTRATION(TestLeafIO<bool>);
CPPUNIT_TEST_SUITE_REGISTRATION(TestLeafIO<openvdb::Byte>);
CPPUNIT_TEST_SUITE_REGISTRATION(TestLeafIO<openvdb::Vec3R>);
CPPUNIT_TEST_SUITE_REGISTRATION(TestLeafIO<std::string>);


template<typename T>
void
TestLeafIO<T>::testBuffer()
{
    openvdb::tree::LeafNode<T, 3> leaf(openvdb::Coord(0, 0, 0));

    leaf.setValueOn(openvdb::Coord(0, 1, 0), T(1));
    leaf.setValueOn(openvdb::Coord(1, 0, 0), T(1));

    std::ostringstream ostr(std::ios_base::binary);

    leaf.writeBuffers(ostr);

    leaf.setValueOn(openvdb::Coord(0, 1, 0), T(0));
    leaf.setValueOn(openvdb::Coord(0, 1, 1), T(1));

    std::istringstream istr(ostr.str(), std::ios_base::binary);

    // Since the input stream doesn't include a VDB header with file format version info,
    // tag the input stream explicitly with the current version number.
    openvdb::io::setCurrentVersion(istr);

    leaf.readBuffers(istr);

    CPPUNIT_ASSERT_DOUBLES_EQUAL(T(1), leaf.getValue(openvdb::Coord(0, 1, 0)), /*tolerance=*/0);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(T(1), leaf.getValue(openvdb::Coord(1, 0, 0)), /*tolerance=*/0);

    CPPUNIT_ASSERT(leaf.onVoxelCount() == 2);
}


template<>
void
TestLeafIO<std::string>::testBuffer()
{
    openvdb::tree::LeafNode<std::string, 3>
        leaf(openvdb::Coord(0, 0, 0), std::string());

    leaf.setValueOn(openvdb::Coord(0, 1, 0), std::string("test"));
    leaf.setValueOn(openvdb::Coord(1, 0, 0), std::string("test"));

    std::ostringstream ostr(std::ios_base::binary);

    leaf.writeBuffers(ostr);

    leaf.setValueOn(openvdb::Coord(0, 1, 0), std::string("douche"));
    leaf.setValueOn(openvdb::Coord(0, 1, 1), std::string("douche"));

    std::istringstream istr(ostr.str(), std::ios_base::binary);

    // Since the input stream doesn't include a VDB header with file format version info,
    // tag the input stream explicitly with the current version number.
    openvdb::io::setCurrentVersion(istr);

    leaf.readBuffers(istr);

    CPPUNIT_ASSERT_EQUAL(std::string("test"), leaf.getValue(openvdb::Coord(0, 1, 0)));
    CPPUNIT_ASSERT_EQUAL(std::string("test"), leaf.getValue(openvdb::Coord(1, 0, 0)));

    CPPUNIT_ASSERT(leaf.onVoxelCount() == 2);
}


template<>
void
TestLeafIO<openvdb::Vec3R>::testBuffer()
{
    openvdb::tree::LeafNode<openvdb::Vec3R, 3> leaf(openvdb::Coord(0, 0, 0));

    leaf.setValueOn(openvdb::Coord(0, 1, 0), openvdb::Vec3R(1, 1, 1));
    leaf.setValueOn(openvdb::Coord(1, 0, 0), openvdb::Vec3R(1, 1, 1));

    std::ostringstream ostr(std::ios_base::binary);

    leaf.writeBuffers(ostr);

    leaf.setValueOn(openvdb::Coord(0, 1, 0), openvdb::Vec3R(0, 0, 0));
    leaf.setValueOn(openvdb::Coord(0, 1, 1), openvdb::Vec3R(1, 1, 1));

    std::istringstream istr(ostr.str(), std::ios_base::binary);

    // Since the input stream doesn't include a VDB header with file format version info,
    // tag the input stream explicitly with the current version number.
    openvdb::io::setCurrentVersion(istr);

    leaf.readBuffers(istr);

    CPPUNIT_ASSERT(leaf.getValue(openvdb::Coord(0, 1, 0)) == openvdb::Vec3R(1, 1, 1));
    CPPUNIT_ASSERT(leaf.getValue(openvdb::Coord(1, 0, 0)) == openvdb::Vec3R(1, 1, 1));

    CPPUNIT_ASSERT(leaf.onVoxelCount() == 2);
}

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
