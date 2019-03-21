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
#include <openvdb/util/Name.h>

class TestName : public CppUnit::TestCase
{
public:
    CPPUNIT_TEST_SUITE(TestName);
    CPPUNIT_TEST(test);
    CPPUNIT_TEST(testIO);
    CPPUNIT_TEST(testMultipleIO);
    CPPUNIT_TEST_SUITE_END();

    void test();
    void testIO();
    void testMultipleIO();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestName);

void
TestName::test()
{
    using namespace openvdb;

    Name name;
    Name name2("something");
    Name name3 = std::string("something2");
    name = "something";

    CPPUNIT_ASSERT(name == name2);
    CPPUNIT_ASSERT(name != name3);
    CPPUNIT_ASSERT(name != Name("testing"));
    CPPUNIT_ASSERT(name == Name("something"));
}

void
TestName::testIO()
{
    using namespace openvdb;

    Name name("some name that i made up");

    std::ostringstream ostr(std::ios_base::binary);

    openvdb::writeString(ostr, name);

    name = "some other name";

    CPPUNIT_ASSERT(name == Name("some other name"));

    std::istringstream istr(ostr.str(), std::ios_base::binary);

    name = openvdb::readString(istr);

    CPPUNIT_ASSERT(name == Name("some name that i made up"));
}

void
TestName::testMultipleIO()
{
    using namespace openvdb;

    Name name("some name that i made up");
    Name name2("something else");

    std::ostringstream ostr(std::ios_base::binary);

    openvdb::writeString(ostr, name);
    openvdb::writeString(ostr, name2);

    std::istringstream istr(ostr.str(), std::ios_base::binary);

    Name n = openvdb::readString(istr), n2 = openvdb::readString(istr);

    CPPUNIT_ASSERT(name == n);
    CPPUNIT_ASSERT(name2 == n2);
}

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
