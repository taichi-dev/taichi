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
#include <openvdb/Metadata.h>

class TestStringMetadata : public CppUnit::TestCase
{
public:
    CPPUNIT_TEST_SUITE(TestStringMetadata);
    CPPUNIT_TEST(test);
    CPPUNIT_TEST_SUITE_END();

    void test();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestStringMetadata);

void
TestStringMetadata::test()
{
    using namespace openvdb;

    Metadata::Ptr m(new StringMetadata("testing"));
    Metadata::Ptr m2 = m->copy();

    CPPUNIT_ASSERT(dynamic_cast<StringMetadata*>(m.get()) != 0);
    CPPUNIT_ASSERT(dynamic_cast<StringMetadata*>(m2.get()) != 0);

    CPPUNIT_ASSERT(m->typeName().compare("string") == 0);
    CPPUNIT_ASSERT(m2->typeName().compare("string") == 0);

    StringMetadata *s = dynamic_cast<StringMetadata*>(m.get());
    CPPUNIT_ASSERT(s->value().compare("testing") == 0);
    s->value() = "testing2";
    CPPUNIT_ASSERT(s->value().compare("testing2") == 0);

    m2->copy(*s);

    s = dynamic_cast<StringMetadata*>(m2.get());
    CPPUNIT_ASSERT(s->value().compare("testing2") == 0);
}

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
