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

class TestVec3Metadata : public CppUnit::TestCase
{
public:
    CPPUNIT_TEST_SUITE(TestVec3Metadata);
    CPPUNIT_TEST(testVec3i);
    CPPUNIT_TEST(testVec3s);
    CPPUNIT_TEST(testVec3d);
    CPPUNIT_TEST_SUITE_END();

    void testVec3i();
    void testVec3s();
    void testVec3d();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestVec3Metadata);

void
TestVec3Metadata::testVec3i()
{
    using namespace openvdb;

    Metadata::Ptr m(new Vec3IMetadata(openvdb::Vec3i(1, 1, 1)));
    Metadata::Ptr m3 = m->copy();

    CPPUNIT_ASSERT(dynamic_cast<Vec3IMetadata*>( m.get()) != 0);
    CPPUNIT_ASSERT(dynamic_cast<Vec3IMetadata*>(m3.get()) != 0);

    CPPUNIT_ASSERT( m->typeName().compare("vec3i") == 0);
    CPPUNIT_ASSERT(m3->typeName().compare("vec3i") == 0);

    Vec3IMetadata *s = dynamic_cast<Vec3IMetadata*>(m.get());
    CPPUNIT_ASSERT(s->value() == openvdb::Vec3i(1, 1, 1));
    s->value() = openvdb::Vec3i(3, 3, 3);
    CPPUNIT_ASSERT(s->value() == openvdb::Vec3i(3, 3, 3));

    m3->copy(*s);

    s = dynamic_cast<Vec3IMetadata*>(m3.get());
    CPPUNIT_ASSERT(s->value() == openvdb::Vec3i(3, 3, 3));
}

void
TestVec3Metadata::testVec3s()
{
    using namespace openvdb;

    Metadata::Ptr m(new Vec3SMetadata(openvdb::Vec3s(1, 1, 1)));
    Metadata::Ptr m3 = m->copy();

    CPPUNIT_ASSERT(dynamic_cast<Vec3SMetadata*>( m.get()) != 0);
    CPPUNIT_ASSERT(dynamic_cast<Vec3SMetadata*>(m3.get()) != 0);

    CPPUNIT_ASSERT( m->typeName().compare("vec3s") == 0);
    CPPUNIT_ASSERT(m3->typeName().compare("vec3s") == 0);

    Vec3SMetadata *s = dynamic_cast<Vec3SMetadata*>(m.get());
    CPPUNIT_ASSERT(s->value() == openvdb::Vec3s(1, 1, 1));
    s->value() = openvdb::Vec3s(3, 3, 3);
    CPPUNIT_ASSERT(s->value() == openvdb::Vec3s(3, 3, 3));

    m3->copy(*s);

    s = dynamic_cast<Vec3SMetadata*>(m3.get());
    CPPUNIT_ASSERT(s->value() == openvdb::Vec3s(3, 3, 3));
}

void
TestVec3Metadata::testVec3d()
{
    using namespace openvdb;

    Metadata::Ptr m(new Vec3DMetadata(openvdb::Vec3d(1, 1, 1)));
    Metadata::Ptr m3 = m->copy();

    CPPUNIT_ASSERT(dynamic_cast<Vec3DMetadata*>( m.get()) != 0);
    CPPUNIT_ASSERT(dynamic_cast<Vec3DMetadata*>(m3.get()) != 0);

    CPPUNIT_ASSERT( m->typeName().compare("vec3d") == 0);
    CPPUNIT_ASSERT(m3->typeName().compare("vec3d") == 0);

    Vec3DMetadata *s = dynamic_cast<Vec3DMetadata*>(m.get());
    CPPUNIT_ASSERT(s->value() == openvdb::Vec3d(1, 1, 1));
    s->value() = openvdb::Vec3d(3, 3, 3);
    CPPUNIT_ASSERT(s->value() == openvdb::Vec3d(3, 3, 3));

    m3->copy(*s);

    s = dynamic_cast<Vec3DMetadata*>(m3.get());
    CPPUNIT_ASSERT(s->value() == openvdb::Vec3d(3, 3, 3));
}

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
