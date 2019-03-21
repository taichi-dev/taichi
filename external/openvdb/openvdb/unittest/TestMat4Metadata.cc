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

class TestMat4Metadata : public CppUnit::TestCase
{
public:
    CPPUNIT_TEST_SUITE(TestMat4Metadata);
    CPPUNIT_TEST(testMat4s);
    CPPUNIT_TEST(testMat4d);
    CPPUNIT_TEST_SUITE_END();

    void testMat4s();
    void testMat4d();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestMat4Metadata);

void
TestMat4Metadata::testMat4s()
{
    using namespace openvdb;

    Metadata::Ptr m(new Mat4SMetadata(openvdb::math::Mat4s(1.0f, 1.0f, 1.0f, 1.0f,
                                                           1.0f, 1.0f, 1.0f, 1.0f,
                                                           1.0f, 1.0f, 1.0f, 1.0f,
                                                           1.0f, 1.0f, 1.0f, 1.0f)));
    Metadata::Ptr m3 = m->copy();

    CPPUNIT_ASSERT(dynamic_cast<Mat4SMetadata*>( m.get()) != 0);
    CPPUNIT_ASSERT(dynamic_cast<Mat4SMetadata*>(m3.get()) != 0);

    CPPUNIT_ASSERT( m->typeName().compare("mat4s") == 0);
    CPPUNIT_ASSERT(m3->typeName().compare("mat4s") == 0);

    Mat4SMetadata *s = dynamic_cast<Mat4SMetadata*>(m.get());
    CPPUNIT_ASSERT(s->value() == openvdb::math::Mat4s(1.0f, 1.0f, 1.0f, 1.0f,
                                                      1.0f, 1.0f, 1.0f, 1.0f,
                                                      1.0f, 1.0f, 1.0f, 1.0f,
                                                      1.0f, 1.0f, 1.0f, 1.0f));
    s->value() = openvdb::math::Mat4s(3.0f, 3.0f, 3.0f, 3.0f,
                                      3.0f, 3.0f, 3.0f, 3.0f,
                                      3.0f, 3.0f, 3.0f, 3.0f,
                                      3.0f, 3.0f, 3.0f, 3.0f);
    CPPUNIT_ASSERT(s->value() == openvdb::math::Mat4s(3.0f, 3.0f, 3.0f, 3.0f,
                                                      3.0f, 3.0f, 3.0f, 3.0f,
                                                      3.0f, 3.0f, 3.0f, 3.0f,
                                                      3.0f, 3.0f, 3.0f, 3.0f));

    m3->copy(*s);

    s = dynamic_cast<Mat4SMetadata*>(m3.get());
    CPPUNIT_ASSERT(s->value() == openvdb::math::Mat4s(3.0f, 3.0f, 3.0f, 3.0f,
                                                      3.0f, 3.0f, 3.0f, 3.0f,
                                                      3.0f, 3.0f, 3.0f, 3.0f,
                                                      3.0f, 3.0f, 3.0f, 3.0f));
}

void
TestMat4Metadata::testMat4d()
{
    using namespace openvdb;

    Metadata::Ptr m(new Mat4DMetadata(openvdb::math::Mat4d(1.0, 1.0, 1.0, 1.0,
                                                           1.0, 1.0, 1.0, 1.0,
                                                           1.0, 1.0, 1.0, 1.0,
                                                           1.0, 1.0, 1.0, 1.0)));
    Metadata::Ptr m3 = m->copy();

    CPPUNIT_ASSERT(dynamic_cast<Mat4DMetadata*>( m.get()) != 0);
    CPPUNIT_ASSERT(dynamic_cast<Mat4DMetadata*>(m3.get()) != 0);

    CPPUNIT_ASSERT( m->typeName().compare("mat4d") == 0);
    CPPUNIT_ASSERT(m3->typeName().compare("mat4d") == 0);

    Mat4DMetadata *s = dynamic_cast<Mat4DMetadata*>(m.get());
    CPPUNIT_ASSERT(s->value() == openvdb::math::Mat4d(1.0, 1.0, 1.0, 1.0,
                                                      1.0, 1.0, 1.0, 1.0,
                                                      1.0, 1.0, 1.0, 1.0,
                                                      1.0, 1.0, 1.0, 1.0));
    s->value() = openvdb::math::Mat4d(3.0, 3.0, 3.0, 3.0,
                                      3.0, 3.0, 3.0, 3.0,
                                      3.0, 3.0, 3.0, 3.0,
                                      3.0, 3.0, 3.0, 3.0);
    CPPUNIT_ASSERT(s->value() == openvdb::math::Mat4d(3.0, 3.0, 3.0, 3.0,
                                                      3.0, 3.0, 3.0, 3.0,
                                                      3.0, 3.0, 3.0, 3.0,
                                                      3.0, 3.0, 3.0, 3.0));

    m3->copy(*s);

    s = dynamic_cast<Mat4DMetadata*>(m3.get());
    CPPUNIT_ASSERT(s->value() == openvdb::math::Mat4d(3.0, 3.0, 3.0, 3.0,
                                                      3.0, 3.0, 3.0, 3.0,
                                                      3.0, 3.0, 3.0, 3.0,
                                                      3.0, 3.0, 3.0, 3.0));
}

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
