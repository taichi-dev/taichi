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


class TestMetadata: public CppUnit::TestCase
{
public:
    void setUp() override { openvdb::Metadata::clearRegistry(); }
    void tearDown() override { openvdb::Metadata::clearRegistry(); }

    CPPUNIT_TEST_SUITE(TestMetadata);
    CPPUNIT_TEST(testMetadataRegistry);
    CPPUNIT_TEST(testMetadataAsBool);
    CPPUNIT_TEST(testCustomMetadata);
    CPPUNIT_TEST_SUITE_END();

    void testMetadataRegistry();
    void testMetadataAsBool();
    void testCustomMetadata();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestMetadata);


void
TestMetadata::testMetadataRegistry()
{
    using namespace openvdb;

    Int32Metadata::registerType();

    StringMetadata strMetadata;

    CPPUNIT_ASSERT(!Metadata::isRegisteredType(strMetadata.typeName()));

    StringMetadata::registerType();

    CPPUNIT_ASSERT(Metadata::isRegisteredType(strMetadata.typeName()));
    CPPUNIT_ASSERT(Metadata::isRegisteredType(Int32Metadata::staticTypeName()));

    Metadata::Ptr stringMetadata = Metadata::createMetadata(strMetadata.typeName());

    CPPUNIT_ASSERT(stringMetadata->typeName() == strMetadata.typeName());

    StringMetadata::unregisterType();

    CPPUNIT_ASSERT_THROW(Metadata::createMetadata(strMetadata.typeName()), openvdb::LookupError);
}

void
TestMetadata::testMetadataAsBool()
{
    using namespace openvdb;

    {
        FloatMetadata meta(0.0);
        CPPUNIT_ASSERT(!meta.asBool());
        meta.setValue(1.0);
        CPPUNIT_ASSERT(meta.asBool());
        meta.setValue(-1.0);
        CPPUNIT_ASSERT(meta.asBool());
        meta.setValue(999.0);
        CPPUNIT_ASSERT(meta.asBool());
    }
    {
        Int32Metadata meta(0);
        CPPUNIT_ASSERT(!meta.asBool());
        meta.setValue(1);
        CPPUNIT_ASSERT(meta.asBool());
        meta.setValue(-1);
        CPPUNIT_ASSERT(meta.asBool());
        meta.setValue(999);
        CPPUNIT_ASSERT(meta.asBool());
    }
    {
        StringMetadata meta("");
        CPPUNIT_ASSERT(!meta.asBool());
        meta.setValue("abc");
        CPPUNIT_ASSERT(meta.asBool());
    }
    {
        Vec3IMetadata meta(Vec3i(0));
        CPPUNIT_ASSERT(!meta.asBool());
        meta.setValue(Vec3i(-1, 0, 1));
        CPPUNIT_ASSERT(meta.asBool());
    }
    {
        Vec3SMetadata meta(Vec3s(0.0));
        CPPUNIT_ASSERT(!meta.asBool());
        meta.setValue(Vec3s(-1.0, 0.0, 1.0));
        CPPUNIT_ASSERT(meta.asBool());
    }
}


void
TestMetadata::testCustomMetadata()
{
    using namespace openvdb;

    const Vec3i expected(1, 2, 3);
    std::ostringstream ostr(std::ios_base::binary);
    {
        Vec3IMetadata::registerType();
        Vec3IMetadata meta(expected);

        // Write Vec3I metadata to a byte string.
        meta.write(ostr);
    }

    // Unregister Vec3I metadata.
    Metadata::clearRegistry();

#if OPENVDB_ABI_VERSION_NUMBER >= 5
    {
        std::istringstream istr(ostr.str(), std::ios_base::binary);

        UnknownMetadata meta;
        // Verify that metadata of an unregistered type can be read successfully.
        CPPUNIT_ASSERT_NO_THROW(meta.read(istr));

        // Verify that the metadata matches the original vector value.
        CPPUNIT_ASSERT_EQUAL(sizeof(Vec3i), size_t(meta.size()));
        CPPUNIT_ASSERT(meta.value().size() == size_t(meta.size()));
        CPPUNIT_ASSERT_EQUAL(expected, *reinterpret_cast<const Vec3i*>(&meta.value()[0]));

        ostr.str("");
        meta.write(ostr);

        // Verify that UnknownMetadata can be copied.
        auto metaPtr = meta.copy();
        CPPUNIT_ASSERT(metaPtr.get() != nullptr);
        CPPUNIT_ASSERT(meta == *metaPtr);

        // Verify that typed metadata can be copied into UnknownMetadata.
        meta.copy(Vec3IMetadata(expected));
        CPPUNIT_ASSERT_EQUAL(sizeof(expected), size_t(meta.size()));
        const auto* ptr = reinterpret_cast<const uint8_t*>(&expected);
        CPPUNIT_ASSERT(UnknownMetadata::ByteVec(ptr, ptr + sizeof(expected)) == meta.value());
    }
#endif

    Vec3IMetadata::registerType();

    {
        std::istringstream istr(ostr.str(), std::ios_base::binary);
        Vec3IMetadata meta;
        meta.read(istr);

        CPPUNIT_ASSERT_EQUAL(expected, meta.value());
    }
}

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
