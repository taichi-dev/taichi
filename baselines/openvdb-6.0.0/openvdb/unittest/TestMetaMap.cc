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
#include <openvdb/util/logging.h>
#include <openvdb/Metadata.h>
#include <openvdb/MetaMap.h>

class TestMetaMap: public CppUnit::TestCase
{
public:
    CPPUNIT_TEST_SUITE(TestMetaMap);
    CPPUNIT_TEST(testInsert);
    CPPUNIT_TEST(testRemove);
    CPPUNIT_TEST(testGetMetadata);
    CPPUNIT_TEST(testIO);
    CPPUNIT_TEST(testEmptyIO);
    CPPUNIT_TEST(testCopyConstructor);
    CPPUNIT_TEST(testCopyConstructorEmpty);
    CPPUNIT_TEST(testAssignment);
    CPPUNIT_TEST(testEquality);
    CPPUNIT_TEST_SUITE_END();

    void testInsert();
    void testRemove();
    void testGetMetadata();
    void testIO();
    void testEmptyIO();
    void testCopyConstructor();
    void testCopyConstructorEmpty();
    void testAssignment();
    void testEquality();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestMetaMap);

void
TestMetaMap::testInsert()
{
    using namespace openvdb;

    MetaMap meta;
    meta.insertMeta("meta1", StringMetadata("testing"));
    meta.insertMeta("meta2", Int32Metadata(20));
    meta.insertMeta("meta3", FloatMetadata(2.0));

    MetaMap::MetaIterator iter = meta.beginMeta();
    int i = 1;
    for( ; iter != meta.endMeta(); ++iter, ++i) {
        if(i == 1) {
            CPPUNIT_ASSERT(iter->first.compare("meta1") == 0);
            std::string val = meta.metaValue<std::string>("meta1");
            CPPUNIT_ASSERT(val == "testing");
        } else if(i == 2) {
            CPPUNIT_ASSERT(iter->first.compare("meta2") == 0);
            int32_t val = meta.metaValue<int32_t>("meta2");
            CPPUNIT_ASSERT(val == 20);
        } else if(i == 3) {
            CPPUNIT_ASSERT(iter->first.compare("meta3") == 0);
            float val = meta.metaValue<float>("meta3");
            //CPPUNIT_ASSERT(val == 2.0);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0f,val,0);
        }
    }
}

void
TestMetaMap::testRemove()
{
    using namespace openvdb;

    MetaMap meta;
    meta.insertMeta("meta1", StringMetadata("testing"));
    meta.insertMeta("meta2", Int32Metadata(20));
    meta.insertMeta("meta3", FloatMetadata(2.0));

    meta.removeMeta("meta2");

    MetaMap::MetaIterator iter = meta.beginMeta();
    int i = 1;
    for( ; iter != meta.endMeta(); ++iter, ++i) {
        if(i == 1) {
            CPPUNIT_ASSERT(iter->first.compare("meta1") == 0);
            std::string val = meta.metaValue<std::string>("meta1");
            CPPUNIT_ASSERT(val == "testing");
        } else if(i == 2) {
            CPPUNIT_ASSERT(iter->first.compare("meta3") == 0);
            float val = meta.metaValue<float>("meta3");
            //CPPUNIT_ASSERT(val == 2.0);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0f,val,0);
        }
    }

    meta.removeMeta("meta1");

    iter = meta.beginMeta();
    for( ; iter != meta.endMeta(); ++iter, ++i) {
        CPPUNIT_ASSERT(iter->first.compare("meta3") == 0);
        float val = meta.metaValue<float>("meta3");
        //CPPUNIT_ASSERT(val == 2.0);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0f,val,0);
    }

    meta.removeMeta("meta3");

    CPPUNIT_ASSERT_EQUAL(0, int(meta.metaCount()));
}

void
TestMetaMap::testGetMetadata()
{
    using namespace openvdb;

    MetaMap meta;
    meta.insertMeta("meta1", StringMetadata("testing"));
    meta.insertMeta("meta2", Int32Metadata(20));
    meta.insertMeta("meta3", DoubleMetadata(2.0));

    Metadata::Ptr metadata = meta["meta2"];
    CPPUNIT_ASSERT(metadata);
    CPPUNIT_ASSERT(metadata->typeName().compare("int32") == 0);

    DoubleMetadata::Ptr dm = meta.getMetadata<DoubleMetadata>("meta3");
    //CPPUNIT_ASSERT(dm->value() == 2.0);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0,dm->value(),0);

    const DoubleMetadata::Ptr cdm = meta.getMetadata<DoubleMetadata>("meta3");
    //CPPUNIT_ASSERT(dm->value() == 2.0);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0,cdm->value(),0);

    CPPUNIT_ASSERT(!meta.getMetadata<StringMetadata>("meta2"));

    CPPUNIT_ASSERT_THROW(meta.metaValue<int32_t>("meta3"),
                         openvdb::TypeError);

    CPPUNIT_ASSERT_THROW(meta.metaValue<double>("meta5"),
                         openvdb::LookupError);
}

void
TestMetaMap::testIO()
{
    using namespace openvdb;

    logging::LevelScope suppressLogging{logging::Level::Fatal};

    Metadata::clearRegistry();

    // Write some metadata using unregistered types.
    MetaMap meta;
    meta.insertMeta("meta1", StringMetadata("testing"));
    meta.insertMeta("meta2", Int32Metadata(20));
    meta.insertMeta("meta3", DoubleMetadata(2.0));
    std::ostringstream ostr(std::ios_base::binary);
    meta.writeMeta(ostr);

    // Verify that reading metadata of unregistered types is possible,
    // though the values cannot be retrieved.
    MetaMap meta2;
    std::istringstream istr(ostr.str(), std::ios_base::binary);
    CPPUNIT_ASSERT_NO_THROW(meta2.readMeta(istr));
#if OPENVDB_ABI_VERSION_NUMBER < 5
    CPPUNIT_ASSERT_EQUAL(0, int(meta2.metaCount()));
#else
    CPPUNIT_ASSERT_EQUAL(3, int(meta2.metaCount()));

    // Verify that writing metadata of unknown type (i.e., UnknownMetadata) is possible.
    std::ostringstream ostrUnknown(std::ios_base::binary);
    meta2.writeMeta(ostrUnknown);
#endif

    // Register just one of the three types, then reread and verify that
    // the value of the registered type can be retrieved.
    Int32Metadata::registerType();
    istr.seekg(0, std::ios_base::beg);
    CPPUNIT_ASSERT_NO_THROW(meta2.readMeta(istr));
#if OPENVDB_ABI_VERSION_NUMBER >= 5
    CPPUNIT_ASSERT_EQUAL(3, int(meta2.metaCount()));
#else
    CPPUNIT_ASSERT_EQUAL(1, int(meta2.metaCount()));
#endif
    CPPUNIT_ASSERT_EQUAL(meta.metaValue<int>("meta2"), meta2.metaValue<int>("meta2"));

    // Register the remaining types.
    StringMetadata::registerType();
    DoubleMetadata::registerType();

    {
        // Now seek to beginning and read again.
        istr.seekg(0, std::ios_base::beg);
        meta2.clearMetadata();

        CPPUNIT_ASSERT_NO_THROW(meta2.readMeta(istr));
        CPPUNIT_ASSERT_EQUAL(meta.metaCount(), meta2.metaCount());

        std::string val = meta.metaValue<std::string>("meta1");
        std::string val2 = meta2.metaValue<std::string>("meta1");
        CPPUNIT_ASSERT_EQUAL(0, val.compare(val2));

        int intval = meta.metaValue<int>("meta2");
        int intval2 = meta2.metaValue<int>("meta2");
        CPPUNIT_ASSERT_EQUAL(intval, intval2);

        double dval = meta.metaValue<double>("meta3");
        double dval2 = meta2.metaValue<double>("meta3");
        CPPUNIT_ASSERT_DOUBLES_EQUAL(dval, dval2,0);
    }
    {
#if OPENVDB_ABI_VERSION_NUMBER >= 5
        // Verify that metadata that was written as UnknownMetadata can
        // be read as typed metadata once the underlying types are registered.
        std::istringstream istrUnknown(ostrUnknown.str(), std::ios_base::binary);

        meta2.clearMetadata();
        CPPUNIT_ASSERT_NO_THROW(meta2.readMeta(istrUnknown));

        CPPUNIT_ASSERT_EQUAL(meta.metaCount(), meta2.metaCount());
        CPPUNIT_ASSERT_EQUAL(
            meta.metaValue<std::string>("meta1"), meta2.metaValue<std::string>("meta1"));
        CPPUNIT_ASSERT_EQUAL(meta.metaValue<int>("meta2"), meta2.metaValue<int>("meta2"));
        CPPUNIT_ASSERT_DOUBLES_EQUAL(
            meta.metaValue<double>("meta3"), meta2.metaValue<double>("meta3"), 0.0);
#endif
    }

    // Clear the registry once the test is done.
    Metadata::clearRegistry();
}

void
TestMetaMap::testEmptyIO()
{
    using namespace openvdb;

    MetaMap meta;

    // Write out an empty metadata
    std::ostringstream ostr(std::ios_base::binary);

    // Read in the metadata;
    MetaMap meta2;
    std::istringstream istr(ostr.str(), std::ios_base::binary);
    CPPUNIT_ASSERT_NO_THROW(meta2.readMeta(istr));

    CPPUNIT_ASSERT(meta2.metaCount() == 0);
}

void
TestMetaMap::testCopyConstructor()
{
    using namespace openvdb;

    MetaMap meta;
    meta.insertMeta("meta1", StringMetadata("testing"));
    meta.insertMeta("meta2", Int32Metadata(20));
    meta.insertMeta("meta3", FloatMetadata(2.0));

    // copy constructor
    MetaMap meta2(meta);

    CPPUNIT_ASSERT(meta.metaCount() == meta2.metaCount());

    std::string str = meta.metaValue<std::string>("meta1");
    std::string str2 = meta2.metaValue<std::string>("meta1");
    CPPUNIT_ASSERT(str == str2);

    CPPUNIT_ASSERT(meta.metaValue<int32_t>("meta2") ==
            meta2.metaValue<int32_t>("meta2"));

    CPPUNIT_ASSERT_DOUBLES_EQUAL(meta.metaValue<float>("meta3"),
                                 meta2.metaValue<float>("meta3"),0);
    //CPPUNIT_ASSERT(meta.metaValue<float>("meta3") ==
    //        meta2.metaValue<float>("meta3"));
}


void
TestMetaMap::testCopyConstructorEmpty()
{
    using namespace openvdb;

    MetaMap meta;

    MetaMap meta2(meta);

    CPPUNIT_ASSERT(meta.metaCount() == 0);
    CPPUNIT_ASSERT(meta2.metaCount() == meta.metaCount());
}


void
TestMetaMap::testAssignment()
{
    using namespace openvdb;

    // Populate a map with data.
    MetaMap meta;
    meta.insertMeta("meta1", StringMetadata("testing"));
    meta.insertMeta("meta2", Int32Metadata(20));
    meta.insertMeta("meta3", FloatMetadata(2.0));

    // Create an empty map.
    MetaMap meta2;
    CPPUNIT_ASSERT_EQUAL(0, int(meta2.metaCount()));

    // Copy the first map to the second.
    meta2 = meta;
    CPPUNIT_ASSERT_EQUAL(meta.metaCount(), meta2.metaCount());

    // Verify that the contents of the two maps are the same.
    CPPUNIT_ASSERT_EQUAL(
        meta.metaValue<std::string>("meta1"), meta2.metaValue<std::string>("meta1"));
    CPPUNIT_ASSERT_EQUAL(meta.metaValue<int32_t>("meta2"), meta2.metaValue<int32_t>("meta2"));
    CPPUNIT_ASSERT_DOUBLES_EQUAL(
        meta.metaValue<float>("meta3"), meta2.metaValue<float>("meta3"), /*tolerance=*/0);

    // Verify that changing one map doesn't affect the other.
    meta.insertMeta("meta1", StringMetadata("changed"));
    std::string str = meta.metaValue<std::string>("meta1");
    CPPUNIT_ASSERT_EQUAL(std::string("testing"), meta2.metaValue<std::string>("meta1"));
}


void
TestMetaMap::testEquality()
{
    using namespace openvdb;

    // Populate a map with data.
    MetaMap meta;
    meta.insertMeta("meta1", StringMetadata("testing"));
    meta.insertMeta("meta2", Int32Metadata(20));
    meta.insertMeta("meta3", FloatMetadata(3.14159f));

    // Create an empty map.
    MetaMap meta2;

    // Verify that the two maps differ.
    CPPUNIT_ASSERT(meta != meta2);
    CPPUNIT_ASSERT(meta2 != meta);

    // Copy the first map to the second.
    meta2 = meta;

    // Verify that the two maps are equivalent.
    CPPUNIT_ASSERT(meta == meta2);
    CPPUNIT_ASSERT(meta2 == meta);

    // Modify the first map.
    meta.removeMeta("meta1");
    meta.insertMeta("abc", DoubleMetadata(2.0));

    // Verify that the two maps differ.
    CPPUNIT_ASSERT(meta != meta2);
    CPPUNIT_ASSERT(meta2 != meta);

    // Modify the second map and verify that the two maps differ.
    meta2 = meta;
    meta2.insertMeta("meta2", Int32Metadata(42));
    CPPUNIT_ASSERT(meta != meta2);
    CPPUNIT_ASSERT(meta2 != meta);

    meta2 = meta;
    meta2.insertMeta("meta3", FloatMetadata(2.0001f));
    CPPUNIT_ASSERT(meta != meta2);
    CPPUNIT_ASSERT(meta2 != meta);

    meta2 = meta;
    meta2.insertMeta("abc", DoubleMetadata(2.0001));
    CPPUNIT_ASSERT(meta != meta2);
    CPPUNIT_ASSERT(meta2 != meta);
}

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
