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
#include <openvdb/points/AttributeGroup.h>
#include <openvdb/points/AttributeSet.h>
#include <openvdb/openvdb.h>
#include <openvdb/Types.h>
#include <openvdb/Metadata.h>

#include <iostream>
#include <sstream>

class TestAttributeSet: public CppUnit::TestCase
{
public:
    void setUp() override { openvdb::initialize(); }
    void tearDown() override { openvdb::uninitialize(); }

    CPPUNIT_TEST_SUITE(TestAttributeSet);
    CPPUNIT_TEST(testAttributeSetDescriptor);
    CPPUNIT_TEST(testAttributeSet);
    CPPUNIT_TEST(testAttributeSetGroups);

    CPPUNIT_TEST_SUITE_END();

    void testAttributeSetDescriptor();
    void testAttributeSet();
    void testAttributeSetGroups();
}; // class TestAttributeSet

CPPUNIT_TEST_SUITE_REGISTRATION(TestAttributeSet);


////////////////////////////////////////


using namespace openvdb;
using namespace openvdb::points;

namespace {

bool
matchingAttributeSets(const AttributeSet& lhs,
    const AttributeSet& rhs)
{
    if (lhs.size() != rhs.size()) return false;
    if (lhs.memUsage() != rhs.memUsage()) return false;
    if (lhs.descriptor() != rhs.descriptor()) return false;

    for (size_t n = 0, N = lhs.size(); n < N; ++n) {

        const AttributeArray* a = lhs.getConst(n);
        const AttributeArray* b = rhs.getConst(n);

        if (a->size() != b->size()) return false;
        if (a->isUniform() != b->isUniform()) return false;
// disable deprecated warnings for in-memory compression
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
        if (a->isCompressed() != b->isCompressed()) return false;
#pragma GCC diagnostic pop
        if (a->isHidden() != b->isHidden()) return false;
        if (a->type() != b->type()) return false;
    }

    return true;
}

bool
attributeSetMatchesDescriptor(  const AttributeSet& attrSet,
                                const AttributeSet::Descriptor& descriptor)
{
    if (descriptor.size() != attrSet.size())    return false;

    // check default metadata

    const openvdb::MetaMap& meta1 = descriptor.getMetadata();
    const openvdb::MetaMap& meta2 = attrSet.descriptor().getMetadata();

    // build vector of all default keys

    std::vector<openvdb::Name> defaultKeys;

    for (auto it = meta1.beginMeta(), itEnd = meta1.endMeta(); it != itEnd; ++it)
    {
        const openvdb::Name& name = it->first;

        if (name.compare(0, 8, "default:") == 0) {
            defaultKeys.push_back(name);
        }
    }

    for (auto it = meta2.beginMeta(), itEnd = meta2.endMeta(); it != itEnd; ++it)
    {
        const openvdb::Name& name = it->first;

        if (name.compare(0, 8, "default:") == 0) {
            if (std::find(defaultKeys.begin(), defaultKeys.end(), name) != defaultKeys.end()) {
                defaultKeys.push_back(name);
            }
        }
    }

    // compare metadata value from each metamap

    for (const openvdb::Name& name : defaultKeys) {
        openvdb::Metadata::ConstPtr metaValue1 = meta1[name];
        openvdb::Metadata::ConstPtr metaValue2 = meta2[name];

        if (!metaValue1)    return false;
        if (!metaValue2)    return false;

        if (*metaValue1 != *metaValue2)     return false;
    }

    // ensure descriptor and attributes are still in sync

    for (const auto& namePos : attrSet.descriptor().map()) {
        const size_t pos = descriptor.find(namePos.first);

        if (pos != size_t(namePos.second))  return false;
        if (descriptor.type(pos) != attrSet.get(pos)->type())   return false;
    }

    return true;
}

bool testStringVector(std::vector<std::string>& input)
{
    return input.empty();
}

bool testStringVector(std::vector<std::string>& input, const std::string& name1)
{
    if (input.size() != 1)  return false;
    if (input[0] != name1)  return false;
    return true;
}

bool testStringVector(std::vector<std::string>& input,
    const std::string& name1, const std::string& name2)
{
    if (input.size() != 2)  return false;
    if (input[0] != name1)  return false;
    if (input[1] != name2)  return false;
    return true;
}

} //unnamed  namespace


////////////////////////////////////////


void
TestAttributeSet::testAttributeSetDescriptor()
{
    // Define and register some common attribute types
    using AttributeVec3f    = TypedAttributeArray<openvdb::Vec3f>;
    using AttributeS        = TypedAttributeArray<float>;
    using AttributeI        = TypedAttributeArray<int32_t>;

    using Descriptor        = AttributeSet::Descriptor;

    { // error on invalid construction
        Descriptor::Ptr invalidDescr = Descriptor::create(AttributeVec3f::attributeType());
        CPPUNIT_ASSERT_THROW(invalidDescr->duplicateAppend("P", AttributeS::attributeType()),
            openvdb::KeyError);
    }

    Descriptor::Ptr descrA = Descriptor::create(AttributeVec3f::attributeType());

    descrA = descrA->duplicateAppend("density", AttributeS::attributeType());
    descrA = descrA->duplicateAppend("id", AttributeI::attributeType());

    Descriptor::Ptr descrB = Descriptor::create(AttributeVec3f::attributeType());

    descrB = descrB->duplicateAppend("density", AttributeS::attributeType());
    descrB = descrB->duplicateAppend("id", AttributeI::attributeType());

    CPPUNIT_ASSERT_EQUAL(descrA->size(), descrB->size());

    CPPUNIT_ASSERT(*descrA == *descrB);

    descrB->setGroup("test", size_t(0));
    descrB->setGroup("test2", size_t(1));

    Descriptor descrC(*descrB);

    CPPUNIT_ASSERT(descrB->hasSameAttributes(descrC));
    CPPUNIT_ASSERT(descrC.hasGroup("test"));
    CPPUNIT_ASSERT(*descrB == descrC);

    descrC.dropGroup("test");
    descrC.dropGroup("test2");

    CPPUNIT_ASSERT(!descrB->hasSameAttributes(descrC));
    CPPUNIT_ASSERT(!descrC.hasGroup("test"));
    CPPUNIT_ASSERT(*descrB != descrC);

    descrC.setGroup("test2", size_t(1));
    descrC.setGroup("test3", size_t(0));

    CPPUNIT_ASSERT(!descrB->hasSameAttributes(descrC));

    descrC.dropGroup("test3");
    descrC.setGroup("test", size_t(0));

    CPPUNIT_ASSERT(descrB->hasSameAttributes(descrC));

    Descriptor::Inserter names;
    names.add("P", AttributeVec3f::attributeType());
    names.add("density", AttributeS::attributeType());
    names.add("id", AttributeI::attributeType());

    // rebuild NameAndTypeVec

    Descriptor::NameAndTypeVec rebuildNames;
    descrA->appendTo(rebuildNames);

    CPPUNIT_ASSERT_EQUAL(rebuildNames.size(), names.vec.size());

    for (auto itA = rebuildNames.cbegin(), itB = names.vec.cbegin(),
              itEndA = rebuildNames.cend(), itEndB = names.vec.cend();
              itA != itEndA && itB != itEndB; ++itA, ++itB) {
        CPPUNIT_ASSERT_EQUAL(itA->name, itB->name);
        CPPUNIT_ASSERT_EQUAL(itA->type.first, itB->type.first);
        CPPUNIT_ASSERT_EQUAL(itA->type.second, itB->type.second);
    }

    Descriptor::NameToPosMap groupMap;
    openvdb::MetaMap metadata;

    // hasSameAttributes (note: uses protected create methods)
    {
        Descriptor::Ptr descr1 = Descriptor::create(Descriptor::Inserter()
                .add("P", AttributeVec3f::attributeType())
                .add("test", AttributeI::attributeType())
                .add("id", AttributeI::attributeType())
                .vec, groupMap, metadata);

        // test same names with different types, should be false
        Descriptor::Ptr descr2 = Descriptor::create(Descriptor::Inserter()
                .add("P", AttributeVec3f::attributeType())
                .add("test", AttributeS::attributeType())
                .add("id", AttributeI::attributeType())
                .vec, groupMap, metadata);

        CPPUNIT_ASSERT(!descr1->hasSameAttributes(*descr2));

        // test different names, should be false
        Descriptor::Ptr descr3 = Descriptor::create(Descriptor::Inserter()
                .add("P", AttributeVec3f::attributeType())
                .add("test2", AttributeI::attributeType())
                .add("id", AttributeI::attributeType())
                .vec, groupMap, metadata);

        CPPUNIT_ASSERT(!descr1->hasSameAttributes(*descr3));

        // test same names and types but different order, should be true
        Descriptor::Ptr descr4 = Descriptor::create(Descriptor::Inserter()
                .add("test", AttributeI::attributeType())
                .add("id", AttributeI::attributeType())
                .add("P", AttributeVec3f::attributeType())
                .vec, groupMap, metadata);

        CPPUNIT_ASSERT(descr1->hasSameAttributes(*descr4));
    }

    { // Test uniqueName
        Descriptor::Inserter names2;
        Descriptor::Ptr emptyDescr = Descriptor::create(AttributeVec3f::attributeType());
        const openvdb::Name uniqueNameEmpty = emptyDescr->uniqueName("test");
        CPPUNIT_ASSERT_EQUAL(uniqueNameEmpty, openvdb::Name("test"));

        names2.add("test", AttributeS::attributeType());
        names2.add("test1", AttributeI::attributeType());

        Descriptor::Ptr descr1 = Descriptor::create(names2.vec, groupMap, metadata);

        const openvdb::Name uniqueName1 = descr1->uniqueName("test");
        CPPUNIT_ASSERT_EQUAL(uniqueName1, openvdb::Name("test0"));

        Descriptor::Ptr descr2 = descr1->duplicateAppend(uniqueName1, AttributeI::attributeType());

        const openvdb::Name uniqueName2 = descr2->uniqueName("test");
        CPPUNIT_ASSERT_EQUAL(uniqueName2, openvdb::Name("test2"));
    }

    { // Test name validity

        CPPUNIT_ASSERT(Descriptor::validName("test1"));
        CPPUNIT_ASSERT(Descriptor::validName("abc_def"));
        CPPUNIT_ASSERT(Descriptor::validName("abc|def"));
        CPPUNIT_ASSERT(Descriptor::validName("abc:def"));

        CPPUNIT_ASSERT(!Descriptor::validName(""));
        CPPUNIT_ASSERT(!Descriptor::validName("test1!"));
        CPPUNIT_ASSERT(!Descriptor::validName("abc=def"));
        CPPUNIT_ASSERT(!Descriptor::validName("abc def"));
        CPPUNIT_ASSERT(!Descriptor::validName("abc*def"));
    }

    { // Test enforcement of valid names
        Descriptor::Ptr descr = Descriptor::create(Descriptor::Inserter().add(
            "test1", AttributeS::attributeType()).vec, groupMap, metadata);
        CPPUNIT_ASSERT_THROW(descr->rename("test1", "test1!"), openvdb::RuntimeError);
        CPPUNIT_ASSERT_THROW(descr->setGroup("group1!", 1), openvdb::RuntimeError);

        Descriptor::NameAndType invalidAttr("test1!", AttributeS::attributeType());
        CPPUNIT_ASSERT_THROW(descr->duplicateAppend(invalidAttr.name, invalidAttr.type),
            openvdb::RuntimeError);

        const openvdb::Index64 offset(0);
        const openvdb::Index64 zeroLength(0);
        const openvdb::Index64 oneLength(1);

        // write a stream with an invalid attribute
        std::ostringstream attrOstr(std::ios_base::binary);

        attrOstr.write(reinterpret_cast<const char*>(&oneLength), sizeof(openvdb::Index64));
        openvdb::writeString(attrOstr, invalidAttr.type.first);
        openvdb::writeString(attrOstr, invalidAttr.type.second);
        openvdb::writeString(attrOstr, invalidAttr.name);
        attrOstr.write(reinterpret_cast<const char*>(&offset), sizeof(openvdb::Index64));

        attrOstr.write(reinterpret_cast<const char*>(&zeroLength), sizeof(openvdb::Index64));

        // write a stream with an invalid group
        std::ostringstream groupOstr(std::ios_base::binary);

        groupOstr.write(reinterpret_cast<const char*>(&zeroLength), sizeof(openvdb::Index64));

        groupOstr.write(reinterpret_cast<const char*>(&oneLength), sizeof(openvdb::Index64));
        openvdb::writeString(groupOstr, "group1!");
        groupOstr.write(reinterpret_cast<const char*>(&offset), sizeof(openvdb::Index64));

        // read the streams back
        Descriptor inputDescr;
        std::istringstream attrIstr(attrOstr.str(), std::ios_base::binary);
        CPPUNIT_ASSERT_THROW(inputDescr.read(attrIstr), openvdb::IoError);
        std::istringstream groupIstr(groupOstr.str(), std::ios_base::binary);
        CPPUNIT_ASSERT_THROW(inputDescr.read(groupIstr), openvdb::IoError);
    }

    { // Test empty string parse
        std::vector<std::string> includeNames;
        std::vector<std::string> excludeNames;
        Descriptor::parseNames(includeNames, excludeNames, "");
        CPPUNIT_ASSERT(testStringVector(includeNames));
        CPPUNIT_ASSERT(testStringVector(excludeNames));
    }

    { // Test single token parse
        std::vector<std::string> includeNames;
        std::vector<std::string> excludeNames;
        bool includeAll = false;
        Descriptor::parseNames(includeNames, excludeNames, includeAll, "group1");
        CPPUNIT_ASSERT(!includeAll);
        CPPUNIT_ASSERT(testStringVector(includeNames, "group1"));
        CPPUNIT_ASSERT(testStringVector(excludeNames));
    }

    { // Test parse with two include tokens
        std::vector<std::string> includeNames;
        std::vector<std::string> excludeNames;
        Descriptor::parseNames(includeNames, excludeNames, "group1 group2");
        CPPUNIT_ASSERT(testStringVector(includeNames, "group1", "group2"));
        CPPUNIT_ASSERT(testStringVector(excludeNames));
    }

    { // Test parse with one include and one ^ exclude token
        std::vector<std::string> includeNames;
        std::vector<std::string> excludeNames;
        Descriptor::parseNames(includeNames, excludeNames, "group1 ^group2");
        CPPUNIT_ASSERT(testStringVector(includeNames, "group1"));
        CPPUNIT_ASSERT(testStringVector(excludeNames, "group2"));
    }

    { // Test parse with one include and one ! exclude token
        std::vector<std::string> includeNames;
        std::vector<std::string> excludeNames;
        Descriptor::parseNames(includeNames, excludeNames, "group1 !group2");
        CPPUNIT_ASSERT(testStringVector(includeNames, "group1"));
        CPPUNIT_ASSERT(testStringVector(excludeNames, "group2"));
    }

    { // Test parse one include and one exclude backwards
        std::vector<std::string> includeNames;
        std::vector<std::string> excludeNames;
        Descriptor::parseNames(includeNames, excludeNames, "^group1 group2");
        CPPUNIT_ASSERT(testStringVector(includeNames, "group2"));
        CPPUNIT_ASSERT(testStringVector(excludeNames, "group1"));
    }

    { // Test parse with two exclude tokens
        std::vector<std::string> includeNames;
        std::vector<std::string> excludeNames;
        Descriptor::parseNames(includeNames, excludeNames, "^group1 ^group2");
        CPPUNIT_ASSERT(testStringVector(includeNames));
        CPPUNIT_ASSERT(testStringVector(excludeNames, "group1", "group2"));
    }

    { // Test parse multiple includes and excludes at the same time
        std::vector<std::string> includeNames;
        std::vector<std::string> excludeNames;
        Descriptor::parseNames(includeNames, excludeNames, "group1 ^group2 ^group3 group4");
        CPPUNIT_ASSERT(testStringVector(includeNames, "group1", "group4"));
        CPPUNIT_ASSERT(testStringVector(excludeNames, "group2", "group3"));
    }

    { // Test parse misplaced negate character failure
        std::vector<std::string> includeNames;
        std::vector<std::string> excludeNames;
        CPPUNIT_ASSERT_THROW(Descriptor::parseNames(includeNames, excludeNames, "group1 ^ group2"),
            openvdb::RuntimeError);
    }

    { // Test parse (*) character
        std::vector<std::string> includeNames;
        std::vector<std::string> excludeNames;
        bool includeAll = false;
        Descriptor::parseNames(includeNames, excludeNames, includeAll, "*");
        CPPUNIT_ASSERT(includeAll);
        CPPUNIT_ASSERT(testStringVector(includeNames));
        CPPUNIT_ASSERT(testStringVector(excludeNames));
    }

    { // Test parse invalid character failure
        std::vector<std::string> includeNames;
        std::vector<std::string> excludeNames;
        CPPUNIT_ASSERT_THROW(Descriptor::parseNames(includeNames, excludeNames, "group$1"),
            openvdb::RuntimeError);
    }

    { //  Test hasGroup(), setGroup(), dropGroup(), clearGroups()
        Descriptor descr;

        CPPUNIT_ASSERT(!descr.hasGroup("test1"));

        descr.setGroup("test1", 1);

        CPPUNIT_ASSERT(descr.hasGroup("test1"));
        CPPUNIT_ASSERT_EQUAL(descr.groupMap().at("test1"), size_t(1));

        descr.setGroup("test5", 5);

        CPPUNIT_ASSERT(descr.hasGroup("test1"));
        CPPUNIT_ASSERT(descr.hasGroup("test5"));
        CPPUNIT_ASSERT_EQUAL(descr.groupMap().at("test1"), size_t(1));
        CPPUNIT_ASSERT_EQUAL(descr.groupMap().at("test5"), size_t(5));

        descr.setGroup("test1", 2);

        CPPUNIT_ASSERT(descr.hasGroup("test1"));
        CPPUNIT_ASSERT(descr.hasGroup("test5"));
        CPPUNIT_ASSERT_EQUAL(descr.groupMap().at("test1"), size_t(2));
        CPPUNIT_ASSERT_EQUAL(descr.groupMap().at("test5"), size_t(5));

        descr.dropGroup("test1");

        CPPUNIT_ASSERT(!descr.hasGroup("test1"));
        CPPUNIT_ASSERT(descr.hasGroup("test5"));

        descr.setGroup("test3", 3);

        CPPUNIT_ASSERT(descr.hasGroup("test3"));
        CPPUNIT_ASSERT(descr.hasGroup("test5"));

        descr.clearGroups();

        CPPUNIT_ASSERT(!descr.hasGroup("test1"));
        CPPUNIT_ASSERT(!descr.hasGroup("test3"));
        CPPUNIT_ASSERT(!descr.hasGroup("test5"));
    }

    // I/O test

    std::ostringstream ostr(std::ios_base::binary);
    descrA->write(ostr);

    Descriptor inputDescr;

    std::istringstream istr(ostr.str(), std::ios_base::binary);
    inputDescr.read(istr);

    CPPUNIT_ASSERT_EQUAL(descrA->size(), inputDescr.size());
    CPPUNIT_ASSERT(*descrA == inputDescr);
}


void
TestAttributeSet::testAttributeSet()
{
    // Define and register some common attribute types
    using AttributeS        = TypedAttributeArray<float>;
    using AttributeI        = TypedAttributeArray<int32_t>;
    using AttributeL        = TypedAttributeArray<int64_t>;
    using AttributeVec3s    = TypedAttributeArray<Vec3s>;

    using Descriptor        = AttributeSet::Descriptor;

    Descriptor::NameToPosMap groupMap;
    openvdb::MetaMap metadata;

    { // construction
        Descriptor::Ptr descr = Descriptor::create(AttributeVec3s::attributeType());
        descr = descr->duplicateAppend("test", AttributeI::attributeType());
        AttributeSet attrSet(descr);
        CPPUNIT_ASSERT_EQUAL(attrSet.size(), size_t(2));

        Descriptor::Ptr newDescr = Descriptor::create(AttributeVec3s::attributeType());
        CPPUNIT_ASSERT_THROW(attrSet.resetDescriptor(newDescr), openvdb::LookupError);
        CPPUNIT_ASSERT_NO_THROW(
            attrSet.resetDescriptor(newDescr, /*allowMismatchingDescriptors=*/true));
    }

    { // transfer of flags on construction
        AttributeSet attrSet(Descriptor::create(AttributeVec3s::attributeType()));
        AttributeArray::Ptr array1 = attrSet.appendAttribute(
            "hidden", AttributeS::attributeType());
        array1->setHidden(true);
        AttributeArray::Ptr array2 = attrSet.appendAttribute(
            "transient", AttributeS::attributeType());
        array2->setTransient(true);
        AttributeSet attrSet2(attrSet, size_t(1));
        CPPUNIT_ASSERT(attrSet2.getConst("hidden")->isHidden());
        CPPUNIT_ASSERT(attrSet2.getConst("transient")->isTransient());
    }

    // construct

    { // invalid append
        Descriptor::Ptr descr = Descriptor::create(AttributeVec3s::attributeType());
        AttributeSet invalidAttrSetA(descr, /*arrayLength=*/50);

        CPPUNIT_ASSERT_THROW(invalidAttrSetA.appendAttribute("id", AttributeI::attributeType(),
            /*stride=*/0, /*constantStride=*/true), openvdb::ValueError);
        CPPUNIT_ASSERT(invalidAttrSetA.find("id") == AttributeSet::INVALID_POS);
        CPPUNIT_ASSERT_THROW(invalidAttrSetA.appendAttribute("id", AttributeI::attributeType(),
            /*stride=*/49, /*constantStride=*/false), openvdb::ValueError);
        CPPUNIT_ASSERT_NO_THROW(
            invalidAttrSetA.appendAttribute("testStride1", AttributeI::attributeType(),
            /*stride=*/50, /*constantStride=*/false));
        CPPUNIT_ASSERT_NO_THROW(
            invalidAttrSetA.appendAttribute("testStride2", AttributeI::attributeType(),
            /*stride=*/51, /*constantStride=*/false));
    }

    Descriptor::Ptr descr = Descriptor::create(AttributeVec3s::attributeType());
    AttributeSet attrSetA(descr, /*arrayLength=*/50);

    attrSetA.appendAttribute("id", AttributeI::attributeType());

    // check equality against duplicate array

    Descriptor::Ptr descr2 = Descriptor::create(AttributeVec3s::attributeType());
    AttributeSet attrSetA2(descr2, /*arrayLength=*/50);

    attrSetA2.appendAttribute("id", AttributeI::attributeType());

    CPPUNIT_ASSERT(attrSetA == attrSetA2);

    // expand uniform values and check equality

    attrSetA.get("P")->expand();
    attrSetA2.get("P")->expand();

    CPPUNIT_ASSERT(attrSetA == attrSetA2);

    CPPUNIT_ASSERT_EQUAL(size_t(2), attrSetA.size());
    CPPUNIT_ASSERT_EQUAL(openvdb::Index(50), attrSetA.get(0)->size());
    CPPUNIT_ASSERT_EQUAL(openvdb::Index(50), attrSetA.get(1)->size());

    { // copy
        CPPUNIT_ASSERT(!attrSetA.isShared(0));
        CPPUNIT_ASSERT(!attrSetA.isShared(1));

        AttributeSet attrSetB(attrSetA);

        CPPUNIT_ASSERT(matchingAttributeSets(attrSetA, attrSetB));

        CPPUNIT_ASSERT(attrSetA.isShared(0));
        CPPUNIT_ASSERT(attrSetA.isShared(1));
        CPPUNIT_ASSERT(attrSetB.isShared(0));
        CPPUNIT_ASSERT(attrSetB.isShared(1));

        attrSetB.makeUnique(0);
        attrSetB.makeUnique(1);

        CPPUNIT_ASSERT(matchingAttributeSets(attrSetA, attrSetB));

        CPPUNIT_ASSERT(!attrSetA.isShared(0));
        CPPUNIT_ASSERT(!attrSetA.isShared(1));
        CPPUNIT_ASSERT(!attrSetB.isShared(0));
        CPPUNIT_ASSERT(!attrSetB.isShared(1));
    }

    { // attribute insertion
        AttributeSet attrSetB(attrSetA);

        attrSetB.makeUnique(0);
        attrSetB.makeUnique(1);

        Descriptor::Ptr targetDescr = Descriptor::create(Descriptor::Inserter()
            .add("P", AttributeVec3s::attributeType())
            .add("id", AttributeI::attributeType())
            .add("test", AttributeS::attributeType())
            .vec, groupMap, metadata);

        Descriptor::Ptr descrB =
            attrSetB.descriptor().duplicateAppend("test", AttributeS::attributeType());

        // should throw if we attempt to add the same attribute name but a different type
        CPPUNIT_ASSERT_THROW(
            descrB->insert("test", AttributeI::attributeType()), openvdb::KeyError);

        // shouldn't throw if we attempt to add the same attribute name and type
        CPPUNIT_ASSERT_NO_THROW(descrB->insert("test", AttributeS::attributeType()));

        openvdb::TypedMetadata<AttributeS::ValueType> defaultValueTest(5);

        // add a default value of the wrong type

        openvdb::TypedMetadata<int> defaultValueInt(5);

        CPPUNIT_ASSERT_THROW(descrB->setDefaultValue("test", defaultValueInt), openvdb::TypeError);

        // add a default value with a name that does not exist

        CPPUNIT_ASSERT_THROW(descrB->setDefaultValue("badname", defaultValueTest),
            openvdb::LookupError);

        // add a default value for test of 5

        descrB->setDefaultValue("test", defaultValueTest);

        {
            openvdb::Metadata::Ptr meta = descrB->getMetadata()["default:test"];
            CPPUNIT_ASSERT(meta);
            CPPUNIT_ASSERT(meta->typeName() == "float");
        }

        // ensure attribute order persists

        CPPUNIT_ASSERT_EQUAL(descrB->find("P"), size_t(0));
        CPPUNIT_ASSERT_EQUAL(descrB->find("id"), size_t(1));
        CPPUNIT_ASSERT_EQUAL(descrB->find("test"), size_t(2));

        { // simple method
            AttributeSet attrSetC(attrSetB);

            attrSetC.makeUnique(0);
            attrSetC.makeUnique(1);

            attrSetC.appendAttribute("test", AttributeS::attributeType(), /*stride=*/1,
                                        /*constantStride=*/true, defaultValueTest.copy());

            CPPUNIT_ASSERT(attributeSetMatchesDescriptor(attrSetC, *descrB));
        }
        { // descriptor-sharing method
            AttributeSet attrSetC(attrSetB);

            attrSetC.makeUnique(0);
            attrSetC.makeUnique(1);

            attrSetC.appendAttribute(attrSetC.descriptor(), descrB, size_t(2));

            CPPUNIT_ASSERT(attributeSetMatchesDescriptor(attrSetC, *targetDescr));
        }

        // add a default value for pos of (1, 3, 1)

        openvdb::TypedMetadata<AttributeVec3s::ValueType> defaultValuePos(
            AttributeVec3s::ValueType(1, 3, 1));

        descrB->setDefaultValue("P", defaultValuePos);

        {
            openvdb::Metadata::Ptr meta = descrB->getMetadata()["default:P"];
            CPPUNIT_ASSERT(meta);
            CPPUNIT_ASSERT(meta->typeName() == "vec3s");
            CPPUNIT_ASSERT_EQUAL(descrB->getDefaultValue<AttributeVec3s::ValueType>("P"),
                defaultValuePos.value());
        }

        // remove default value

        CPPUNIT_ASSERT(descrB->hasDefaultValue("test"));

        descrB->removeDefaultValue("test");

        CPPUNIT_ASSERT(!descrB->hasDefaultValue("test"));
    }

    { // attribute removal

        Descriptor::Ptr descr1 = Descriptor::create(AttributeVec3s::attributeType());

        AttributeSet attrSetB(descr1, /*arrayLength=*/50);

        attrSetB.appendAttribute("test", AttributeI::attributeType());
        attrSetB.appendAttribute("id", AttributeL::attributeType());
        attrSetB.appendAttribute("test2", AttributeI::attributeType());
        attrSetB.appendAttribute("id2", AttributeL::attributeType());
        attrSetB.appendAttribute("test3", AttributeI::attributeType());

        descr1 = attrSetB.descriptorPtr();

        Descriptor::Ptr targetDescr = Descriptor::create(AttributeVec3s::attributeType());

        targetDescr = targetDescr->duplicateAppend("id", AttributeL::attributeType());
        targetDescr = targetDescr->duplicateAppend("id2", AttributeL::attributeType());

        // add some default values

        openvdb::TypedMetadata<AttributeI::ValueType> defaultOne(AttributeI::ValueType(1));

        descr1->setDefaultValue("test", defaultOne);
        descr1->setDefaultValue("test2", defaultOne);

        openvdb::TypedMetadata<AttributeL::ValueType> defaultThree(AttributeL::ValueType(3));

        descr1->setDefaultValue("id", defaultThree);

        std::vector<size_t> toDrop{
            descr1->find("test"), descr1->find("test2"), descr1->find("test3")};

        CPPUNIT_ASSERT_EQUAL(toDrop[0], size_t(1));
        CPPUNIT_ASSERT_EQUAL(toDrop[1], size_t(3));
        CPPUNIT_ASSERT_EQUAL(toDrop[2], size_t(5));

        { // simple method
            AttributeSet attrSetC(attrSetB);

            attrSetC.makeUnique(0);
            attrSetC.makeUnique(1);
            attrSetC.makeUnique(2);
            attrSetC.makeUnique(3);

            CPPUNIT_ASSERT(attrSetC.descriptor().getMetadata()["default:test"]);

            attrSetC.dropAttributes(toDrop);

            CPPUNIT_ASSERT_EQUAL(attrSetC.size(), size_t(3));

            CPPUNIT_ASSERT(attributeSetMatchesDescriptor(attrSetC, *targetDescr));

            // check default values have been removed for the relevant attributes

            const Descriptor& descrC = attrSetC.descriptor();

            CPPUNIT_ASSERT(!descrC.getMetadata()["default:test"]);
            CPPUNIT_ASSERT(!descrC.getMetadata()["default:test2"]);
            CPPUNIT_ASSERT(!descrC.getMetadata()["default:test3"]);

            CPPUNIT_ASSERT(descrC.getMetadata()["default:id"]);
        }

        { // reverse removal order
            std::vector<size_t> toDropReverse{
                descr1->find("test3"), descr1->find("test2"), descr1->find("test")};

            AttributeSet attrSetC(attrSetB);

            attrSetC.makeUnique(0);
            attrSetC.makeUnique(1);
            attrSetC.makeUnique(2);
            attrSetC.makeUnique(3);

            attrSetC.dropAttributes(toDropReverse);

            CPPUNIT_ASSERT_EQUAL(attrSetC.size(), size_t(3));

            CPPUNIT_ASSERT(attributeSetMatchesDescriptor(attrSetC, *targetDescr));
        }

        { // descriptor-sharing method
            AttributeSet attrSetC(attrSetB);

            attrSetC.makeUnique(0);
            attrSetC.makeUnique(1);
            attrSetC.makeUnique(2);
            attrSetC.makeUnique(3);

            Descriptor::Ptr descrB = attrSetB.descriptor().duplicateDrop(toDrop);

            attrSetC.dropAttributes(toDrop, attrSetC.descriptor(), descrB);

            CPPUNIT_ASSERT_EQUAL(attrSetC.size(), size_t(3));

            CPPUNIT_ASSERT(attributeSetMatchesDescriptor(attrSetC, *targetDescr));
        }

        { // test duplicateDrop configures group mapping
            AttributeSet attrSetC;

            const size_t GROUP_BITS = sizeof(GroupType) * CHAR_BIT;

            attrSetC.appendAttribute("test1", AttributeI::attributeType());
            attrSetC.appendAttribute("__group1", GroupAttributeArray::attributeType());
            attrSetC.appendAttribute("test2", AttributeI::attributeType());
            attrSetC.appendAttribute("__group2", GroupAttributeArray::attributeType());
            attrSetC.appendAttribute("__group3", GroupAttributeArray::attributeType());
            attrSetC.appendAttribute("__group4", GroupAttributeArray::attributeType());

            // 5 attributes exist - append a group as the sixth and then drop

            Descriptor::Ptr descriptor = attrSetC.descriptorPtr();
            size_t count = descriptor->count(GroupAttributeArray::attributeType());
            CPPUNIT_ASSERT_EQUAL(count, size_t(4));

            descriptor->setGroup("test_group1", /*offset*/0); // __group1
            descriptor->setGroup("test_group2", /*offset=8*/GROUP_BITS); // __group2
            descriptor->setGroup("test_group3", /*offset=16*/GROUP_BITS*2); // __group3
            descriptor->setGroup("test_group4", /*offset=28*/GROUP_BITS*3 + GROUP_BITS/2); // __group4

            descriptor = descriptor->duplicateDrop({ 1, 2, 3 });
            count = descriptor->count(GroupAttributeArray::attributeType());
            CPPUNIT_ASSERT_EQUAL(count, size_t(2));

            CPPUNIT_ASSERT_EQUAL(size_t(3), descriptor->size());
            CPPUNIT_ASSERT(!descriptor->hasGroup("test_group1"));
            CPPUNIT_ASSERT(!descriptor->hasGroup("test_group2"));
            CPPUNIT_ASSERT(descriptor->hasGroup("test_group3"));
            CPPUNIT_ASSERT(descriptor->hasGroup("test_group4"));

            CPPUNIT_ASSERT_EQUAL(descriptor->find("__group1"), size_t(AttributeSet::INVALID_POS));
            CPPUNIT_ASSERT_EQUAL(descriptor->find("__group2"), size_t(AttributeSet::INVALID_POS));
            CPPUNIT_ASSERT_EQUAL(descriptor->find("__group3"), size_t(1));
            CPPUNIT_ASSERT_EQUAL(descriptor->find("__group4"), size_t(2));

            CPPUNIT_ASSERT_EQUAL(descriptor->groupOffset("test_group3"), size_t(0));
            CPPUNIT_ASSERT_EQUAL(descriptor->groupOffset("test_group4"), size_t(GROUP_BITS + GROUP_BITS/2));
        }
    }

    // replace existing arrays

    // this replace call should not take effect since the new attribute
    // array type does not match with the descriptor type for the given position.
    AttributeArray::Ptr floatAttr(new AttributeS(15));
    CPPUNIT_ASSERT(attrSetA.replace(1, floatAttr) == AttributeSet::INVALID_POS);

    AttributeArray::Ptr intAttr(new AttributeI(10));
    CPPUNIT_ASSERT(attrSetA.replace(1, intAttr) != AttributeSet::INVALID_POS);

    CPPUNIT_ASSERT_EQUAL(openvdb::Index(10), attrSetA.get(1)->size());

    { // reorder attribute set
        Descriptor::Ptr descr1 = Descriptor::create(AttributeVec3s::attributeType());

        AttributeSet attrSetA1(descr1);

        attrSetA1.appendAttribute("test", AttributeI::attributeType());
        attrSetA1.appendAttribute("id", AttributeI::attributeType());
        attrSetA1.appendAttribute("test2", AttributeI::attributeType());

        descr1 = attrSetA1.descriptorPtr();

        Descriptor::Ptr descr2x = Descriptor::create(AttributeVec3s::attributeType());

        AttributeSet attrSetB1(descr2x);

        attrSetB1.appendAttribute("test2", AttributeI::attributeType());
        attrSetB1.appendAttribute("test", AttributeI::attributeType());
        attrSetB1.appendAttribute("id", AttributeI::attributeType());

        CPPUNIT_ASSERT(attrSetA1 != attrSetB1);

        attrSetB1.reorderAttributes(descr1);

        CPPUNIT_ASSERT(attrSetA1 == attrSetB1);
    }

    { // metadata test
        Descriptor::Ptr descr1A = Descriptor::create(AttributeVec3s::attributeType());

        Descriptor::Ptr descr2A = Descriptor::create(AttributeVec3s::attributeType());

        openvdb::MetaMap& meta = descr1A->getMetadata();
        meta.insertMeta("exampleMeta", openvdb::FloatMetadata(2.0));

        AttributeSet attrSetA1(descr1A);
        AttributeSet attrSetB1(descr2A);
        AttributeSet attrSetC1(attrSetA1);

        CPPUNIT_ASSERT(attrSetA1 != attrSetB1);
        CPPUNIT_ASSERT(attrSetA1 == attrSetC1);
    }

    // add some metadata and register the type

    openvdb::MetaMap& meta = attrSetA.descriptor().getMetadata();
    meta.insertMeta("exampleMeta", openvdb::FloatMetadata(2.0));

    { // I/O test
        std::ostringstream ostr(std::ios_base::binary);
        attrSetA.write(ostr);

        AttributeSet attrSetB;
        std::istringstream istr(ostr.str(), std::ios_base::binary);
        attrSetB.read(istr);

        CPPUNIT_ASSERT(matchingAttributeSets(attrSetA, attrSetB));
    }

    { // I/O transient test
        AttributeArray* array = attrSetA.get(0);
        array->setTransient(true);

        std::ostringstream ostr(std::ios_base::binary);
        attrSetA.write(ostr);

        AttributeSet attrSetB;
        std::istringstream istr(ostr.str(), std::ios_base::binary);
        attrSetB.read(istr);

        // ensures transient attribute is not written out

        CPPUNIT_ASSERT_EQUAL(attrSetB.size(), size_t(1));

        std::ostringstream ostr2(std::ios_base::binary);
        attrSetA.write(ostr2, /*transient=*/true);

        AttributeSet attrSetC;
        std::istringstream istr2(ostr2.str(), std::ios_base::binary);
        attrSetC.read(istr2);

        CPPUNIT_ASSERT_EQUAL(attrSetC.size(), size_t(2));
    }
}


void
TestAttributeSet::testAttributeSetGroups()
{
    // Define and register some common attribute types
    using AttributeI        = TypedAttributeArray<int32_t>;
    using AttributeVec3s    = TypedAttributeArray<openvdb::Vec3s>;

    using Descriptor        = AttributeSet::Descriptor;

    Descriptor::NameToPosMap groupMap;
    openvdb::MetaMap metadata;

    { // construct
        Descriptor::Ptr descr = Descriptor::create(AttributeVec3s::attributeType());
        AttributeSet attrSet(descr, /*arrayLength=*/3);
        attrSet.appendAttribute("id", AttributeI::attributeType());
        CPPUNIT_ASSERT(!descr->hasGroup("test1"));
    }

    { // group offset
        Descriptor::Ptr descr = Descriptor::create(AttributeVec3s::attributeType());

        descr->setGroup("test1", 1);

        CPPUNIT_ASSERT(descr->hasGroup("test1"));
        CPPUNIT_ASSERT_EQUAL(descr->groupMap().at("test1"), size_t(1));

        AttributeSet attrSet(descr);

        CPPUNIT_ASSERT_EQUAL(attrSet.groupOffset("test1"), size_t(1));
    }

    { // group index
        Descriptor::Ptr descr = Descriptor::create(AttributeVec3s::attributeType());

        AttributeSet attrSet(descr);

        attrSet.appendAttribute("test", AttributeI::attributeType());
        attrSet.appendAttribute("test2", AttributeI::attributeType());
        attrSet.appendAttribute("group1", GroupAttributeArray::attributeType());
        attrSet.appendAttribute("test3", AttributeI::attributeType());
        attrSet.appendAttribute("group2", GroupAttributeArray::attributeType());
        attrSet.appendAttribute("test4", AttributeI::attributeType());
        attrSet.appendAttribute("group3", GroupAttributeArray::attributeType());

        descr = attrSet.descriptorPtr();

        std::stringstream ss;
        for (int i = 0; i < 17; i++) {
            ss.str("");
            ss << "test" << i;
            descr->setGroup(ss.str(), i);
        }

        Descriptor::GroupIndex index15 = attrSet.groupIndex(15);
        CPPUNIT_ASSERT_EQUAL(index15.first, size_t(5));
        CPPUNIT_ASSERT_EQUAL(index15.second, uint8_t(7));

        CPPUNIT_ASSERT_EQUAL(attrSet.groupOffset(index15), size_t(15));
        CPPUNIT_ASSERT_EQUAL(attrSet.groupOffset("test15"), size_t(15));

        Descriptor::GroupIndex index15b = attrSet.groupIndex("test15");
        CPPUNIT_ASSERT_EQUAL(index15b.first, size_t(5));
        CPPUNIT_ASSERT_EQUAL(index15b.second, uint8_t(7));

        Descriptor::GroupIndex index16 = attrSet.groupIndex(16);
        CPPUNIT_ASSERT_EQUAL(index16.first, size_t(7));
        CPPUNIT_ASSERT_EQUAL(index16.second, uint8_t(0));

        CPPUNIT_ASSERT_EQUAL(attrSet.groupOffset(index16), size_t(16));
        CPPUNIT_ASSERT_EQUAL(attrSet.groupOffset("test16"), size_t(16));

        Descriptor::GroupIndex index16b = attrSet.groupIndex("test16");
        CPPUNIT_ASSERT_EQUAL(index16b.first, size_t(7));
        CPPUNIT_ASSERT_EQUAL(index16b.second, uint8_t(0));

        // check out of range exception

        CPPUNIT_ASSERT_NO_THROW(attrSet.groupIndex(23));
        CPPUNIT_ASSERT_THROW(attrSet.groupIndex(24), LookupError);
    }

    { // group unique name
        Descriptor::Ptr descr = Descriptor::create(AttributeVec3s::attributeType());
        const openvdb::Name uniqueNameEmpty = descr->uniqueGroupName("test");
        CPPUNIT_ASSERT_EQUAL(uniqueNameEmpty, openvdb::Name("test"));

        descr->setGroup("test", 1);
        descr->setGroup("test1", 2);

        const openvdb::Name uniqueName1 = descr->uniqueGroupName("test");
        CPPUNIT_ASSERT_EQUAL(uniqueName1, openvdb::Name("test0"));
        descr->setGroup(uniqueName1, 3);

        const openvdb::Name uniqueName2 = descr->uniqueGroupName("test");
        CPPUNIT_ASSERT_EQUAL(uniqueName2, openvdb::Name("test2"));
    }

    { // group rename
        Descriptor::Ptr descr = Descriptor::create(AttributeVec3s::attributeType());
        descr->setGroup("test", 1);
        descr->setGroup("test1", 2);

        size_t pos = descr->renameGroup("test", "test1");
        CPPUNIT_ASSERT(pos == AttributeSet::INVALID_POS);
        CPPUNIT_ASSERT(descr->hasGroup("test"));
        CPPUNIT_ASSERT(descr->hasGroup("test1"));

        pos = descr->renameGroup("test", "test2");
        CPPUNIT_ASSERT_EQUAL(pos, size_t(1));
        CPPUNIT_ASSERT(!descr->hasGroup("test"));
        CPPUNIT_ASSERT(descr->hasGroup("test1"));
        CPPUNIT_ASSERT(descr->hasGroup("test2"));
    }
}

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
