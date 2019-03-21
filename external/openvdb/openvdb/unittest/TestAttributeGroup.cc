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
#include <openvdb/points/AttributeArray.h>
#include <openvdb/points/AttributeGroup.h>
#include <openvdb/points/IndexIterator.h>
#include <openvdb/points/IndexFilter.h>

#include <openvdb/openvdb.h>

#include <iostream>
#include <sstream>

using namespace openvdb;
using namespace openvdb::points;

class TestAttributeGroup: public CppUnit::TestCase
{
public:
    virtual void setUp() { openvdb::initialize(); }
    virtual void tearDown() { openvdb::uninitialize(); }

    CPPUNIT_TEST_SUITE(TestAttributeGroup);
    CPPUNIT_TEST(testAttributeGroup);
    CPPUNIT_TEST(testAttributeGroupHandle);
    CPPUNIT_TEST(testAttributeGroupFilter);

    CPPUNIT_TEST_SUITE_END();

    void testAttributeGroup();
    void testAttributeGroupHandle();
    void testAttributeGroupFilter();
}; // class TestAttributeGroup

CPPUNIT_TEST_SUITE_REGISTRATION(TestAttributeGroup);


////////////////////////////////////////


namespace {

bool
matchingNamePairs(const openvdb::NamePair& lhs,
                  const openvdb::NamePair& rhs)
{
    if (lhs.first != rhs.first)     return false;
    if (lhs.second != rhs.second)     return false;

    return true;
}

} // namespace


////////////////////////////////////////


void
TestAttributeGroup::testAttributeGroup()
{
    { // Typed class API

        const size_t count = 50;
        GroupAttributeArray attr(count);

        CPPUNIT_ASSERT(!attr.isTransient());
        CPPUNIT_ASSERT(!attr.isHidden());
        CPPUNIT_ASSERT(isGroup(attr));

        attr.setTransient(true);
        CPPUNIT_ASSERT(attr.isTransient());
        CPPUNIT_ASSERT(!attr.isHidden());
        CPPUNIT_ASSERT(isGroup(attr));

        attr.setHidden(true);
        CPPUNIT_ASSERT(attr.isTransient());
        CPPUNIT_ASSERT(attr.isHidden());
        CPPUNIT_ASSERT(isGroup(attr));

        attr.setTransient(false);
        CPPUNIT_ASSERT(!attr.isTransient());
        CPPUNIT_ASSERT(attr.isHidden());
        CPPUNIT_ASSERT(isGroup(attr));

        GroupAttributeArray attrB(attr);

        CPPUNIT_ASSERT(matchingNamePairs(attr.type(), attrB.type()));
        CPPUNIT_ASSERT_EQUAL(attr.size(), attrB.size());
        CPPUNIT_ASSERT_EQUAL(attr.memUsage(), attrB.memUsage());
        CPPUNIT_ASSERT_EQUAL(attr.isUniform(), attrB.isUniform());
        CPPUNIT_ASSERT_EQUAL(attr.isTransient(), attrB.isTransient());
        CPPUNIT_ASSERT_EQUAL(attr.isHidden(), attrB.isHidden());
        CPPUNIT_ASSERT_EQUAL(isGroup(attr), isGroup(attrB));

#if OPENVDB_ABI_VERSION_NUMBER >= 6
        AttributeArray& baseAttr(attr);
        CPPUNIT_ASSERT_EQUAL(Name(typeNameAsString<GroupType>()), baseAttr.valueType());
        CPPUNIT_ASSERT_EQUAL(Name("grp"), baseAttr.codecType());
        CPPUNIT_ASSERT_EQUAL(Index(1), baseAttr.valueTypeSize());
        CPPUNIT_ASSERT_EQUAL(Index(1), baseAttr.storageTypeSize());
        CPPUNIT_ASSERT(!baseAttr.valueTypeIsFloatingPoint());
#endif
    }

    { // casting
        TypedAttributeArray<float> floatAttr(4);
        AttributeArray& floatArray = floatAttr;
        const AttributeArray& constFloatArray = floatAttr;

        CPPUNIT_ASSERT_THROW(GroupAttributeArray::cast(floatArray), TypeError);
        CPPUNIT_ASSERT_THROW(GroupAttributeArray::cast(constFloatArray), TypeError);

        GroupAttributeArray groupAttr(4);
        AttributeArray& groupArray = groupAttr;
        const AttributeArray& constGroupArray = groupAttr;

        CPPUNIT_ASSERT_NO_THROW(GroupAttributeArray::cast(groupArray));
        CPPUNIT_ASSERT_NO_THROW(GroupAttributeArray::cast(constGroupArray));
    }

    { // IO
        const size_t count = 50;
        GroupAttributeArray attrA(count);

        for (unsigned i = 0; i < unsigned(count); ++i) {
            attrA.set(i, int(i));
        }

        attrA.setHidden(true);

        std::ostringstream ostr(std::ios_base::binary);
        attrA.write(ostr);

        GroupAttributeArray attrB;

        std::istringstream istr(ostr.str(), std::ios_base::binary);
        attrB.read(istr);

        CPPUNIT_ASSERT(matchingNamePairs(attrA.type(), attrB.type()));
        CPPUNIT_ASSERT_EQUAL(attrA.size(), attrB.size());
        CPPUNIT_ASSERT_EQUAL(attrA.memUsage(), attrB.memUsage());
        CPPUNIT_ASSERT_EQUAL(attrA.isUniform(), attrB.isUniform());
        CPPUNIT_ASSERT_EQUAL(attrA.isTransient(), attrB.isTransient());
        CPPUNIT_ASSERT_EQUAL(attrA.isHidden(), attrB.isHidden());
        CPPUNIT_ASSERT_EQUAL(isGroup(attrA), isGroup(attrB));

        for (unsigned i = 0; i < unsigned(count); ++i) {
            CPPUNIT_ASSERT_EQUAL(attrA.get(i), attrB.get(i));
        }
    }
}


void
TestAttributeGroup::testAttributeGroupHandle()
{
    GroupAttributeArray attr(4);
    GroupHandle handle(attr, 3);

    CPPUNIT_ASSERT_EQUAL(handle.size(), Index(4));
    CPPUNIT_ASSERT_EQUAL(handle.size(), attr.size());

    // construct bitmasks

    const GroupType bitmask3 = GroupType(1) << 3;
    const GroupType bitmask6 = GroupType(1) << 6;
    const GroupType bitmask36 = GroupType(1) << 3 | GroupType(1) << 6;

    // enable attribute 1,2,3 for group permutations of 3 and 6
    attr.set(0, 0);
    attr.set(1, bitmask3);
    attr.set(2, bitmask6);
    attr.set(3, bitmask36);

    CPPUNIT_ASSERT(attr.get(2) != bitmask36);
    CPPUNIT_ASSERT_EQUAL(attr.get(3), bitmask36);

    { // group 3 valid for attributes 1 and 3 (using specific offset)
        GroupHandle handle3(attr, 3);

        CPPUNIT_ASSERT(!handle3.get(0));
        CPPUNIT_ASSERT(handle3.get(1));
        CPPUNIT_ASSERT(!handle3.get(2));
        CPPUNIT_ASSERT(handle3.get(3));
    }

    { // test group 3 valid for attributes 1 and 3 (unsafe access)
        GroupHandle handle3(attr, 3);

        CPPUNIT_ASSERT(!handle3.getUnsafe(0));
        CPPUNIT_ASSERT(handle3.getUnsafe(1));
        CPPUNIT_ASSERT(!handle3.getUnsafe(2));
        CPPUNIT_ASSERT(handle3.getUnsafe(3));
    }

    { // group 6 valid for attributes 2 and 3 (using specific offset)
        GroupHandle handle6(attr, 6);

        CPPUNIT_ASSERT(!handle6.get(0));
        CPPUNIT_ASSERT(!handle6.get(1));
        CPPUNIT_ASSERT(handle6.get(2));
        CPPUNIT_ASSERT(handle6.get(3));
    }

    { // groups 3 and 6 only valid for attribute 3 (using bitmask)
        GroupHandle handle36(attr, bitmask36, GroupHandle::BitMask());

        CPPUNIT_ASSERT(!handle36.get(0));
        CPPUNIT_ASSERT(!handle36.get(1));
        CPPUNIT_ASSERT(!handle36.get(2));
        CPPUNIT_ASSERT(handle36.get(3));
    }

    // clear the array

    attr.fill(0);

    CPPUNIT_ASSERT_EQUAL(attr.get(1), GroupType(0));

    // write handles

    GroupWriteHandle writeHandle3(attr, 3);
    GroupWriteHandle writeHandle6(attr, 6);

    // test collapse

    CPPUNIT_ASSERT_EQUAL(writeHandle3.get(1), false);
    CPPUNIT_ASSERT_EQUAL(writeHandle6.get(1), false);

    CPPUNIT_ASSERT(writeHandle6.compact());
    CPPUNIT_ASSERT(writeHandle6.isUniform());

    attr.expand();

    CPPUNIT_ASSERT(!writeHandle6.isUniform());

    CPPUNIT_ASSERT(writeHandle3.collapse(true));

    CPPUNIT_ASSERT(attr.isUniform());
    CPPUNIT_ASSERT(writeHandle3.isUniform());
    CPPUNIT_ASSERT(writeHandle6.isUniform());

    CPPUNIT_ASSERT_EQUAL(writeHandle3.get(1), true);
    CPPUNIT_ASSERT_EQUAL(writeHandle6.get(1), false);

    CPPUNIT_ASSERT(writeHandle3.collapse(false));

    CPPUNIT_ASSERT(writeHandle3.isUniform());
    CPPUNIT_ASSERT_EQUAL(writeHandle3.get(1), false);

    attr.fill(0);

    writeHandle3.set(1, true);

    CPPUNIT_ASSERT(!attr.isUniform());
    CPPUNIT_ASSERT(!writeHandle3.isUniform());
    CPPUNIT_ASSERT(!writeHandle6.isUniform());

    CPPUNIT_ASSERT(!writeHandle3.collapse(true));

    CPPUNIT_ASSERT(!attr.isUniform());
    CPPUNIT_ASSERT(!writeHandle3.isUniform());
    CPPUNIT_ASSERT(!writeHandle6.isUniform());

    CPPUNIT_ASSERT_EQUAL(writeHandle3.get(1), true);
    CPPUNIT_ASSERT_EQUAL(writeHandle6.get(1), false);

    writeHandle6.set(2, true);

    CPPUNIT_ASSERT(!writeHandle3.collapse(false));

    CPPUNIT_ASSERT(!writeHandle3.isUniform());

    attr.fill(0);

    writeHandle3.set(1, true);
    writeHandle6.set(2, true);
    writeHandle3.set(3, true);
    writeHandle6.set(3, true);

    { // group 3 valid for attributes 1 and 3 (using specific offset)
        GroupHandle handle3(attr, 3);

        CPPUNIT_ASSERT(!handle3.get(0));
        CPPUNIT_ASSERT(handle3.get(1));
        CPPUNIT_ASSERT(!handle3.get(2));
        CPPUNIT_ASSERT(handle3.get(3));

        CPPUNIT_ASSERT(!writeHandle3.get(0));
        CPPUNIT_ASSERT(writeHandle3.get(1));
        CPPUNIT_ASSERT(!writeHandle3.get(2));
        CPPUNIT_ASSERT(writeHandle3.get(3));
    }

    { // group 6 valid for attributes 2 and 3 (using specific offset)
        GroupHandle handle6(attr, 6);

        CPPUNIT_ASSERT(!handle6.get(0));
        CPPUNIT_ASSERT(!handle6.get(1));
        CPPUNIT_ASSERT(handle6.get(2));
        CPPUNIT_ASSERT(handle6.get(3));

        CPPUNIT_ASSERT(!writeHandle6.get(0));
        CPPUNIT_ASSERT(!writeHandle6.get(1));
        CPPUNIT_ASSERT(writeHandle6.get(2));
        CPPUNIT_ASSERT(writeHandle6.get(3));
    }

    writeHandle3.set(3, false);

    { // group 3 valid for attributes 1 and 3 (using specific offset)
        GroupHandle handle3(attr, 3);

        CPPUNIT_ASSERT(!handle3.get(0));
        CPPUNIT_ASSERT(handle3.get(1));
        CPPUNIT_ASSERT(!handle3.get(2));
        CPPUNIT_ASSERT(!handle3.get(3));

        CPPUNIT_ASSERT(!writeHandle3.get(0));
        CPPUNIT_ASSERT(writeHandle3.get(1));
        CPPUNIT_ASSERT(!writeHandle3.get(2));
        CPPUNIT_ASSERT(!writeHandle3.get(3));
    }

    { // group 6 valid for attributes 2 and 3 (using specific offset)
        GroupHandle handle6(attr, 6);

        CPPUNIT_ASSERT(!handle6.get(0));
        CPPUNIT_ASSERT(!handle6.get(1));
        CPPUNIT_ASSERT(handle6.get(2));
        CPPUNIT_ASSERT(handle6.get(3));

        CPPUNIT_ASSERT(!writeHandle6.get(0));
        CPPUNIT_ASSERT(!writeHandle6.get(1));
        CPPUNIT_ASSERT(writeHandle6.get(2));
        CPPUNIT_ASSERT(writeHandle6.get(3));
    }
}


class GroupNotFilter
{
public:
    explicit GroupNotFilter(const AttributeSet::Descriptor::GroupIndex& index)
        : mFilter(index) { }

    inline bool initialized() const { return mFilter.initialized(); }

    template <typename LeafT>
    void reset(const LeafT& leaf) {
        mFilter.reset(leaf);
    }

    template <typename IterT>
    bool valid(const IterT& iter) const {
        return !mFilter.valid(iter);
    }

private:
    GroupFilter mFilter;
}; // class GroupNotFilter


struct HandleWrapper
{
    HandleWrapper(const GroupHandle& handle)
        : mHandle(handle) { }

    GroupHandle groupHandle(const AttributeSet::Descriptor::GroupIndex& /*index*/) const {
        return mHandle;
    }

private:
    const GroupHandle mHandle;
}; // struct HandleWrapper


void
TestAttributeGroup::testAttributeGroupFilter()
{
    using GroupIndex = AttributeSet::Descriptor::GroupIndex;

    GroupIndex zeroIndex;

    typedef IndexIter<ValueVoxelCIter, GroupFilter> IndexGroupAllIter;

    GroupAttributeArray attrGroup(4);
    const Index32 size = attrGroup.size();

    { // group values all zero
        ValueVoxelCIter indexIter(0, size);
        GroupFilter filter(zeroIndex);
        CPPUNIT_ASSERT(filter.state() == index::PARTIAL);
        filter.reset(HandleWrapper(GroupHandle(attrGroup, 0)));
        IndexGroupAllIter iter(indexIter, filter);

        CPPUNIT_ASSERT(!iter);
    }

    // enable attributes 0 and 2 for groups 3 and 6

    const GroupType bitmask = GroupType(1) << 3 | GroupType(1) << 6;

    attrGroup.set(0, bitmask);
    attrGroup.set(2, bitmask);

    // index iterator only valid in groups 3 and 6
    {
        ValueVoxelCIter indexIter(0, size);

        GroupFilter filter(zeroIndex);

        filter.reset(HandleWrapper(GroupHandle(attrGroup, 0)));
        CPPUNIT_ASSERT(!IndexGroupAllIter(indexIter, filter));
        filter.reset(HandleWrapper(GroupHandle(attrGroup, 1)));
        CPPUNIT_ASSERT(!IndexGroupAllIter(indexIter, filter));
        filter.reset(HandleWrapper(GroupHandle(attrGroup, 2)));
        CPPUNIT_ASSERT(!IndexGroupAllIter(indexIter, filter));
        filter.reset(HandleWrapper(GroupHandle(attrGroup, 3)));
        CPPUNIT_ASSERT(IndexGroupAllIter(indexIter, filter));
        filter.reset(HandleWrapper(GroupHandle(attrGroup, 4)));
        CPPUNIT_ASSERT(!IndexGroupAllIter(indexIter, filter));
        filter.reset(HandleWrapper(GroupHandle(attrGroup, 5)));
        CPPUNIT_ASSERT(!IndexGroupAllIter(indexIter, filter));
        filter.reset(HandleWrapper(GroupHandle(attrGroup, 6)));
        CPPUNIT_ASSERT(IndexGroupAllIter(indexIter, filter));
        filter.reset(HandleWrapper(GroupHandle(attrGroup, 7)));
        CPPUNIT_ASSERT(!IndexGroupAllIter(indexIter, filter));
    }

    attrGroup.set(1, bitmask);
    attrGroup.set(3, bitmask);

    using IndexNotGroupAllIter = IndexIter<ValueVoxelCIter, GroupNotFilter>;

    // index iterator only not valid in groups 3 and 6
    {
        ValueVoxelCIter indexIter(0, size);

        GroupNotFilter filter(zeroIndex);

        filter.reset(HandleWrapper(GroupHandle(attrGroup, 0)));
        CPPUNIT_ASSERT(IndexNotGroupAllIter(indexIter, filter));
        filter.reset(HandleWrapper(GroupHandle(attrGroup, 1)));
        CPPUNIT_ASSERT(IndexNotGroupAllIter(indexIter, filter));
        filter.reset(HandleWrapper(GroupHandle(attrGroup, 2)));
        CPPUNIT_ASSERT(IndexNotGroupAllIter(indexIter, filter));
        filter.reset(HandleWrapper(GroupHandle(attrGroup, 3)));
        CPPUNIT_ASSERT(!IndexNotGroupAllIter(indexIter, filter));
        filter.reset(HandleWrapper(GroupHandle(attrGroup, 4)));
        CPPUNIT_ASSERT(IndexNotGroupAllIter(indexIter, filter));
        filter.reset(HandleWrapper(GroupHandle(attrGroup, 5)));
        CPPUNIT_ASSERT(IndexNotGroupAllIter(indexIter, filter));
        filter.reset(HandleWrapper(GroupHandle(attrGroup, 6)));
        CPPUNIT_ASSERT(!IndexNotGroupAllIter(indexIter, filter));
        filter.reset(HandleWrapper(GroupHandle(attrGroup, 7)));
        CPPUNIT_ASSERT(IndexNotGroupAllIter(indexIter, filter));
    }

    // clear group membership for attributes 1 and 3

    attrGroup.set(1, GroupType(0));
    attrGroup.set(3, GroupType(0));

    { // index in group next
        ValueVoxelCIter indexIter(0, size);
        GroupFilter filter(zeroIndex);
        filter.reset(HandleWrapper(GroupHandle(attrGroup, 3)));
        IndexGroupAllIter iter(indexIter, filter);

        CPPUNIT_ASSERT(iter);
        CPPUNIT_ASSERT_EQUAL(*iter, Index32(0));

        CPPUNIT_ASSERT(iter.next());
        CPPUNIT_ASSERT_EQUAL(*iter, Index32(2));

        CPPUNIT_ASSERT(!iter.next());
    }

    { // index in group prefix ++
        ValueVoxelCIter indexIter(0, size);
        GroupFilter filter(zeroIndex);
        filter.reset(HandleWrapper(GroupHandle(attrGroup, 3)));
        IndexGroupAllIter iter(indexIter, filter);

        CPPUNIT_ASSERT(iter);
        CPPUNIT_ASSERT_EQUAL(*iter, Index32(0));

        IndexGroupAllIter old = ++iter;
        CPPUNIT_ASSERT_EQUAL(*old, Index32(2));
        CPPUNIT_ASSERT_EQUAL(*iter, Index32(2));

        CPPUNIT_ASSERT(!iter.next());
    }

    { // index in group postfix ++/--
        ValueVoxelCIter indexIter(0, size);
        GroupFilter filter(zeroIndex);
        filter.reset(HandleWrapper(GroupHandle(attrGroup, 3)));
        IndexGroupAllIter iter(indexIter, filter);

        CPPUNIT_ASSERT(iter);
        CPPUNIT_ASSERT_EQUAL(*iter, Index32(0));

        IndexGroupAllIter old = iter++;
        CPPUNIT_ASSERT_EQUAL(*old, Index32(0));
        CPPUNIT_ASSERT_EQUAL(*iter, Index32(2));

        CPPUNIT_ASSERT(!iter.next());
    }

    { // index not in group next
        ValueVoxelCIter indexIter(0, size);
        GroupNotFilter filter(zeroIndex);
        filter.reset(HandleWrapper(GroupHandle(attrGroup, 3)));
        IndexNotGroupAllIter iter(indexIter, filter);

        CPPUNIT_ASSERT(iter);
        CPPUNIT_ASSERT_EQUAL(*iter, Index32(1));

        CPPUNIT_ASSERT(iter.next());
        CPPUNIT_ASSERT_EQUAL(*iter, Index32(3));

        CPPUNIT_ASSERT(!iter.next());
    }

    { // index not in group prefix ++
        ValueVoxelCIter indexIter(0, size);
        GroupNotFilter filter(zeroIndex);
        filter.reset(HandleWrapper(GroupHandle(attrGroup, 3)));
        IndexNotGroupAllIter iter(indexIter, filter);

        CPPUNIT_ASSERT(iter);
        CPPUNIT_ASSERT_EQUAL(*iter, Index32(1));

        IndexNotGroupAllIter old = ++iter;
        CPPUNIT_ASSERT_EQUAL(*old, Index32(3));
        CPPUNIT_ASSERT_EQUAL(*iter, Index32(3));

        CPPUNIT_ASSERT(!iter.next());
    }

    { // index not in group postfix ++
        ValueVoxelCIter indexIter(0, size);
        GroupNotFilter filter(zeroIndex);
        filter.reset(HandleWrapper(GroupHandle(attrGroup, 3)));
        IndexNotGroupAllIter iter(indexIter, filter);

        CPPUNIT_ASSERT(iter);
        CPPUNIT_ASSERT_EQUAL(*iter, Index32(1));

        IndexNotGroupAllIter old = iter++;
        CPPUNIT_ASSERT_EQUAL(*old, Index32(1));
        CPPUNIT_ASSERT_EQUAL(*iter, Index32(3));

        CPPUNIT_ASSERT(!iter.next());
    }
}

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
