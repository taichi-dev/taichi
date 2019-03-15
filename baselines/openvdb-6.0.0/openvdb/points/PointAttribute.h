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

/// @author Dan Bailey, Khang Ngo
///
/// @file points/PointAttribute.h
///
/// @brief  Point attribute manipulation in a VDB Point Grid.

#ifndef OPENVDB_POINTS_POINT_ATTRIBUTE_HAS_BEEN_INCLUDED
#define OPENVDB_POINTS_POINT_ATTRIBUTE_HAS_BEEN_INCLUDED

#include <openvdb/openvdb.h>

#include "AttributeArrayString.h"
#include "AttributeSet.h"
#include "AttributeGroup.h"
#include "PointDataGrid.h"


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace points {

namespace point_attribute_internal {

template <typename ValueType>
struct Default
{
    static inline ValueType value() { return zeroVal<ValueType>(); }
};

} // namespace point_attribute_internal


/// @brief Appends a new attribute to the VDB tree
/// (this method does not require a templated AttributeType)
///
/// @param tree               the PointDataTree to be appended to.
/// @param name               name for the new attribute.
/// @param type               the type of the attibute.
/// @param strideOrTotalSize  the stride of the attribute
/// @param constantStride     if @c false, stride is interpreted as total size of the array
/// @param metaDefaultValue   metadata default attribute value
/// @param hidden             mark attribute as hidden
/// @param transient          mark attribute as transient
template <typename PointDataTreeT>
inline void appendAttribute(PointDataTreeT& tree,
                            const Name& name,
                            const NamePair& type,
                            const Index strideOrTotalSize = 1,
                            const bool constantStride = true,
                            Metadata::Ptr metaDefaultValue = Metadata::Ptr(),
                            const bool hidden = false,
                            const bool transient = false);

/// @brief Appends a new attribute to the VDB tree.
///
/// @param tree               the PointDataTree to be appended to.
/// @param name               name for the new attribute
/// @param uniformValue       the initial value of the attribute
/// @param strideOrTotalSize  the stride of the attribute
/// @param constantStride     if @c false, stride is interpreted as total size of the array
/// @param metaDefaultValue   metadata default attribute value
/// @param hidden             mark attribute as hidden
/// @param transient          mark attribute as transient
template <typename ValueType,
          typename CodecType = NullCodec,
          typename PointDataTreeT = PointDataTree>
inline void appendAttribute(PointDataTreeT& tree,
                            const std::string& name,
                            const ValueType& uniformValue =
                                point_attribute_internal::Default<ValueType>::value(),
                            const Index strideOrTotalSize = 1,
                            const bool constantStride = true,
                            Metadata::Ptr metaDefaultValue = Metadata::Ptr(),
                            const bool hidden = false,
                            const bool transient = false);

/// @brief Collapse the attribute into a uniform value
///
/// @param tree         the PointDataTree in which to collapse the attribute.
/// @param name         name for the attribute.
/// @param uniformValue value of the attribute
template <typename ValueType, typename PointDataTreeT>
inline void collapseAttribute(  PointDataTreeT& tree,
                                const Name& name,
                                const ValueType& uniformValue =
                                    point_attribute_internal::Default<ValueType>::value());

/// @brief Drops attributes from the VDB tree.
///
/// @param tree          the PointDataTree to be dropped from.
/// @param indices       indices of the attributes to drop.
template <typename PointDataTreeT>
inline void dropAttributes( PointDataTreeT& tree,
                            const std::vector<size_t>& indices);

/// @brief Drops attributes from the VDB tree.
///
/// @param tree          the PointDataTree to be dropped from.
/// @param names         names of the attributes to drop.
template <typename PointDataTreeT>
inline void dropAttributes( PointDataTreeT& tree,
                            const std::vector<Name>& names);

/// @brief Drop one attribute from the VDB tree (convenience method).
///
/// @param tree          the PointDataTree to be dropped from.
/// @param index         index of the attribute to drop.
template <typename PointDataTreeT>
inline void dropAttribute(  PointDataTreeT& tree,
                            const size_t& index);

/// @brief Drop one attribute from the VDB tree (convenience method).
///
/// @param tree          the PointDataTree to be dropped from.
/// @param name          name of the attribute to drop.
template <typename PointDataTreeT>
inline void dropAttribute(  PointDataTreeT& tree,
                            const Name& name);

/// @brief Rename attributes in a VDB tree.
///
/// @param tree          the PointDataTree.
/// @param oldNames      a list of old attribute names to rename from.
/// @param newNames      a list of new attribute names to rename to.
///
/// @note Number of oldNames must match the number of newNames.
///
/// @note Duplicate names and renaming group attributes are not allowed.
template <typename PointDataTreeT>
inline void renameAttributes(PointDataTreeT& tree,
                            const std::vector<Name>& oldNames,
                            const std::vector<Name>& newNames);

/// @brief Rename an attribute in a VDB tree.
///
/// @param tree          the PointDataTree.
/// @param oldName       the old attribute name to rename from.
/// @param newName       the new attribute name to rename to.
///
/// @note newName must not already exist and must not be a group attribute.
template <typename PointDataTreeT>
inline void renameAttribute(PointDataTreeT& tree,
                            const Name& oldName,
                            const Name& newName);

/// @brief Compact attributes in a VDB tree (if possible).
///
/// @param tree          the PointDataTree.
template <typename PointDataTreeT>
inline void compactAttributes(PointDataTreeT& tree);


////////////////////////////////////////


namespace point_attribute_internal {

template<typename PointDataTreeT>
struct AppendAttributeOp {

    using LeafManagerT  = typename tree::LeafManager<PointDataTreeT>;
    using LeafRangeT    = typename LeafManagerT::LeafRange;

    AppendAttributeOp(  AttributeSet::DescriptorPtr& descriptor,
                        const size_t pos,
                        const Index strideOrTotalSize = 1,
                        const bool constantStride = true,
                        const bool hidden = false,
                        const bool transient = false)
        : mDescriptor(descriptor)
        , mPos(pos)
        , mStrideOrTotalSize(strideOrTotalSize)
        , mConstantStride(constantStride)
        , mHidden(hidden)
        , mTransient(transient) { }

    void operator()(const LeafRangeT& range) const {

        for (auto leaf = range.begin(); leaf; ++leaf) {
            const AttributeSet::Descriptor::Ptr expected = leaf->attributeSet().descriptorPtr();

            AttributeArray::Ptr attribute = leaf->appendAttribute(
                *expected, mDescriptor, mPos, mStrideOrTotalSize, mConstantStride);

            if (mHidden)      attribute->setHidden(true);
            if (mTransient)   attribute->setTransient(true);
        }
    }

    //////////

    AttributeSet::DescriptorPtr&    mDescriptor;
    const size_t                    mPos;
    const Index                     mStrideOrTotalSize;
    const bool                      mConstantStride;
    const bool                      mHidden;
    const bool                      mTransient;
}; // class AppendAttributeOp


////////////////////////////////////////


template <typename ValueType, typename PointDataTreeT>
struct CollapseAttributeOp {

    using LeafManagerT  = typename tree::LeafManager<PointDataTreeT>;
    using LeafRangeT    = typename LeafManagerT::LeafRange;

    CollapseAttributeOp(const size_t pos,
                        const ValueType& uniformValue)
        : mPos(pos)
        , mUniformValue(uniformValue) { }

    void operator()(const LeafRangeT& range) const {

        for (auto leaf = range.begin(); leaf; ++leaf) {
            assert(leaf->hasAttribute(mPos));
            AttributeArray& array = leaf->attributeArray(mPos);
            AttributeWriteHandle<ValueType> handle(array);
            handle.collapse(mUniformValue);
        }
    }

    //////////

    const size_t                                mPos;
    const ValueType                             mUniformValue;
}; // class CollapseAttributeOp


////////////////////////////////////////


template <typename PointDataTreeT>
struct CollapseAttributeOp<Name, PointDataTreeT> {

    using LeafManagerT  = typename tree::LeafManager<PointDataTreeT>;
    using LeafRangeT    = typename LeafManagerT::LeafRange;

    CollapseAttributeOp(const size_t pos,
                        const Name& uniformValue)
        : mPos(pos)
        , mUniformValue(uniformValue) { }

    void operator()(const LeafRangeT& range) const {

        for (auto leaf = range.begin(); leaf; ++leaf) {
            assert(leaf->hasAttribute(mPos));
            AttributeArray& array = leaf->attributeArray(mPos);

            const AttributeSet::Descriptor& descriptor = leaf->attributeSet().descriptor();
            const MetaMap& metadata = descriptor.getMetadata();

            StringAttributeWriteHandle handle(array, metadata);
            handle.collapse(mUniformValue);
        }
    }

    //////////

    const size_t                                mPos;
    const Name                                  mUniformValue;
}; // class CollapseAttributeOp


////////////////////////////////////////


template<typename PointDataTreeT>
struct DropAttributesOp {

    using LeafManagerT  = typename tree::LeafManager<PointDataTreeT>;
    using LeafRangeT    = typename LeafManagerT::LeafRange;
    using Indices       = std::vector<size_t>;

    DropAttributesOp(   const Indices& indices,
                        AttributeSet::DescriptorPtr& descriptor)
        : mIndices(indices)
        , mDescriptor(descriptor) { }

    void operator()(const LeafRangeT& range) const {

        for (auto leaf = range.begin(); leaf; ++leaf) {

            const AttributeSet::Descriptor::Ptr expected = leaf->attributeSet().descriptorPtr();

            leaf->dropAttributes(mIndices, *expected, mDescriptor);
        }
    }

    //////////

    const Indices&                  mIndices;
    AttributeSet::DescriptorPtr&    mDescriptor;
}; // class DropAttributesOp


////////////////////////////////////////


template<typename PointDataTreeT>
struct CompactAttributesOp {

    using LeafManagerT  = typename tree::LeafManager<PointDataTreeT>;
    using LeafRangeT    = typename LeafManagerT::LeafRange;

    void operator()(const LeafRangeT& range) const {
        for (auto leaf = range.begin(); leaf; ++leaf) {
            leaf->compactAttributes();
        }
    }
}; // class CompactAttributesOp


////////////////////////////////////////


template <typename ValueType, typename CodecType>
struct AttributeTypeConversion
{
    static const NamePair& type() {
        return TypedAttributeArray<ValueType, CodecType>::attributeType();
    }
};


template <typename CodecType>
struct AttributeTypeConversion<Name, CodecType>
{
    static const NamePair& type() { return StringAttributeArray::attributeType(); }
};


////////////////////////////////////////


template <typename PointDataTreeT, typename ValueType>
struct MetadataStorage
{
    static void add(PointDataTreeT&, const ValueType&) {}

    template<typename AttributeListType>
    static void add(PointDataTreeT&, const AttributeListType&) {}
};


template <typename PointDataTreeT>
struct MetadataStorage<PointDataTreeT, Name>
{
    static void add(PointDataTreeT& tree, const Name& uniformValue) {
        MetaMap& metadata = makeDescriptorUnique(tree)->getMetadata();
        StringMetaInserter inserter(metadata);
        inserter.insert(uniformValue);
    }

    template<typename AttributeListType>
    static void add(PointDataTreeT& tree, const AttributeListType& data) {
        MetaMap& metadata = makeDescriptorUnique(tree)->getMetadata();
        StringMetaInserter inserter(metadata);
        Name value;

        for (size_t i = 0; i < data.size(); i++) {
            data.get(value, i);
            inserter.insert(value);
        }
    }
};


} // namespace point_attribute_internal


////////////////////////////////////////


template <typename PointDataTreeT>
inline void appendAttribute(PointDataTreeT& tree,
                            const Name& name,
                            const NamePair& type,
                            const Index strideOrTotalSize,
                            const bool constantStride,
                            Metadata::Ptr metaDefaultValue,
                            const bool hidden,
                            const bool transient)
{
    using Descriptor = AttributeSet::Descriptor;

    using point_attribute_internal::AppendAttributeOp;

    auto iter = tree.cbeginLeaf();

    if (!iter)  return;

    // do not append a non-unique attribute

    const Descriptor& descriptor = iter->attributeSet().descriptor();
    const size_t index = descriptor.find(name);

    if (index != AttributeSet::INVALID_POS) {
        OPENVDB_THROW(KeyError,
            "Cannot append an attribute with a non-unique name - " << name << ".");
    }

    // create a new attribute descriptor

    Descriptor::Ptr newDescriptor = descriptor.duplicateAppend(name, type);

    // store the attribute default value in the descriptor metadata

    if (metaDefaultValue) {
        newDescriptor->setDefaultValue(name, *metaDefaultValue);
    }

    // extract new pos

    const size_t pos = newDescriptor->find(name);

    // insert attributes using the new descriptor

    tree::LeafManager<PointDataTreeT> leafManager(tree);
    AppendAttributeOp<PointDataTreeT> append(newDescriptor, pos, strideOrTotalSize,
                                            constantStride, hidden, transient);
    tbb::parallel_for(leafManager.leafRange(), append);
}


////////////////////////////////////////


template <typename ValueType, typename CodecType, typename PointDataTreeT>
inline void appendAttribute(PointDataTreeT& tree,
                            const std::string& name,
                            const ValueType& uniformValue,
                            const Index strideOrTotalSize,
                            const bool constantStride,
                            Metadata::Ptr metaDefaultValue,
                            const bool hidden,
                            const bool transient)
{
    static_assert(!std::is_base_of<AttributeArray, ValueType>::value,
        "ValueType must not be derived from AttributeArray");

    using point_attribute_internal::AttributeTypeConversion;
    using point_attribute_internal::Default;
    using point_attribute_internal::MetadataStorage;

    appendAttribute(tree, name, AttributeTypeConversion<ValueType, CodecType>::type(),
        strideOrTotalSize, constantStride, metaDefaultValue, hidden, transient);

    if (!math::isExactlyEqual(uniformValue, Default<ValueType>::value())) {
        MetadataStorage<PointDataTreeT, ValueType>::add(tree, uniformValue);
        collapseAttribute<ValueType>(tree, name, uniformValue);
    }
}


////////////////////////////////////////


template <typename ValueType, typename PointDataTreeT>
inline void collapseAttribute(  PointDataTreeT& tree,
                                const Name& name,
                                const ValueType& uniformValue)
{
    static_assert(!std::is_base_of<AttributeArray, ValueType>::value,
        "ValueType must not be derived from AttributeArray");

    using LeafManagerT  = typename tree::LeafManager<PointDataTreeT>;
    using Descriptor    = AttributeSet::Descriptor;

    using point_attribute_internal::CollapseAttributeOp;

    auto iter = tree.cbeginLeaf();

    if (!iter)  return;


    const Descriptor& descriptor = iter->attributeSet().descriptor();

    // throw if attribute name does not exist

    const size_t index = descriptor.find(name);
    if (index == AttributeSet::INVALID_POS) {
        OPENVDB_THROW(KeyError, "Cannot find attribute name in PointDataTree.");
    }

    LeafManagerT leafManager(tree);
    tbb::parallel_for(leafManager.leafRange(),
        CollapseAttributeOp<ValueType, PointDataTreeT>(index, uniformValue));
}


////////////////////////////////////////


template <typename PointDataTreeT>
inline void dropAttributes( PointDataTreeT& tree,
                            const std::vector<size_t>& indices)
{
    using LeafManagerT  = typename tree::LeafManager<PointDataTreeT>;
    using Descriptor    = AttributeSet::Descriptor;

    using point_attribute_internal::DropAttributesOp;

    auto iter = tree.cbeginLeaf();

    if (!iter)  return;

    const Descriptor& descriptor = iter->attributeSet().descriptor();

    // throw if position index present in the indices as this attribute is mandatory

    const size_t positionIndex = descriptor.find("P");
    if (positionIndex!= AttributeSet::INVALID_POS &&
        std::find(indices.begin(), indices.end(), positionIndex) != indices.end()) {
        OPENVDB_THROW(KeyError, "Cannot drop mandatory position attribute.");
    }

    // insert attributes using the new descriptor

    Descriptor::Ptr newDescriptor = descriptor.duplicateDrop(indices);

    LeafManagerT leafManager(tree);
    tbb::parallel_for(leafManager.leafRange(),
        DropAttributesOp<PointDataTreeT>(indices, newDescriptor));
}


////////////////////////////////////////


template <typename PointDataTreeT>
inline void dropAttributes( PointDataTreeT& tree,
                            const std::vector<Name>& names)
{
    auto iter = tree.cbeginLeaf();

    if (!iter)  return;

    const AttributeSet& attributeSet = iter->attributeSet();
    const AttributeSet::Descriptor& descriptor = attributeSet.descriptor();

    std::vector<size_t> indices;

    for (const Name& name : names) {
        const size_t index = descriptor.find(name);

        // do not attempt to drop an attribute that does not exist
        if (index == AttributeSet::INVALID_POS) {
            OPENVDB_THROW(KeyError,
                "Cannot drop an attribute that does not exist - " << name << ".");
        }

        indices.push_back(index);
    }

    dropAttributes(tree, indices);
}


////////////////////////////////////////


template <typename PointDataTreeT>
inline void dropAttribute(  PointDataTreeT& tree,
                            const size_t& index)
{
    std::vector<size_t> indices{index};
    dropAttributes(tree, indices);
}


template <typename PointDataTreeT>
inline void dropAttribute(  PointDataTreeT& tree,
                            const Name& name)
{
    std::vector<Name> names{name};
    dropAttributes(tree, names);
}


////////////////////////////////////////


template <typename PointDataTreeT>
inline void renameAttributes(   PointDataTreeT& tree,
                                const std::vector<Name>& oldNames,
                                const std::vector<Name>& newNames)
{
    if (oldNames.size() != newNames.size()) {
        OPENVDB_THROW(ValueError, "Mis-matching sizes of name vectors, cannot rename attributes.");
    }

    using Descriptor = AttributeSet::Descriptor;

    auto iter = tree.beginLeaf();

    if (!iter)  return;

    const AttributeSet& attributeSet = iter->attributeSet();
    const Descriptor::Ptr descriptor = attributeSet.descriptorPtr();
    auto newDescriptor = std::make_shared<Descriptor>(*descriptor);

    for (size_t i = 0; i < oldNames.size(); i++) {
        const Name& oldName = oldNames[i];
        if (descriptor->find(oldName) == AttributeSet::INVALID_POS) {
            OPENVDB_THROW(KeyError, "Cannot find requested attribute - " << oldName << ".");
        }

        const Name& newName = newNames[i];
        if (descriptor->find(newName) != AttributeSet::INVALID_POS) {
            OPENVDB_THROW(KeyError,
                "Cannot rename attribute as new name already exists - " << newName << ".");
        }

        const AttributeArray* array = attributeSet.getConst(oldName);
        assert(array);

        if (isGroup(*array)) {
            OPENVDB_THROW(KeyError, "Cannot rename group attribute - " << oldName << ".");
        }

        newDescriptor->rename(oldName, newName);
    }

    for (; iter; ++iter) {
        iter->renameAttributes(*descriptor, newDescriptor);
    }
}


template <typename PointDataTreeT>
inline void renameAttribute(PointDataTreeT& tree,
                            const Name& oldName,
                            const Name& newName)
{
    renameAttributes(tree, {oldName}, {newName});
}


////////////////////////////////////////


template <typename PointDataTreeT>
inline void compactAttributes(PointDataTreeT& tree)
{
    using LeafManagerT = typename tree::LeafManager<PointDataTreeT>;

    using point_attribute_internal::CompactAttributesOp;

    auto iter = tree.beginLeaf();
    if (!iter)  return;

    LeafManagerT leafManager(tree);
    tbb::parallel_for(leafManager.leafRange(), CompactAttributesOp<PointDataTreeT>());
}


////////////////////////////////////////


template <typename PointDataTreeT>
OPENVDB_DEPRECATED inline void bloscCompressAttribute(  PointDataTreeT&,
                                                        const Name&)
{
    // in-memory compression is no longer supported
}


////////////////////////////////////////


} // namespace points
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_POINTS_POINT_ATTRIBUTE_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
