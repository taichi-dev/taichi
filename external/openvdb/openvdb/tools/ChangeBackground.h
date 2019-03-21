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
//
/// @file ChangeBackground.h
///
/// @brief Efficient multi-threaded replacement of the background
/// values in tree.
///
/// @author Ken Museth

#ifndef OPENVDB_TOOLS_ChangeBACKGROUND_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_ChangeBACKGROUND_HAS_BEEN_INCLUDED

#include <openvdb/math/Math.h> // for isNegative and negative
#include <openvdb/Types.h> // for Index typedef
#include <openvdb/tree/NodeManager.h>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {

/// @brief Replace the background value in all the nodes of a tree.
/// @details The sign of the background value is preserved, and only
/// inactive values equal to the old background value are replaced.
///
/// @note If a LeafManager is used the cached leaf nodes are reused,
/// resulting in slightly better overall performance.
///
/// @param tree          Tree (or LeafManager) that will have its background value changed
/// @param background    the new background value
/// @param threaded      enable or disable threading  (threading is enabled by default)
/// @param grainSize     used to control the threading granularity (default is 32)
template<typename TreeOrLeafManagerT>
inline void
changeBackground(
    TreeOrLeafManagerT& tree,
    const typename TreeOrLeafManagerT::ValueType& background,
    bool threaded = true,
    size_t grainSize = 32);


/// @brief Replace the background value in all the nodes of a floating-point tree
/// containing a symmetric narrow-band level set.
/// @details All inactive values will be set to +| @a halfWidth | if outside
/// and -| @a halfWidth | if inside, where @a halfWidth is half the width
/// of the symmetric narrow band.
///
/// @note This method is faster than changeBackground since it does not
/// perform tests to see if inactive values are equal to the old background value.
/// @note If a LeafManager is used the cached leaf nodes are reused,
/// resulting in slightly better overall performance.
///
/// @param tree          Tree (or LeafManager) that will have its background value changed
/// @param halfWidth     half of the width of the symmetric narrow band
/// @param threaded      enable or disable threading  (threading is enabled by default)
/// @param grainSize     used to control the threading granularity (default is 32)
///
/// @throw ValueError if @a halfWidth is negative (as defined by math::isNegative)
template<typename TreeOrLeafManagerT>
inline void
changeLevelSetBackground(
    TreeOrLeafManagerT& tree,
    const typename TreeOrLeafManagerT::ValueType& halfWidth,
    bool threaded = true,
    size_t grainSize = 32);


/// @brief Replace the background values in all the nodes of a floating-point tree
/// containing a possibly asymmetric narrow-band level set.
/// @details All inactive values will be set to +| @a outsideWidth | if outside
/// and -| @a insideWidth | if inside, where @a outsideWidth is the outside
/// width of the narrow band and @a insideWidth is its inside width.
///
/// @note This method is faster than changeBackground since it does not
/// perform tests to see if inactive values are equal to the old background value.
/// @note If a LeafManager is used the cached leaf nodes are reused,
/// resulting in slightly better overall performance.
///
/// @param tree          Tree (or LeafManager) that will have its background value changed
/// @param outsideWidth  The width of the outside of the narrow band
/// @param insideWidth   The width of the inside of the narrow band
/// @param threaded      enable or disable threading  (threading is enabled by default)
/// @param grainSize     used to control the threading granularity (default is 32)
///
/// @throw ValueError if @a outsideWidth is negative or @a insideWidth is
/// not negative (as defined by math::isNegative)
template<typename TreeOrLeafManagerT>
inline void
changeAsymmetricLevelSetBackground(
    TreeOrLeafManagerT& tree,
    const typename TreeOrLeafManagerT::ValueType& outsideWidth,
    const typename TreeOrLeafManagerT::ValueType& insideWidth,
    bool threaded = true,
    size_t grainSize = 32);


//////////////////////////////////////////////////////


// Replaces the background value in a Tree of any type.
template<typename TreeOrLeafManagerT>
class ChangeBackgroundOp
{
public:
    typedef typename TreeOrLeafManagerT::ValueType    ValueT;
    typedef typename TreeOrLeafManagerT::RootNodeType RootT;
    typedef typename TreeOrLeafManagerT::LeafNodeType LeafT;


    ChangeBackgroundOp(const TreeOrLeafManagerT& tree, const ValueT& newValue)
        : mOldValue(tree.root().background())
        , mNewValue(newValue)
    {
    }
    void operator()(RootT& root) const
    {
        for (typename RootT::ValueOffIter it = root.beginValueOff(); it; ++it) this->set(it);
        root.setBackground(mNewValue, false);
    }
    void operator()(LeafT& node) const
    {
        for (typename LeafT::ValueOffIter it = node.beginValueOff(); it; ++it) this->set(it);
    }
    template<typename NodeT>
    void operator()(NodeT& node) const
    {
        typename NodeT::NodeMaskType mask = node.getValueOffMask();
        for (typename NodeT::ValueOnIter it(mask.beginOn(), &node); it; ++it) this->set(it);
    }
private:

    template<typename IterT>
    inline void set(IterT& iter) const
    {
        if (math::isApproxEqual(*iter, mOldValue)) {
            iter.setValue(mNewValue);
        } else if (math::isApproxEqual(*iter, math::negative(mOldValue))) {
            iter.setValue(math::negative(mNewValue));
        }
    }
    const ValueT mOldValue, mNewValue;
};// ChangeBackgroundOp


// Replaces the background value in a Tree assumed to represent a
// level set. It is generally faster than ChangeBackgroundOp.
// Note that is follows the sign-convention that outside is positive
// and inside is negative!
template<typename TreeOrLeafManagerT>
class ChangeLevelSetBackgroundOp
{
public:
    typedef typename TreeOrLeafManagerT::ValueType    ValueT;
    typedef typename TreeOrLeafManagerT::RootNodeType RootT;
    typedef typename TreeOrLeafManagerT::LeafNodeType LeafT;

    /// @brief Constructor for asymmetric narrow-bands
    ChangeLevelSetBackgroundOp(const ValueT& outside, const ValueT& inside)
        : mOutside(outside)
        , mInside(inside)
    {
        if (math::isNegative(mOutside)) {
            OPENVDB_THROW(ValueError,
                          "ChangeLevelSetBackgroundOp: the outside value cannot be negative!");
        }
        if (!math::isNegative(mInside)) {
            OPENVDB_THROW(ValueError,
                          "ChangeLevelSetBackgroundOp: the inside value must be negative!");
        }
    }
    void operator()(RootT& root) const
    {
        for (typename RootT::ValueOffIter it = root.beginValueOff(); it; ++it) this->set(it);
        root.setBackground(mOutside, false);
    }
    void operator()(LeafT& node) const
    {
        for(typename LeafT::ValueOffIter it = node.beginValueOff(); it; ++it) this->set(it);
    }
    template<typename NodeT>
    void operator()(NodeT& node) const
    {
        typedef typename NodeT::ValueOffIter IterT;
        for (IterT it(node.getChildMask().beginOff(), &node); it; ++it) this->set(it);
    }
private:

    template<typename IterT>
    inline void set(IterT& iter) const
    {
        //this is safe since we know ValueType is_floating_point
        ValueT& v = const_cast<ValueT&>(*iter);
        v = v < 0 ? mInside : mOutside;
    }
    const ValueT mOutside, mInside;
};// ChangeLevelSetBackgroundOp


template<typename TreeOrLeafManagerT>
inline void
changeBackground(
    TreeOrLeafManagerT& tree,
    const typename TreeOrLeafManagerT::ValueType& background,
    bool threaded,
    size_t grainSize)
{
    tree::NodeManager<TreeOrLeafManagerT> linearTree(tree);
    ChangeBackgroundOp<TreeOrLeafManagerT> op(tree, background);
    linearTree.foreachTopDown(op, threaded, grainSize);
}


template<typename TreeOrLeafManagerT>
inline void
changeAsymmetricLevelSetBackground(
    TreeOrLeafManagerT& tree,
    const typename TreeOrLeafManagerT::ValueType& outsideValue,
    const typename TreeOrLeafManagerT::ValueType& insideValue,
    bool threaded,
    size_t grainSize)
{
    tree::NodeManager<TreeOrLeafManagerT> linearTree(tree);
    ChangeLevelSetBackgroundOp<TreeOrLeafManagerT> op(outsideValue, insideValue);
    linearTree.foreachTopDown(op, threaded, grainSize);
}


// If the narrow-band is symmetric only one background value is required
template<typename TreeOrLeafManagerT>
inline void
changeLevelSetBackground(
    TreeOrLeafManagerT& tree,
    const typename TreeOrLeafManagerT::ValueType& background,
    bool threaded,
    size_t grainSize)
{
    changeAsymmetricLevelSetBackground(
        tree, background, math::negative(background), threaded, grainSize);
}

} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_CHANGEBACKGROUND_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
