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
/// @file tools/LevelSetFracture.h
///
/// @brief Divide volumes represented by level set grids into multiple,
/// disjoint pieces by intersecting them with one or more "cutter" volumes,
/// also represented by level sets.

#ifndef OPENVDB_TOOLS_LEVELSETFRACTURE_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_LEVELSETFRACTURE_HAS_BEEN_INCLUDED

#include <openvdb/Grid.h>
#include <openvdb/math/Quat.h>
#include <openvdb/util/NullInterrupter.h>

#include "Composite.h" // for csgIntersectionCopy() and csgDifferenceCopy()
#include "GridTransformer.h" // for resampleToMatch()
#include "LevelSetUtil.h" // for sdfSegmentation()

#include <limits>
#include <list>

#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {

/// @brief Level set fracturing
template<class GridType, class InterruptType = util::NullInterrupter>
class LevelSetFracture
{
public:
    typedef std::vector<Vec3s> Vec3sList;
    typedef std::vector<math::Quats> QuatsList;
    typedef std::list<typename GridType::Ptr> GridPtrList;
    typedef typename GridPtrList::iterator GridPtrListIter;


    /// @brief Default constructor
    ///
    /// @param interrupter  optional interrupter object
    explicit LevelSetFracture(InterruptType* interrupter = NULL);

    /// @brief Divide volumes represented by level set grids into multiple,
    /// disjoint pieces by intersecting them with one or more "cutter" volumes,
    /// also represented by level sets.
    /// @details If desired, the process can be applied iteratively, so that
    /// fragments created with one cutter are subdivided by other cutters.
    ///
    /// @note  The incoming @a grids and the @a cutter are required to have matching
    ///        transforms and narrow band widths!
    ///
    /// @param grids          list of grids to fracture. The residuals of the
    ///                       fractured grids will remain in this list
    /// @param cutter         a level set grid to use as the cutter object
    /// @param segment        toggle to split disjoint fragments into their own grids
    /// @param points         optional list of world space points at which to instance the
    ///                       cutter object (if null, use the cutter's current position only)
    /// @param rotations      optional list of custom rotations for each cutter instance
    /// @param cutterOverlap  toggle to allow consecutive cutter instances to fracture
    ///                       previously generated fragments
    void fracture(GridPtrList& grids, const GridType& cutter, bool segment = false,
        const Vec3sList* points = NULL, const QuatsList* rotations = NULL,
        bool cutterOverlap = true);

    /// Return a list of new fragments, not including the residuals from the input grids.
    GridPtrList& fragments() { return mFragments; }

    /// Remove all elements from the fragment list.
    void clear() { mFragments.clear(); }

private:
    // disallow copy by assignment
    void operator=(const LevelSetFracture&) {}

    bool wasInterrupted(int percent = -1) const {
        return mInterrupter && mInterrupter->wasInterrupted(percent);
    }

    bool isValidFragment(GridType&) const;
    void segmentFragments(GridPtrList&) const;
    void process(GridPtrList&, const GridType& cutter);

    InterruptType* mInterrupter;
    GridPtrList mFragments;
};


////////////////////////////////////////


// Internal utility objects and implementation details

namespace level_set_fracture_internal {


template<typename LeafNodeType>
struct FindMinMaxVoxelValue {

    typedef typename LeafNodeType::ValueType    ValueType;

    FindMinMaxVoxelValue(const std::vector<const LeafNodeType*>& nodes)
        : minValue(std::numeric_limits<ValueType>::max())
        , maxValue(-minValue)
        , mNodes(nodes.empty() ? NULL : &nodes.front())
    {
    }

    FindMinMaxVoxelValue(FindMinMaxVoxelValue& rhs, tbb::split)
        : minValue(std::numeric_limits<ValueType>::max())
        , maxValue(-minValue)
        , mNodes(rhs.mNodes)
    {
    }

    void operator()(const tbb::blocked_range<size_t>& range) {
        for (size_t n = range.begin(), N = range.end(); n < N; ++n) {
            const ValueType* data = mNodes[n]->buffer().data();
            for (Index i = 0; i < LeafNodeType::SIZE; ++i) {
                minValue = std::min(minValue, data[i]);
                maxValue = std::max(maxValue, data[i]);
            }
        }
    }

    void join(FindMinMaxVoxelValue& rhs) {
        minValue = std::min(minValue, rhs.minValue);
        maxValue = std::max(maxValue, rhs.maxValue);
    }

    ValueType minValue, maxValue;

    LeafNodeType const * const * const mNodes;
}; // struct FindMinMaxVoxelValue


} // namespace level_set_fracture_internal


////////////////////////////////////////


template<class GridType, class InterruptType>
LevelSetFracture<GridType, InterruptType>::LevelSetFracture(InterruptType* interrupter)
    : mInterrupter(interrupter)
    , mFragments()
{
}


template<class GridType, class InterruptType>
void
LevelSetFracture<GridType, InterruptType>::fracture(GridPtrList& grids, const GridType& cutter,
    bool segmentation, const Vec3sList* points, const QuatsList* rotations, bool cutterOverlap)
{
    // We can process all incoming grids with the same cutter instance,
    // this optimization is enabled by the requirement of having matching
    // transforms between all incoming grids and the cutter object.
    if (points && points->size() != 0) {


        math::Transform::Ptr originalCutterTransform = cutter.transform().copy();
        GridType cutterGrid(*const_cast<GridType*>(&cutter), ShallowCopy());

        const bool hasInstanceRotations =
            points && rotations && points->size() == rotations->size();

        // for each instance point..
        for (size_t p = 0, P = points->size(); p < P; ++p) {
            int percent = int((float(p) / float(P)) * 100.0);
            if (wasInterrupted(percent)) break;

            GridType instCutterGrid;
            instCutterGrid.setTransform(originalCutterTransform->copy());
            math::Transform::Ptr xform = originalCutterTransform->copy();

            if (hasInstanceRotations) {
                const Vec3s& rot = (*rotations)[p].eulerAngles(math::XYZ_ROTATION);
                xform->preRotate(rot[0], math::X_AXIS);
                xform->preRotate(rot[1], math::Y_AXIS);
                xform->preRotate(rot[2], math::Z_AXIS);
                xform->postTranslate((*points)[p]);
            } else {
                xform->postTranslate((*points)[p]);
            }

            cutterGrid.setTransform(xform);

            // Since there is no scaling, use the generic resampler instead of
            // the more expensive level set rebuild tool.
            if (mInterrupter != NULL) {

                if (hasInstanceRotations) {
                    doResampleToMatch<BoxSampler>(cutterGrid, instCutterGrid, *mInterrupter);
                } else {
                    doResampleToMatch<PointSampler>(cutterGrid, instCutterGrid, *mInterrupter);
                }
            } else {
                util::NullInterrupter interrupter;
                if (hasInstanceRotations) {
                    doResampleToMatch<BoxSampler>(cutterGrid, instCutterGrid, interrupter);
                } else {
                    doResampleToMatch<PointSampler>(cutterGrid, instCutterGrid, interrupter);
                }
            }

            if (wasInterrupted(percent)) break;

            if (cutterOverlap && !mFragments.empty()) process(mFragments, instCutterGrid);
            process(grids, instCutterGrid);
        }

    } else {
        // use cutter in place
        if (cutterOverlap && !mFragments.empty()) process(mFragments, cutter);
        process(grids, cutter);
    }

    if (segmentation) {
        segmentFragments(mFragments);
        segmentFragments(grids);
    }
}


template<class GridType, class InterruptType>
bool
LevelSetFracture<GridType, InterruptType>::isValidFragment(GridType& grid) const
{
    typedef typename GridType::TreeType::LeafNodeType LeafNodeType;

    if (grid.tree().leafCount() < 9) {

        std::vector<const LeafNodeType*> nodes;
        grid.tree().getNodes(nodes);

        Index64 activeVoxelCount = 0;

        for (size_t n = 0, N = nodes.size(); n < N; ++n) {
            activeVoxelCount += nodes[n]->onVoxelCount();
        }

        if (activeVoxelCount < 27) return false;

        level_set_fracture_internal::FindMinMaxVoxelValue<LeafNodeType> op(nodes);
        tbb::parallel_reduce(tbb::blocked_range<size_t>(0, nodes.size()), op);

        if ((op.minValue < 0) == (op.maxValue < 0)) return false;
    }

    return true;
}


template<class GridType, class InterruptType>
void
LevelSetFracture<GridType, InterruptType>::segmentFragments(GridPtrList& grids) const
{
    GridPtrList newFragments;

    for (GridPtrListIter it = grids.begin(); it != grids.end(); ++it) {

        std::vector<typename GridType::Ptr> segments;
        segmentSDF(*(*it), segments);

        for (size_t n = 0, N = segments.size(); n < N; ++n) {
            newFragments.push_back(segments[n]);
        }
    }

    grids.swap(newFragments);
}


template<class GridType, class InterruptType>
void
LevelSetFracture<GridType, InterruptType>::process(
    GridPtrList& grids, const GridType& cutter)
{
    typedef typename GridType::Ptr GridPtr;
    GridPtrList newFragments;

    for (GridPtrListIter it = grids.begin(); it != grids.end(); ++it) {

        if (wasInterrupted()) break;

        GridPtr& grid = *it;

        GridPtr fragment = csgIntersectionCopy(*grid, cutter);
        if (!isValidFragment(*fragment)) continue;

        GridPtr residual = csgDifferenceCopy(*grid, cutter);
        if (!isValidFragment(*residual)) continue;

        newFragments.push_back(fragment);

        grid->tree().clear();
        grid->tree().merge(residual->tree());
    }

    if (!newFragments.empty()) {
        mFragments.splice(mFragments.end(), newFragments);
    }
}

} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_LEVELSETFRACTURE_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
