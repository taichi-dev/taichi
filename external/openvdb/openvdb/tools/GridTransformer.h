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

/// @file GridTransformer.h
/// @author Peter Cucka

#ifndef OPENVDB_TOOLS_GRIDTRANSFORMER_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_GRIDTRANSFORMER_HAS_BEEN_INCLUDED

#include <openvdb/Grid.h>
#include <openvdb/Types.h>
#include <openvdb/math/Math.h> // for isApproxEqual()
#include <openvdb/util/NullInterrupter.h>
#include "ChangeBackground.h"
#include "Interpolation.h"
#include "LevelSetRebuild.h" // for doLevelSetRebuild()
#include "SignedFloodFill.h" // for signedFloodFill
#include "Prune.h" // for pruneLevelSet
#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>
#include <cmath>
#include <functional>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {

/// @brief Resample an input grid into an output grid of the same type such that,
/// after resampling, the input and output grids coincide (apart from sampling
/// artifacts), but the output grid's transform is unchanged.
/// @details Specifically, this function resamples the input grid into the output
/// grid's index space, using a sampling kernel like PointSampler, BoxSampler,
/// or QuadraticSampler.
/// @param inGrid       the grid to be resampled
/// @param outGrid      the grid into which to write the resampled voxel data
/// @param interrupter  an object adhering to the util::NullInterrupter interface
/// @par Example:
/// @code
/// // Create an input grid with the default identity transform
/// // and populate it with a level-set sphere.
/// FloatGrid::ConstPtr src = tools::makeSphere(...);
/// // Create an output grid and give it a uniform-scale transform.
/// FloatGrid::Ptr dest = FloatGrid::create();
/// const float voxelSize = 0.5;
/// dest->setTransform(math::Transform::createLinearTransform(voxelSize));
/// // Resample the input grid into the output grid, reproducing
/// // the level-set sphere at a smaller voxel size.
/// MyInterrupter interrupter = ...;
/// tools::resampleToMatch<tools::QuadraticSampler>(*src, *dest, interrupter);
/// @endcode
template<typename Sampler, typename Interrupter, typename GridType>
inline void
resampleToMatch(const GridType& inGrid, GridType& outGrid, Interrupter& interrupter);

/// @brief Resample an input grid into an output grid of the same type such that,
/// after resampling, the input and output grids coincide (apart from sampling
/// artifacts), but the output grid's transform is unchanged.
/// @details Specifically, this function resamples the input grid into the output
/// grid's index space, using a sampling kernel like PointSampler, BoxSampler,
/// or QuadraticSampler.
/// @param inGrid       the grid to be resampled
/// @param outGrid      the grid into which to write the resampled voxel data
/// @par Example:
/// @code
/// // Create an input grid with the default identity transform
/// // and populate it with a level-set sphere.
/// FloatGrid::ConstPtr src = tools::makeSphere(...);
/// // Create an output grid and give it a uniform-scale transform.
/// FloatGrid::Ptr dest = FloatGrid::create();
/// const float voxelSize = 0.5;
/// dest->setTransform(math::Transform::createLinearTransform(voxelSize));
/// // Resample the input grid into the output grid, reproducing
/// // the level-set sphere at a smaller voxel size.
/// tools::resampleToMatch<tools::QuadraticSampler>(*src, *dest);
/// @endcode
template<typename Sampler, typename GridType>
inline void
resampleToMatch(const GridType& inGrid, GridType& outGrid);


////////////////////////////////////////


namespace internal {

/// @brief A TileSampler wraps a grid sampler of another type (BoxSampler,
/// QuadraticSampler, etc.), and for samples that fall within a given tile
/// of the grid, it returns a cached tile value instead of accessing the grid.
template<typename Sampler, typename TreeT>
class TileSampler: public Sampler
{
public:
    using ValueT = typename TreeT::ValueType;

    /// @param b        the index-space bounding box of a particular grid tile
    /// @param tileVal  the tile's value
    /// @param on       the tile's active state
    TileSampler(const CoordBBox& b, const ValueT& tileVal, bool on):
        mBBox(b.min().asVec3d(), b.max().asVec3d()), mVal(tileVal), mActive(on), mEmpty(false)
    {
        mBBox.expand(-this->radius()); // shrink the bounding box by the sample radius
        mEmpty = mBBox.empty();
    }

    bool sample(const TreeT& inTree, const Vec3R& inCoord, ValueT& result) const
    {
        if (!mEmpty && mBBox.isInside(inCoord)) { result = mVal; return mActive; }
        return Sampler::sample(inTree, inCoord, result);
    }

protected:
    BBoxd mBBox;
    ValueT mVal;
    bool mActive, mEmpty;
};


/// @brief For point sampling, tree traversal is less expensive than testing
/// bounding box membership.
template<typename TreeT>
class TileSampler<PointSampler, TreeT>: public PointSampler {
public:
    TileSampler(const CoordBBox&, const typename TreeT::ValueType&, bool) {}
};

/// @brief For point sampling, tree traversal is less expensive than testing
/// bounding box membership.
template<typename TreeT>
class TileSampler<StaggeredPointSampler, TreeT>: public StaggeredPointSampler {
public:
    TileSampler(const CoordBBox&, const typename TreeT::ValueType&, bool) {}
};

} // namespace internal


////////////////////////////////////////


/// A GridResampler applies a geometric transformation to an
/// input grid using one of several sampling schemes, and stores
/// the result in an output grid.
///
/// Usage:
/// @code
/// GridResampler resampler();
/// resampler.transformGrid<BoxSampler>(xform, inGrid, outGrid);
/// @endcode
/// where @c xform is a functor that implements the following methods:
/// @code
/// bool isAffine() const
/// openvdb::Vec3d transform(const openvdb::Vec3d&) const
/// openvdb::Vec3d invTransform(const openvdb::Vec3d&) const
/// @endcode
/// @note When the transform is affine and can be expressed as a 4 x 4 matrix,
/// a GridTransformer is much more efficient than a GridResampler.
class GridResampler
{
public:
    using Ptr = SharedPtr<GridResampler>;
    using InterruptFunc = std::function<bool (void)>;

    GridResampler(): mThreaded(true), mTransformTiles(true) {}
    virtual ~GridResampler() {}

    GridResampler(const GridResampler&) = default;
    GridResampler& operator=(const GridResampler&) = default;

    /// Enable or disable threading.  (Threading is enabled by default.)
    void setThreaded(bool b) { mThreaded = b; }
    /// Return @c true if threading is enabled.
    bool threaded() const { return mThreaded; }
    /// Enable or disable processing of tiles.  (Enabled by default, except for level set grids.)
    void setTransformTiles(bool b) { mTransformTiles = b; }
    /// Return @c true if tile processing is enabled.
    bool transformTiles() const { return mTransformTiles; }

    /// @brief Allow processing to be aborted by providing an interrupter object.
    /// The interrupter will be queried periodically during processing.
    /// @see util/NullInterrupter.h for interrupter interface requirements.
    template<typename InterrupterType> void setInterrupter(InterrupterType&);

    template<typename Sampler, typename GridT, typename Transformer>
    void transformGrid(const Transformer&,
        const GridT& inGrid, GridT& outGrid) const;

protected:
    template<typename Sampler, typename GridT, typename Transformer>
    void applyTransform(const Transformer&, const GridT& inGrid, GridT& outGrid) const;

    bool interrupt() const { return mInterrupt && mInterrupt(); }

private:
    template<typename Sampler, typename InTreeT, typename OutTreeT, typename Transformer>
    static void transformBBox(const Transformer&, const CoordBBox& inBBox,
        const InTreeT& inTree, OutTreeT& outTree, const InterruptFunc&,
        const Sampler& = Sampler());

    template<typename Sampler, typename TreeT, typename Transformer>
    class RangeProcessor;

    bool mThreaded, mTransformTiles;
    InterruptFunc mInterrupt;
};


////////////////////////////////////////


/// @brief A GridTransformer applies a geometric transformation to an
/// input grid using one of several sampling schemes, and stores
/// the result in an output grid.
///
/// @note GridTransformer is optimized for affine transformations.
///
/// Usage:
/// @code
/// Mat4R xform = ...;
/// GridTransformer transformer(xform);
/// transformer.transformGrid<BoxSampler>(inGrid, outGrid);
/// @endcode
/// or
/// @code
/// Vec3R pivot = ..., scale = ..., rotate = ..., translate = ...;
/// GridTransformer transformer(pivot, scale, rotate, translate);
/// transformer.transformGrid<QuadraticSampler>(inGrid, outGrid);
/// @endcode
class GridTransformer: public GridResampler
{
public:
    using Ptr = SharedPtr<GridTransformer>;

    GridTransformer(const Mat4R& xform);
    GridTransformer(
        const Vec3R& pivot,
        const Vec3R& scale,
        const Vec3R& rotate,
        const Vec3R& translate,
        const std::string& xformOrder = "tsr",
        const std::string& rotationOrder = "zyx");
    ~GridTransformer() override = default;

    GridTransformer(const GridTransformer&) = default;
    GridTransformer& operator=(const GridTransformer&) = default;

    const Mat4R& getTransform() const { return mTransform; }

    template<class Sampler, class GridT>
    void transformGrid(const GridT& inGrid, GridT& outGrid) const;

private:
    struct MatrixTransform;

    inline void init(const Vec3R& pivot, const Vec3R& scale,
        const Vec3R& rotate, const Vec3R& translate,
        const std::string& xformOrder, const std::string& rotOrder);

    Vec3R mPivot;
    Vec3i mMipLevels;
    Mat4R mTransform, mPreScaleTransform, mPostScaleTransform;
};


////////////////////////////////////////


namespace local_util {

/// @brief Decompose an affine transform into scale, rotation and translation components.
/// @return @c false if the given matrix is not affine or cannot otherwise be decomposed.
template<typename T>
inline bool
decompose(const math::Mat4<T>& m, math::Vec3<T>& scale,
    math::Vec3<T>& rotate, math::Vec3<T>& translate)
{
    if (!math::isAffine(m)) return false;

    // This is the translation in world space
    translate = m.getTranslation();
    // Extract translation.
    const math::Mat3<T> xform = m.getMat3();

    const math::Vec3<T> unsignedScale(
        (math::Vec3<T>(1, 0, 0) * xform).length(),
        (math::Vec3<T>(0, 1, 0) * xform).length(),
        (math::Vec3<T>(0, 0, 1) * xform).length());

    const bool hasUniformScale = unsignedScale.eq(math::Vec3<T>(unsignedScale[0]));

    bool hasRotation = false;
    bool validDecomposition = false;

    T minAngle = std::numeric_limits<T>::max();

    // If the transformation matrix contains a reflection,
    // test different negative scales to find a decomposition
    // that favors the optimal resampling algorithm.
    for (size_t n = 0; n < 8; ++n) {

        const math::Vec3<T> signedScale(
            n & 0x1 ? -unsignedScale.x() : unsignedScale.x(),
            n & 0x2 ? -unsignedScale.y() : unsignedScale.y(),
            n & 0x4 ? -unsignedScale.z() : unsignedScale.z());

        // Extract scale and potentially reflection.
        const math::Mat3<T> mat = xform * math::scale<math::Mat3<T> >(signedScale).inverse();
        if (mat.det() < T(0.0)) continue; // Skip if mat contains a reflection.

        const math::Vec3<T> tmpAngle = math::eulerAngles(mat, math::XYZ_ROTATION);

        const math::Mat3<T> rebuild =
            math::rotation<math::Mat3<T> >(math::Vec3<T>(1, 0, 0), tmpAngle.x()) *
            math::rotation<math::Mat3<T> >(math::Vec3<T>(0, 1, 0), tmpAngle.y()) *
            math::rotation<math::Mat3<T> >(math::Vec3<T>(0, 0, 1), tmpAngle.z()) *
            math::scale<math::Mat3<T> >(signedScale);

        if (xform.eq(rebuild)) {

            const T maxAngle = std::max(std::abs(tmpAngle[0]),
                std::max(std::abs(tmpAngle[1]), std::abs(tmpAngle[2])));

            if (!(minAngle < maxAngle)) { // Update if less or equal.

                minAngle = maxAngle;
                rotate = tmpAngle;
                scale = signedScale;

                hasRotation = !rotate.eq(math::Vec3<T>::zero());
                validDecomposition = true;

                if (hasUniformScale || !hasRotation) {
                    // Current decomposition is optimal.
                    break;
                }
            }
        }
    }

    if (!validDecomposition || (hasRotation && !hasUniformScale)) {
        // The decomposition is invalid if the transformation matrix contains shear.
        // No unique decomposition if scale is nonuniform and rotation is nonzero.
        return false;
    }

    return true;
}

} // namespace local_util


////////////////////////////////////////


/// This class implements the Transformer functor interface (specifically,
/// the isAffine(), transform() and invTransform() methods) for a transform
/// that is expressed as a 4 x 4 matrix.
struct GridTransformer::MatrixTransform
{
    MatrixTransform(): mat(Mat4R::identity()), invMat(Mat4R::identity()) {}
    MatrixTransform(const Mat4R& xform): mat(xform), invMat(xform.inverse()) {}

    bool isAffine() const { return math::isAffine(mat); }

    Vec3R transform(const Vec3R& pos) const { return mat.transformH(pos); }

    Vec3R invTransform(const Vec3R& pos) const { return invMat.transformH(pos); }

    Mat4R mat, invMat;
};


////////////////////////////////////////


/// @brief This class implements the Transformer functor interface (specifically,
/// the isAffine(), transform() and invTransform() methods) for a transform
/// that maps an A grid into a B grid's index space such that, after resampling,
/// A's index space and transform match B's index space and transform.
class ABTransform
{
public:
    /// @param aXform  the A grid's transform
    /// @param bXform  the B grid's transform
    ABTransform(const math::Transform& aXform, const math::Transform& bXform):
        mAXform(aXform),
        mBXform(bXform),
        mIsAffine(mAXform.isLinear() && mBXform.isLinear()),
        mIsIdentity(mIsAffine && mAXform == mBXform)
        {}

    bool isAffine() const { return mIsAffine; }

    bool isIdentity() const { return mIsIdentity; }

    openvdb::Vec3R transform(const openvdb::Vec3R& pos) const
    {
        return mBXform.worldToIndex(mAXform.indexToWorld(pos));
    }

    openvdb::Vec3R invTransform(const openvdb::Vec3R& pos) const
    {
        return mAXform.worldToIndex(mBXform.indexToWorld(pos));
    }

    const math::Transform& getA() const { return mAXform; }
    const math::Transform& getB() const { return mBXform; }

private:
    const math::Transform &mAXform, &mBXform;
    const bool mIsAffine;
    const bool mIsIdentity;
};


/// The normal entry points for resampling are the resampleToMatch() functions,
/// which correctly handle level set grids under scaling and shearing.
/// doResampleToMatch() is mainly for internal use but is typically faster
/// for level sets, and correct provided that no scaling or shearing is needed.
///
/// @warning Do not use this function to scale or shear a level set grid.
template<typename Sampler, typename Interrupter, typename GridType>
inline void
doResampleToMatch(const GridType& inGrid, GridType& outGrid, Interrupter& interrupter)
{
    ABTransform xform(inGrid.transform(), outGrid.transform());

    if (Sampler::consistent() && xform.isIdentity()) {
        // If the transforms of the input and output are identical, the
        // output tree is simply a deep copy of the input tree.
        outGrid.setTree(inGrid.tree().copy());
    } else if (xform.isAffine()) {
        // If the input and output transforms are both affine, create an
        // input to output transform (in:index-to-world * out:world-to-index)
        // and use the fast GridTransformer API.
        Mat4R mat = xform.getA().baseMap()->getAffineMap()->getMat4() *
            ( xform.getB().baseMap()->getAffineMap()->getMat4().inverse() );

        GridTransformer transformer(mat);
        transformer.setInterrupter(interrupter);

        // Transform the input grid and store the result in the output grid.
        transformer.transformGrid<Sampler>(inGrid, outGrid);
    } else {
        // If either the input or the output transform is non-affine,
        // use the slower GridResampler API.
        GridResampler resampler;
        resampler.setInterrupter(interrupter);

        resampler.transformGrid<Sampler>(xform, inGrid, outGrid);
    }
}


template<typename Sampler, typename Interrupter, typename GridType>
inline void
resampleToMatch(const GridType& inGrid, GridType& outGrid, Interrupter& interrupter)
{
    if (inGrid.getGridClass() == GRID_LEVEL_SET) {
        // If the input grid is a level set, resample it using the level set rebuild tool.

        if (inGrid.constTransform() == outGrid.constTransform()) {
            // If the transforms of the input and output grids are identical,
            // the output tree is simply a deep copy of the input tree.
            outGrid.setTree(inGrid.tree().copy());
            return;
        }

        // If the output grid is a level set, resample the input grid to have the output grid's
        // background value.  Otherwise, preserve the input grid's background value.
        using ValueT = typename GridType::ValueType;
        const ValueT halfWidth = ((outGrid.getGridClass() == openvdb::GRID_LEVEL_SET)
            ? ValueT(outGrid.background() * (1.0 / outGrid.voxelSize()[0]))
            : ValueT(inGrid.background() * (1.0 / inGrid.voxelSize()[0])));

        typename GridType::Ptr tempGrid;
        try {
            tempGrid = doLevelSetRebuild(inGrid, /*iso=*/zeroVal<ValueT>(),
                /*exWidth=*/halfWidth, /*inWidth=*/halfWidth,
                &outGrid.constTransform(), &interrupter);
        } catch (TypeError&) {
            // The input grid is classified as a level set, but it has a value type
            // that is not supported by the level set rebuild tool.  Fall back to
            // using the generic resampler.
            tempGrid.reset();
        }
        if (tempGrid) {
            outGrid.setTree(tempGrid->treePtr());
            return;
        }
    }

    // If the input grid is not a level set, use the generic resampler.
    doResampleToMatch<Sampler>(inGrid, outGrid, interrupter);
}


template<typename Sampler, typename GridType>
inline void
resampleToMatch(const GridType& inGrid, GridType& outGrid)
{
    util::NullInterrupter interrupter;
    resampleToMatch<Sampler>(inGrid, outGrid, interrupter);
}


////////////////////////////////////////


inline
GridTransformer::GridTransformer(const Mat4R& xform):
    mPivot(0, 0, 0),
    mMipLevels(0, 0, 0),
    mTransform(xform),
    mPreScaleTransform(Mat4R::identity()),
    mPostScaleTransform(Mat4R::identity())
{
    Vec3R scale, rotate, translate;
    if (local_util::decompose(mTransform, scale, rotate, translate)) {
        // If the transform can be decomposed into affine components,
        // use them to set up a mipmapping-like scheme for downsampling.
        init(mPivot, scale, rotate, translate, "srt", "zyx");
    }
}


inline
GridTransformer::GridTransformer(
    const Vec3R& pivot, const Vec3R& scale,
    const Vec3R& rotate, const Vec3R& translate,
    const std::string& xformOrder, const std::string& rotOrder):
    mPivot(0, 0, 0),
    mMipLevels(0, 0, 0),
    mPreScaleTransform(Mat4R::identity()),
    mPostScaleTransform(Mat4R::identity())
{
    init(pivot, scale, rotate, translate, xformOrder, rotOrder);
}


////////////////////////////////////////


inline void
GridTransformer::init(
    const Vec3R& pivot, const Vec3R& scale,
    const Vec3R& rotate, const Vec3R& translate,
    const std::string& xformOrder, const std::string& rotOrder)
{
    if (xformOrder.size() != 3) {
        OPENVDB_THROW(ValueError, "invalid transform order (" + xformOrder + ")");
    }
    if (rotOrder.size() != 3) {
        OPENVDB_THROW(ValueError, "invalid rotation order (" + rotOrder + ")");
    }

    mPivot = pivot;

    // Scaling is handled via a mipmapping-like scheme of successive
    // halvings of the tree resolution, until the remaining scale
    // factor is greater than or equal to 1/2.
    Vec3R scaleRemainder = scale;
    for (int i = 0; i < 3; ++i) {
        double s = std::fabs(scale(i));
        if (s < 0.5) {
            mMipLevels(i) = int(std::floor(-std::log(s)/std::log(2.0)));
            scaleRemainder(i) = scale(i) * (1 << mMipLevels(i));
        }
    }

    // Build pre-scale and post-scale transform matrices based on
    // the user-specified order of operations.
    // Note that we iterate over the transform order string in reverse order
    // (e.g., "t", "r", "s", given "srt").  This is because math::Mat matrices
    // postmultiply row vectors rather than premultiplying column vectors.
    mTransform = mPreScaleTransform = mPostScaleTransform = Mat4R::identity();
    Mat4R* remainder = &mPostScaleTransform;
    int rpos, spos, tpos;
    rpos = spos = tpos = 3;
    for (int ix = 2; ix >= 0; --ix) { // reverse iteration
        switch (xformOrder[ix]) {

        case 'r':
            rpos = ix;
            mTransform.preTranslate(pivot);
            remainder->preTranslate(pivot);

            int xpos, ypos, zpos;
            xpos = ypos = zpos = 3;
            for (int ir = 2; ir >= 0; --ir) {
                switch (rotOrder[ir]) {
                case 'x':
                    xpos = ir;
                    mTransform.preRotate(math::X_AXIS, rotate.x());
                    remainder->preRotate(math::X_AXIS, rotate.x());
                    break;
                case 'y':
                    ypos = ir;
                    mTransform.preRotate(math::Y_AXIS, rotate.y());
                    remainder->preRotate(math::Y_AXIS, rotate.y());
                    break;
                case 'z':
                    zpos = ir;
                    mTransform.preRotate(math::Z_AXIS, rotate.z());
                    remainder->preRotate(math::Z_AXIS, rotate.z());
                    break;
                }
            }
            // Reject rotation order strings that don't contain exactly one
            // instance of "x", "y" and "z".
            if (xpos > 2 || ypos > 2 || zpos > 2) {
                OPENVDB_THROW(ValueError, "invalid rotation order (" + rotOrder + ")");
            }

            mTransform.preTranslate(-pivot);
            remainder->preTranslate(-pivot);
            break;

        case 's':
            spos = ix;
            mTransform.preTranslate(pivot);
            mTransform.preScale(scale);
            mTransform.preTranslate(-pivot);

            remainder->preTranslate(pivot);
            remainder->preScale(scaleRemainder);
            remainder->preTranslate(-pivot);
            remainder = &mPreScaleTransform;
            break;

        case 't':
            tpos = ix;
            mTransform.preTranslate(translate);
            remainder->preTranslate(translate);
            break;
        }
    }
    // Reject transform order strings that don't contain exactly one
    // instance of "t", "r" and "s".
    if (tpos > 2 || rpos > 2 || spos > 2) {
        OPENVDB_THROW(ValueError, "invalid transform order (" + xformOrder + ")");
    }
}


////////////////////////////////////////


template<typename InterrupterType>
void
GridResampler::setInterrupter(InterrupterType& interrupter)
{
    mInterrupt = std::bind(&InterrupterType::wasInterrupted,
        /*this=*/&interrupter, /*percent=*/-1);
}


template<typename Sampler, typename GridT, typename Transformer>
void
GridResampler::transformGrid(const Transformer& xform,
    const GridT& inGrid, GridT& outGrid) const
{
    tools::changeBackground(outGrid.tree(), inGrid.background());
    applyTransform<Sampler>(xform, inGrid, outGrid);
}


template<class Sampler, class GridT>
void
GridTransformer::transformGrid(const GridT& inGrid, GridT& outGrid) const
{
    tools::changeBackground(outGrid.tree(), inGrid.background());

    if (!Sampler::mipmap() || mMipLevels == Vec3i::zero()) {
        // Skip the mipmapping step.
        const MatrixTransform xform(mTransform);
        applyTransform<Sampler>(xform, inGrid, outGrid);

    } else {
        bool firstPass = true;
        const typename GridT::ValueType background = inGrid.background();
        typename GridT::Ptr tempGrid = GridT::create(background);

        if (!mPreScaleTransform.eq(Mat4R::identity())) {
            firstPass = false;
            // Apply the pre-scale transform to the input grid
            // and store the result in a temporary grid.
            const MatrixTransform xform(mPreScaleTransform);
            applyTransform<Sampler>(xform, inGrid, *tempGrid);
        }

        // While the scale factor along one or more axes is less than 1/2,
        // scale the grid by half along those axes.
        Vec3i count = mMipLevels; // # of halvings remaining per axis
        while (count != Vec3i::zero()) {
            MatrixTransform xform;
            xform.mat.setTranslation(mPivot);
            xform.mat.preScale(Vec3R(
                count.x() ? .5 : 1, count.y() ? .5 : 1, count.z() ? .5 : 1));
            xform.mat.preTranslate(-mPivot);
            xform.invMat = xform.mat.inverse();

            if (firstPass) {
                firstPass = false;
                // Scale the input grid and store the result in a temporary grid.
                applyTransform<Sampler>(xform, inGrid, *tempGrid);
            } else {
                // Scale the temporary grid and store the result in a transient grid,
                // then swap the two and discard the transient grid.
                typename GridT::Ptr destGrid = GridT::create(background);
                applyTransform<Sampler>(xform, *tempGrid, *destGrid);
                tempGrid.swap(destGrid);
            }
            // (3, 2, 1) -> (2, 1, 0) -> (1, 0, 0) -> (0, 0, 0), etc.
            count = math::maxComponent(count - 1, Vec3i::zero());
        }

        // Apply the post-scale transform and store the result in the output grid.
        if (!mPostScaleTransform.eq(Mat4R::identity())) {
            const MatrixTransform xform(mPostScaleTransform);
            applyTransform<Sampler>(xform, *tempGrid, outGrid);
        } else {
            outGrid.setTree(tempGrid->treePtr());
        }
    }
}


////////////////////////////////////////


template<class Sampler, class TreeT, typename Transformer>
class GridResampler::RangeProcessor
{
public:
    using LeafIterT = typename TreeT::LeafCIter;
    using TileIterT = typename TreeT::ValueAllCIter;
    using LeafRange = typename tree::IteratorRange<LeafIterT>;
    using TileRange = typename tree::IteratorRange<TileIterT>;
    using InTreeAccessor = typename tree::ValueAccessor<const TreeT>;
    using OutTreeAccessor = typename tree::ValueAccessor<TreeT>;

    RangeProcessor(const Transformer& xform, const CoordBBox& b, const TreeT& inT, TreeT& outT):
        mIsRoot(true), mXform(xform), mBBox(b),
        mInTree(inT), mOutTree(&outT), mInAcc(mInTree), mOutAcc(*mOutTree)
    {}

    RangeProcessor(const Transformer& xform, const CoordBBox& b, const TreeT& inTree):
        mIsRoot(false), mXform(xform), mBBox(b),
        mInTree(inTree), mOutTree(new TreeT(inTree.background())),
        mInAcc(mInTree), mOutAcc(*mOutTree)
    {}

    ~RangeProcessor() { if (!mIsRoot) delete mOutTree; }

    /// Splitting constructor: don't copy the original processor's output tree
    RangeProcessor(RangeProcessor& other, tbb::split):
        mIsRoot(false),
        mXform(other.mXform),
        mBBox(other.mBBox),
        mInTree(other.mInTree),
        mOutTree(new TreeT(mInTree.background())),
        mInAcc(mInTree),
        mOutAcc(*mOutTree),
        mInterrupt(other.mInterrupt)
    {}

    void setInterrupt(const InterruptFunc& f) { mInterrupt = f; }

    /// Transform each leaf node in the given range.
    void operator()(LeafRange& r)
    {
        for ( ; r; ++r) {
            if (interrupt()) break;
            LeafIterT i = r.iterator();
            CoordBBox bbox(i->origin(), i->origin() + Coord(i->dim()));
            if (!mBBox.empty()) {
                // Intersect the leaf node's bounding box with mBBox.
                bbox = CoordBBox(
                    Coord::maxComponent(bbox.min(), mBBox.min()),
                    Coord::minComponent(bbox.max(), mBBox.max()));
            }
            if (!bbox.empty()) {
                transformBBox<Sampler>(mXform, bbox, mInAcc, mOutAcc, mInterrupt);
            }
        }
    }

    /// Transform each non-background tile in the given range.
    void operator()(TileRange& r)
    {
        for ( ; r; ++r) {
            if (interrupt()) break;

            TileIterT i = r.iterator();
            // Skip voxels and background tiles.
            if (!i.isTileValue()) continue;
            if (!i.isValueOn() && math::isApproxEqual(*i, mOutTree->background())) continue;

            CoordBBox bbox;
            i.getBoundingBox(bbox);
            if (!mBBox.empty()) {
                // Intersect the tile's bounding box with mBBox.
                bbox = CoordBBox(
                    Coord::maxComponent(bbox.min(), mBBox.min()),
                    Coord::minComponent(bbox.max(), mBBox.max()));
            }
            if (!bbox.empty()) {
                /// @todo This samples the tile voxel-by-voxel, which is much too slow.
                /// Instead, compute the largest axis-aligned bounding box that is
                /// contained in the transformed tile (adjusted for the sampler radius)
                /// and fill it with the tile value.  Then transform the remaining voxels.
                internal::TileSampler<Sampler, InTreeAccessor>
                    sampler(bbox, i.getValue(), i.isValueOn());
                transformBBox(mXform, bbox, mInAcc, mOutAcc, mInterrupt, sampler);
            }
        }
    }

    /// Merge another processor's output tree into this processor's tree.
    void join(RangeProcessor& other)
    {
        if (!interrupt()) mOutTree->merge(*other.mOutTree);
    }

private:
    bool interrupt() const { return mInterrupt && mInterrupt(); }

    const bool mIsRoot; // true if mOutTree is the top-level tree
    Transformer mXform;
    CoordBBox mBBox;
    const TreeT& mInTree;
    TreeT* mOutTree;
    InTreeAccessor mInAcc;
    OutTreeAccessor mOutAcc;
    InterruptFunc mInterrupt;
};


////////////////////////////////////////


template<class Sampler, class GridT, typename Transformer>
void
GridResampler::applyTransform(const Transformer& xform,
    const GridT& inGrid, GridT& outGrid) const
{
    using TreeT = typename GridT::TreeType;
    const TreeT& inTree = inGrid.tree();
    TreeT& outTree = outGrid.tree();

    using RangeProc = RangeProcessor<Sampler, TreeT, Transformer>;

    const GridClass gridClass = inGrid.getGridClass();

    if (gridClass != GRID_LEVEL_SET && mTransformTiles) {
        // Independently transform the tiles of the input grid.
        // Note: Tiles in level sets can only be background tiles, and they
        // are handled more efficiently with a signed flood fill (see below).

        RangeProc proc(xform, CoordBBox(), inTree, outTree);
        proc.setInterrupt(mInterrupt);

        typename RangeProc::TileIterT tileIter = inTree.cbeginValueAll();
        tileIter.setMaxDepth(tileIter.getLeafDepth() - 1); // skip leaf nodes
        typename RangeProc::TileRange tileRange(tileIter);

        if (mThreaded) {
            tbb::parallel_reduce(tileRange, proc);
        } else {
            proc(tileRange);
        }
    }

    CoordBBox clipBBox;
    if (gridClass == GRID_LEVEL_SET) {
        // Inactive voxels in level sets can only be background voxels, and they
        // are handled more efficiently with a signed flood fill (see below).
        clipBBox = inGrid.evalActiveVoxelBoundingBox();
    }

    // Independently transform the leaf nodes of the input grid.

    RangeProc proc(xform, clipBBox, inTree, outTree);
    proc.setInterrupt(mInterrupt);

    typename RangeProc::LeafRange leafRange(inTree.cbeginLeaf());

    if (mThreaded) {
        tbb::parallel_reduce(leafRange, proc);
    } else {
        proc(leafRange);
    }

    // If the grid is a level set, mark inactive voxels as inside or outside.
    if (gridClass == GRID_LEVEL_SET) {
        tools::pruneLevelSet(outTree);
        tools::signedFloodFill(outTree);
    }
}


////////////////////////////////////////


//static
template<class Sampler, class InTreeT, class OutTreeT, class Transformer>
void
GridResampler::transformBBox(
    const Transformer& xform,
    const CoordBBox& bbox,
    const InTreeT& inTree,
    OutTreeT& outTree,
    const InterruptFunc& interrupt,
    const Sampler& sampler)
{
    using ValueT = typename OutTreeT::ValueType;

    // Transform the corners of the input tree's bounding box
    // and compute the enclosing bounding box in the output tree.
    Vec3R
        inRMin(bbox.min().x(), bbox.min().y(), bbox.min().z()),
        inRMax(bbox.max().x()+1, bbox.max().y()+1, bbox.max().z()+1),
        outRMin = math::minComponent(xform.transform(inRMin), xform.transform(inRMax)),
        outRMax = math::maxComponent(xform.transform(inRMin), xform.transform(inRMax));
    for (int i = 0; i < 8; ++i) {
        Vec3R corner(
            i & 1 ? inRMax.x() : inRMin.x(),
            i & 2 ? inRMax.y() : inRMin.y(),
            i & 4 ? inRMax.z() : inRMin.z());
        outRMin = math::minComponent(outRMin, xform.transform(corner));
        outRMax = math::maxComponent(outRMax, xform.transform(corner));
    }
    Vec3i
        outMin = local_util::floorVec3(outRMin) - Sampler::radius(),
        outMax = local_util::ceilVec3(outRMax) + Sampler::radius();

    if (!xform.isAffine()) {
        // If the transform is not affine, back-project each output voxel
        // into the input tree.
        Vec3R xyz, inXYZ;
        Coord outXYZ;
        int &x = outXYZ.x(), &y = outXYZ.y(), &z = outXYZ.z();
        for (x = outMin.x(); x <= outMax.x(); ++x) {
            if (interrupt && interrupt()) break;
            xyz.x() = x;
            for (y = outMin.y(); y <= outMax.y(); ++y) {
                if (interrupt && interrupt()) break;
                xyz.y() = y;
                for (z = outMin.z(); z <= outMax.z(); ++z) {
                    xyz.z() = z;
                    inXYZ = xform.invTransform(xyz);
                    ValueT result;
                    if (sampler.sample(inTree, inXYZ, result)) {
                        outTree.setValueOn(outXYZ, result);
                    } else {
                        // Note: Don't overwrite existing active values with inactive values.
                        if (!outTree.isValueOn(outXYZ)) {
                            outTree.setValueOff(outXYZ, result);
                        }
                    }
                }
            }
        }
    } else { // affine
        // Compute step sizes in the input tree that correspond to
        // unit steps in x, y and z in the output tree.
        const Vec3R
            translation = xform.invTransform(Vec3R(0, 0, 0)),
            deltaX = xform.invTransform(Vec3R(1, 0, 0)) - translation,
            deltaY = xform.invTransform(Vec3R(0, 1, 0)) - translation,
            deltaZ = xform.invTransform(Vec3R(0, 0, 1)) - translation;

#if defined(__ICC)
        /// @todo The following line is a workaround for bad code generation
        /// in opt-icc11.1_64 (but not debug or gcc) builds.  It should be
        /// removed once the problem has been addressed at its source.
        const Vec3R dummy = deltaX;
#endif

        // Step by whole voxels through the output tree, sampling the
        // corresponding fractional voxels of the input tree.
        Vec3R inStartX = xform.invTransform(Vec3R(outMin));
        Coord outXYZ;
        int &x = outXYZ.x(), &y = outXYZ.y(), &z = outXYZ.z();
        for (x = outMin.x(); x <= outMax.x(); ++x, inStartX += deltaX) {
            if (interrupt && interrupt()) break;
            Vec3R inStartY = inStartX;
            for (y = outMin.y(); y <= outMax.y(); ++y, inStartY += deltaY) {
                if (interrupt && interrupt()) break;
                Vec3R inXYZ = inStartY;
                for (z = outMin.z(); z <= outMax.z(); ++z, inXYZ += deltaZ) {
                    ValueT result;
                    if (sampler.sample(inTree, inXYZ, result)) {
                        outTree.setValueOn(outXYZ, result);
                    } else {
                        // Note: Don't overwrite existing active values with inactive values.
                        if (!outTree.isValueOn(outXYZ)) {
                            outTree.setValueOff(outXYZ, result);
                        }
                    }
                }
            }
        }
    }
} // GridResampler::transformBBox()

} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_GRIDTRANSFORMER_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
