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
/// @file math/LegacyFrustum.h

#ifndef OPENVDB_MATH_LEGACYFRUSTUM_HAS_BEEN_INCLUDED
#define OPENVDB_MATH_LEGACYFRUSTUM_HAS_BEEN_INCLUDED

#include <iostream>
#include <openvdb/Types.h> // for Real typedef
#include "Coord.h"
#include "Mat4.h"
#include "Vec3.h"


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace math {
namespace internal {

/// @brief LegacyFrustum class used at DreamWorks for converting old vdb files.
class LegacyFrustum
{
public:
    LegacyFrustum(std::istream& is)
    {
        // First read in the old transform's base class.
        // the "extents"
        Vec3i tmpMin, tmpMax;
        is.read(reinterpret_cast<char*>(&tmpMin), sizeof(Vec3i::ValueType) * 3);
        is.read(reinterpret_cast<char*>(&tmpMax), sizeof(Vec3i::ValueType) * 3);

        Coord tmpMinCoord(tmpMin);
        Coord tmpMaxCoord(tmpMax);

        // set the extents
        mExtents = CoordBBox(tmpMinCoord, tmpMaxCoord);

        // read the old-frustum class member data
        //Mat4d tmpW2C;
        Mat4d tmpW2C, tmpC2S, tmpS2C, tmpWorldToLocal;
        Mat4d tmpS2U, tmpXYLocalToUnit, tmpZLocalToUnit;
        Real tmpWindow[6];
        Real tmpPadding;

        //Mat4d  tmpXYUnitToLocal, tmpZUnitToLocal

        // read in each matrix.
        is.read(reinterpret_cast<char*>(&tmpW2C),
            sizeof(Mat4d::value_type) * Mat4d::size * Mat4d::size);
        is.read(reinterpret_cast<char*>(&mC2W),
            sizeof(Mat4d::value_type) * Mat4d::size * Mat4d::size);
        is.read(reinterpret_cast<char*>(&tmpC2S),
            sizeof(Mat4d::value_type) * Mat4d::size * Mat4d::size);
        is.read(reinterpret_cast<char*>(&tmpS2C),
            sizeof(Mat4d::value_type) * Mat4d::size * Mat4d::size);
        is.read(reinterpret_cast<char*>(&tmpWorldToLocal),
            sizeof(Mat4d::value_type) * Mat4d::size * Mat4d::size);
        is.read(reinterpret_cast<char*>(&mLocalToWorld),
            sizeof(Mat4d::value_type) * Mat4d::size * Mat4d::size);

        is.read(reinterpret_cast<char*>(&tmpWindow[0]), sizeof(Real));
        is.read(reinterpret_cast<char*>(&tmpWindow[1]), sizeof(Real));
        is.read(reinterpret_cast<char*>(&tmpWindow[2]), sizeof(Real));
        is.read(reinterpret_cast<char*>(&tmpWindow[3]), sizeof(Real));
        is.read(reinterpret_cast<char*>(&tmpWindow[4]), sizeof(Real));
        is.read(reinterpret_cast<char*>(&tmpWindow[5]), sizeof(Real));

        is.read(reinterpret_cast<char*>(&tmpPadding), sizeof(Real));

        is.read(reinterpret_cast<char*>(&tmpS2U),
            sizeof(Mat4d::value_type) * Mat4d::size * Mat4d::size);
        is.read(reinterpret_cast<char*>(&mXYUnitToLocal),
            sizeof(Mat4d::value_type) * Mat4d::size * Mat4d::size);
        is.read(reinterpret_cast<char*>(&tmpXYLocalToUnit),
            sizeof(Mat4d::value_type) * Mat4d::size * Mat4d::size);
        is.read(reinterpret_cast<char*>(&mZUnitToLocal),
            sizeof(Mat4d::value_type) * Mat4d::size * Mat4d::size);
        is.read(reinterpret_cast<char*>(&tmpZLocalToUnit),
            sizeof(Mat4d::value_type) * Mat4d::size * Mat4d::size);


        mNearPlane = tmpWindow[4];
        mFarPlane  = tmpWindow[5];

        // Look up the world space corners of the
        // frustum grid.
        mFrNearOrigin = unitToLocalFrustum(Vec3R(0,0,0));
        mFrFarOrigin = unitToLocalFrustum(Vec3R(0,0,1));

        Vec3d frNearXTip = unitToLocalFrustum(Vec3R(1,0,0));
        Vec3d frNearYTip = unitToLocalFrustum(Vec3R(0,1,0));
        mFrNearXBasis = frNearXTip - mFrNearOrigin;
        mFrNearYBasis = frNearYTip - mFrNearOrigin;

        Vec3R frFarXTip = unitToLocalFrustum(Vec3R(1,0,1));
        Vec3R frFarYTip = unitToLocalFrustum(Vec3R(0,1,1));
        mFrFarXBasis = frFarXTip - mFrFarOrigin;
        mFrFarYBasis = frFarYTip - mFrFarOrigin;
    }

    ~LegacyFrustum() {}

    const Mat4d& getCamXForm() const {return mC2W; }

    double getDepth() const {return (mFarPlane - mNearPlane); }
    double getTaper() const {

        return   getNearPlaneWidth() / getFarPlaneWidth();
    }

    double getNearPlaneWidth() const {
        double nearPlaneWidth  = (unitToWorld(Vec3d(0,0,0)) - unitToWorld(Vec3d(1,0,0))).length();
        return nearPlaneWidth;
    }

    double getFarPlaneWidth() const {
        double farPlaneWidth = (unitToWorld(Vec3d(0,0,1)) - unitToWorld(Vec3d(1,0,1))).length();
        return farPlaneWidth;
    }

    double getNearPlaneDist() const { return mNearPlane; }

    const CoordBBox& getBBox() const {return mExtents; }

    Vec3d unitToWorld(const Vec3d& in) const {return mLocalToWorld.transform( unitToLocal(in) ); }

private:
    LegacyFrustum() {}

    Vec3d unitToLocal(const Vec3d& U) const {

        // We first find the local space coordinates
        // of the unit point projected onto the near
        // and far planes of the frustum by using a
        // linear combination of the planes basis vectors
        Vec3d nearLS = ( U[0] * mFrNearXBasis ) + ( U[1] * mFrNearYBasis ) + mFrNearOrigin;
        Vec3d farLS  = ( U[0] * mFrFarXBasis  ) + ( U[1] * mFrFarYBasis  ) + mFrFarOrigin;

        // then we lerp the two ws points in frustum z space
        return U[2] * farLS + ( 1.0 - U[2] ) * nearLS;
    }

    Vec3d unitToLocalFrustum(const Vec3d& u) const {
        Vec3d fzu = mZUnitToLocal.transformH(u);
        Vec3d fu = u;
        fu[2] = fzu.z();
        return mXYUnitToLocal.transformH(fu);
    }

private:
    Mat4d mC2W, mLocalToWorld, mXYUnitToLocal, mZUnitToLocal;
    CoordBBox mExtents;
    Vec3d mFrNearXBasis, mFrNearYBasis, mFrFarXBasis, mFrFarYBasis;
    Vec3d mFrNearOrigin, mFrFarOrigin;
    double mNearPlane, mFarPlane;
};

} // namespace internal
} // namespace math
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_MATH_LEGACYFRUSTUM_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
