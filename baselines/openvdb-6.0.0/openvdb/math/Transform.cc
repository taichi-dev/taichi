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

#include "Transform.h"
#include "LegacyFrustum.h"

#include <openvdb/version.h>
#include <sstream>
#include <string>
#include <vector>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace math {


////////////////////////////////////////


Transform::Transform(const MapBase::Ptr& map):
    mMap(ConstPtrCast</*to=*/MapBase, /*from=*/const MapBase>(map))
{
    // auto-convert to simplest type
    if (!mMap->isType<UniformScaleMap>() && mMap->isLinear()) {
        AffineMap::Ptr affine = mMap->getAffineMap();
        mMap = simplify(affine);
    }
}

Transform::Transform(const Transform& other):
    mMap(ConstPtrCast</*to=*/MapBase, /*from=*/const MapBase>(other.baseMap()))
{
}


////////////////////////////////////////


// Factory methods

Transform::Ptr
Transform::createLinearTransform(double voxelDim)
{
    return Transform::Ptr(new Transform(
        MapBase::Ptr(new UniformScaleMap(voxelDim))));
}

Transform::Ptr
Transform::createLinearTransform(const Mat4R& m)
{
    return Transform::Ptr(new Transform(MapBase::Ptr(new AffineMap(m))));
}

Transform::Ptr
Transform::createFrustumTransform(const BBoxd& bbox, double taper,
    double depth, double voxelDim)
{
    return Transform::Ptr(new  Transform(
        NonlinearFrustumMap(bbox, taper, depth).preScale(Vec3d(voxelDim, voxelDim, voxelDim))));
}


////////////////////////////////////////


void
Transform::read(std::istream& is)
{
    // Read the type name.
    Name type = readString(is);

    if (io::getFormatVersion(is) < OPENVDB_FILE_VERSION_NEW_TRANSFORM) {
        // Handle old-style transforms.

        if (type == "LinearTransform") {
            // First read in the old transform's base class.
            Coord tmpMin, tmpMax;
            is.read(reinterpret_cast<char*>(&tmpMin), sizeof(Coord::ValueType) * 3);
            is.read(reinterpret_cast<char*>(&tmpMax), sizeof(Coord::ValueType) * 3);

            // Second read in the old linear transform
            Mat4d tmpLocalToWorld, tmpWorldToLocal, tmpVoxelToLocal, tmpLocalToVoxel;

            tmpLocalToWorld.read(is);
            tmpWorldToLocal.read(is);
            tmpVoxelToLocal.read(is);
            tmpLocalToVoxel.read(is);

            // Convert and simplify
            AffineMap::Ptr affineMap(new AffineMap(tmpVoxelToLocal*tmpLocalToWorld));
            mMap = simplify(affineMap);

        } else if (type == "FrustumTransform") {

            internal::LegacyFrustum legacyFrustum(is);

            CoordBBox bb      = legacyFrustum.getBBox();
            BBoxd   bbox(bb.min().asVec3d(), bb.max().asVec3d()
                         /* -Vec3d(1,1,1) */
                         );
            double  taper     = legacyFrustum.getTaper();
            double  depth     = legacyFrustum.getDepth();

            double nearPlaneWidth = legacyFrustum.getNearPlaneWidth();
            double nearPlaneDist  = legacyFrustum.getNearPlaneDist();
            const Mat4d& camxform        = legacyFrustum.getCamXForm();

            // create the new frustum with these parameters
            Mat4d xform(Mat4d::identity());
            xform.setToTranslation(Vec3d(0,0, -nearPlaneDist));
            xform.preScale(Vec3d(nearPlaneWidth, nearPlaneWidth, -nearPlaneWidth));

            // create the linear part of the frustum (the second map)
            Mat4d second = xform * camxform;

            // we might have precision problems, the constructor for the
            // affine map is not forgiving (so we fix here).
            const Vec4d col3 = second.col(3);
            const Vec4d ref(0, 0, 0, 1);

            if (ref.eq(col3) ) {
                second.setCol(3, ref);
            }

            MapBase::Ptr linearMap(simplify(AffineMap(second).getAffineMap()));

            // note that the depth is scaled on the nearPlaneSize.
            // the linearMap will uniformly scale the frustum to the correct size
            // and rotate to align with the camera
            mMap = MapBase::Ptr(new NonlinearFrustumMap(
                bbox, taper, depth/nearPlaneWidth, linearMap));

        } else {
            OPENVDB_THROW(IoError, "Transforms of type " + type + " are no longer supported");
        }
    } else {
        // Check if the map has been registered.
        if (!MapRegistry::isRegistered(type)) {
            OPENVDB_THROW(KeyError, "Map " << type << " is not registered");
        }

        // Create the map of the type and then read it in.
        mMap = math::MapRegistry::createMap(type);
        mMap->read(is);
    }
}


void
Transform::write(std::ostream& os) const
{
    if (!mMap) OPENVDB_THROW(IoError, "Transform does not have a map");

    // Write the type-name of the map.
    writeString(os, mMap->type());

    mMap->write(os);
}


////////////////////////////////////////


bool
Transform::isIdentity() const
{
    if (mMap->isLinear()) {
        return mMap->getAffineMap()->isIdentity();
    } else if ( mMap->isType<NonlinearFrustumMap>() ) {
        NonlinearFrustumMap::Ptr frustum =
            StaticPtrCast<NonlinearFrustumMap, MapBase>(mMap);
        return frustum->isIdentity();
    }
    // unknown nonlinear map type
    return false;
}


////////////////////////////////////////


void
Transform::preRotate(double radians, const Axis axis)
{
   mMap = mMap->preRotate(radians, axis);
}

void
Transform::preTranslate(const Vec3d& t)
{
    mMap = mMap->preTranslate(t);
}

void
Transform::preScale(const Vec3d& s)
{
    mMap = mMap->preScale(s);
}

void
Transform::preScale(double s)
{
    const Vec3d vec(s,s,s);
    mMap = mMap->preScale(vec);
}

void
Transform::preShear(double shear, Axis axis0, Axis axis1)
{
    mMap = mMap->preShear(shear, axis0, axis1);
}

void
Transform::preMult(const Mat4d& m)
{
    if (mMap->isLinear()) {

        const Mat4d currentMat4 = mMap->getAffineMap()->getMat4();
        const Mat4d newMat4 = m * currentMat4;

        AffineMap::Ptr affineMap( new AffineMap( newMat4) );
        mMap = simplify(affineMap);

    } else if (mMap->isType<NonlinearFrustumMap>() ) {

        NonlinearFrustumMap::Ptr currentFrustum =
            StaticPtrCast<NonlinearFrustumMap, MapBase>(mMap);

        const Mat4d currentMat4 = currentFrustum->secondMap().getMat4();
        const Mat4d newMat4 = m * currentMat4;

        AffineMap affine{newMat4};

        NonlinearFrustumMap::Ptr frustum{new NonlinearFrustumMap{
            currentFrustum->getBBox(),
            currentFrustum->getTaper(),
            currentFrustum->getDepth(),
            affine.copy()
        }};
        mMap = StaticPtrCast<MapBase, NonlinearFrustumMap>(frustum);
    }

}

void
Transform::preMult(const Mat3d& m)
{
    Mat4d mat4 = Mat4d::identity();
    mat4.setMat3(m);
    preMult(mat4);
}

void
Transform::postRotate(double radians, const Axis axis)
{
   mMap = mMap->postRotate(radians, axis);
}

void
Transform::postTranslate(const Vec3d& t)
{
    mMap = mMap->postTranslate(t);
}

void
Transform::postScale(const Vec3d& s)
{
    mMap = mMap->postScale(s);
}

void
Transform::postScale(double s)
{
    const Vec3d vec(s,s,s);
    mMap = mMap->postScale(vec);
}

void
Transform::postShear(double shear, Axis axis0, Axis axis1)
{
    mMap = mMap->postShear(shear, axis0, axis1);
}


void
Transform::postMult(const Mat4d& m)
{
    if (mMap->isLinear()) {

        const Mat4d currentMat4 = mMap->getAffineMap()->getMat4();
        const Mat4d newMat4 = currentMat4 * m;

        AffineMap::Ptr affineMap{new AffineMap{newMat4}};
        mMap = simplify(affineMap);

    } else if (mMap->isType<NonlinearFrustumMap>()) {

        NonlinearFrustumMap::Ptr currentFrustum =
            StaticPtrCast<NonlinearFrustumMap, MapBase>(mMap);

        const Mat4d currentMat4 = currentFrustum->secondMap().getMat4();
        const Mat4d newMat4 =  currentMat4 * m;

        AffineMap affine{newMat4};

        NonlinearFrustumMap::Ptr frustum{new NonlinearFrustumMap{
            currentFrustum->getBBox(),
            currentFrustum->getTaper(),
            currentFrustum->getDepth(),
            affine.copy()
        }};
        mMap = StaticPtrCast<MapBase, NonlinearFrustumMap>(frustum);
    }

}

void
Transform::postMult(const Mat3d& m)
{
    Mat4d mat4 = Mat4d::identity();
    mat4.setMat3(m);
    postMult(mat4);
}


////////////////////////////////////////


BBoxd
Transform::indexToWorld(const CoordBBox& indexBBox) const
{
    return this->indexToWorld(BBoxd(indexBBox.min().asVec3d(), indexBBox.max().asVec3d()));
}


BBoxd
Transform::indexToWorld(const BBoxd& indexBBox) const
{
    const Vec3d &imin = indexBBox.min(), &imax = indexBBox.max();

    Vec3d corners[8];
    corners[0] = imin;
    corners[1] = Vec3d(imax(0), imin(1), imin(2));
    corners[2] = Vec3d(imax(0), imax(1), imin(2));
    corners[3] = Vec3d(imin(0), imax(1), imin(2));
    corners[4] = Vec3d(imin(0), imin(1), imax(2));
    corners[5] = Vec3d(imax(0), imin(1), imax(2));
    corners[6] = imax;
    corners[7] = Vec3d(imin(0), imax(1), imax(2));

    BBoxd worldBBox;
    Vec3d &wmin = worldBBox.min(), &wmax = worldBBox.max();

    wmin = wmax = this->indexToWorld(corners[0]);
    for (int i = 1; i < 8; ++i) {
        Vec3d image = this->indexToWorld(corners[i]);
        wmin = minComponent(wmin, image);
        wmax = maxComponent(wmax, image);
    }
    return worldBBox;
}


BBoxd
Transform::worldToIndex(const BBoxd& worldBBox) const
{
    Vec3d indexMin, indexMax;
    calculateBounds(*this, worldBBox.min(), worldBBox.max(), indexMin, indexMax);
    return BBoxd(indexMin, indexMax);
}


CoordBBox
Transform::worldToIndexCellCentered(const BBoxd& worldBBox) const
{
    Vec3d indexMin, indexMax;
    calculateBounds(*this, worldBBox.min(), worldBBox.max(), indexMin, indexMax);
    return CoordBBox(Coord::round(indexMin), Coord::round(indexMax));
}


CoordBBox
Transform::worldToIndexNodeCentered(const BBoxd& worldBBox) const
{
    Vec3d indexMin, indexMax;
    calculateBounds(*this, worldBBox.min(), worldBBox.max(), indexMin, indexMax);
    return CoordBBox(Coord::floor(indexMin), Coord::floor(indexMax));
}


////////////////////////////////////////

// Utility methods

void
calculateBounds(const Transform& t,
                const Vec3d& minWS,
                const Vec3d& maxWS,
                Vec3d& minIS,
                Vec3d& maxIS)
{
    /// the pre-image of the 8 corners of the box
    Vec3d corners[8];
    corners[0] = minWS;
    corners[1] = Vec3d(maxWS(0), minWS(1), minWS(2));
    corners[2] = Vec3d(maxWS(0), maxWS(1), minWS(2));
    corners[3] = Vec3d(minWS(0), maxWS(1), minWS(2));
    corners[4] = Vec3d(minWS(0), minWS(1), maxWS(2));
    corners[5] = Vec3d(maxWS(0), minWS(1), maxWS(2));
    corners[6] = maxWS;
    corners[7] = Vec3d(minWS(0), maxWS(1), maxWS(2));

    Vec3d pre_image;
    minIS = t.worldToIndex(corners[0]);
    maxIS = minIS;
    for (int i = 1; i < 8; ++i) {
        pre_image = t.worldToIndex(corners[i]);
        for (int j = 0; j < 3; ++j) {
            minIS(j) = std::min(minIS(j), pre_image(j));
            maxIS(j) = std::max(maxIS(j), pre_image(j));
        }
    }
}


////////////////////////////////////////


bool
Transform::operator==(const Transform& other) const
{
    if (!this->voxelSize().eq(other.voxelSize())) return false;

    if (this->mapType() == other.mapType()) {
        return this->baseMap()->isEqual(*other.baseMap());
    }

    if (this->isLinear() && other.isLinear()) {
        // promote both maps to mat4 form and compare
        return  ( *(this->baseMap()->getAffineMap()) ==
                  *(other.baseMap()->getAffineMap()) );
    }

    return this->baseMap()->isEqual(*other.baseMap());
}


////////////////////////////////////////


void
Transform::print(std::ostream& os, const std::string& indent) const
{
    struct Local {
        // Print a Vec4d more compactly than Vec4d::str() does.
        static std::string rowAsString(const Vec4d& row)
        {
            std::ostringstream ostr;
            ostr << "[" << std::setprecision(3) << row[0] << ", "
                << row[1] << ", " << row[2] << ", " << row[3] << "] ";
            return ostr.str();
        }
    };

    // Write to a string stream so that I/O manipulators don't affect the output stream.
    std::ostringstream ostr;

    {
        Vec3d dim = this->voxelSize();
        if (dim.eq(Vec3d(dim[0]))) {
            ostr << indent << std::left << "voxel size: " << std::setprecision(3) << dim[0];
        } else {
            ostr << indent << std::left << "voxel dimensions: [" << std::setprecision(3)
                << dim[0] << ", " << dim[1] << ", " << dim[2] << "]";
        }
        ostr << "\n";
    }

    if (this->isLinear()) {
        openvdb::Mat4R v2w = this->baseMap()->getAffineMap()->getMat4();

        ostr << indent << std::left << "index to world:\n";
        for (int row = 0; row < 4; ++row) {
            ostr << indent << "   " << std::left << Local::rowAsString(v2w[row]) << "\n";
        }

    } else if (this->mapType() == NonlinearFrustumMap::mapType()) {
        const NonlinearFrustumMap& frustum =
            static_cast<const NonlinearFrustumMap&>(*this->baseMap());
        const openvdb::Mat4R linear = this->baseMap()->getAffineMap()->getMat4();

        std::vector<std::string> linearRow;
        size_t w = 0;
        for (int row = 0; row < 4; ++row) {
            std::string str = Local::rowAsString(linear[row]);
            w = std::max(w, str.size());
            linearRow.push_back(str);
        }
        w = std::max<size_t>(w, 30);
        const int iw = int(w);

        // Print rows of the linear component matrix side-by-side with frustum parameters.
        ostr << indent << std::left << std::setw(iw) << "linear:"
            << "  frustum:\n";
        ostr << indent << "   " << std::left << std::setw(iw) << linearRow[0]
            << "  taper:  " << frustum.getTaper() << "\n";
        ostr << indent << "   " << std::left << std::setw(iw) << linearRow[1]
            << "  depth:  " << frustum.getDepth() << "\n";

        std::ostringstream ostmp;
        ostmp << indent << "   " << std::left << std::setw(iw) << linearRow[2]
            << "  bounds: " << frustum.getBBox();
        if (ostmp.str().size() < 79) {
            ostr << ostmp.str() << "\n";
            ostr << indent << "   " << std::left << std::setw(iw) << linearRow[3] << "\n";
        } else {
            // If the frustum bounding box doesn't fit on one line, split it into two lines.
            ostr << indent << "   " << std::left << std::setw(iw) << linearRow[2]
                << "  bounds: " << frustum.getBBox().min() << " ->\n";
            ostr << indent << "   " << std::left << std::setw(iw) << linearRow[3]
                << "             " << frustum.getBBox().max() << "\n";
        }

    } else {
        /// @todo Handle other map types.
    }

    os << ostr.str();
}


////////////////////////////////////////


std::ostream&
operator<<(std::ostream& os, const Transform& t)
{
    os << "Transform type: " << t.baseMap()->type() << std::endl;
    os << t.baseMap()->str() << std::endl;
    return os;
}


} // namespace math
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
