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
/// @file MapsUtil.h

#ifndef OPENVDB_UTIL_MAPSUTIL_HAS_BEEN_INCLUDED
#define OPENVDB_UTIL_MAPSUTIL_HAS_BEEN_INCLUDED

#include <openvdb/math/Maps.h>
#include <algorithm> // for std::min(), std::max()
#include <cmath>
#include <vector>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace util {

// Utility methods for calculating bounding boxes

/// @brief Calculate an axis-aligned bounding box in the given map's domain
/// (e.g., index space) from an axis-aligned bounding box in its range
/// (e.g., world space)
template<typename MapType>
inline void
calculateBounds(const MapType& map, const BBoxd& in, BBoxd& out)
{
    const Vec3d& min = in.min();
    const Vec3d& max = in.max();

    // the pre-image of the 8 corners of the box
    Vec3d corners[8];
    corners[0] = in.min();;
    corners[1] = Vec3d(min(0), min(1), min(2));
    corners[2] = Vec3d(max(0), max(1), min(2));
    corners[3] = Vec3d(min(0), max(1), min(2));
    corners[4] = Vec3d(min(0), min(1), max(2));
    corners[5] = Vec3d(max(0), min(1), max(2));
    corners[6] = max;
    corners[7] = Vec3d(min(0), max(1), max(2));

    Vec3d pre_image;
    Vec3d& out_min = out.min();
    Vec3d& out_max = out.max();
    out_min = map.applyInverseMap(corners[0]);
    out_max = min;
    for (int i = 1; i < 8; ++i) {
        pre_image = map.applyInverseMap(corners[i]);
        for (int j = 0; j < 3; ++j) {
            out_min(j) = std::min( out_min(j), pre_image(j));
            out_max(j) = std::max( out_max(j), pre_image(j));
        }
    }
}


/// @brief Calculate an axis-aligned bounding box in the given map's domain
/// from a spherical bounding box in its range.
template<typename MapType>
inline void
calculateBounds(const MapType& map, const Vec3d& center, const Real radius, BBoxd& out)
{
    // On return, out gives a bounding box in continuous index space
    // that encloses the sphere.
    //
    // the image of a sphere under the inverse of the linearMap will be an ellipsoid.

    if (math::is_linear<MapType>::value) {
        // I want to find extrema for three functions f(x', y', z') = x', or = y', or = z'
        // with the constraint that g = (x-xo)^2 + (y-yo)^2 + (z-zo)^2 = r^2.
        // Where the point x,y,z is the image of x',y',z'
        // Solve: \lambda Grad(g) = Grad(f) and g = r^2.
        // Note: here (x,y,z) is the image of (x',y',z'), and the gradient
        // is w.r.t the (') space.
        //
        // This can be solved exactly: e_a^T (x' -xo') =\pm r\sqrt(e_a^T J^(-1)J^(-T)e_a)
        // where e_a is one of the three unit vectors.   -  djh.

        /// find the image of the center of the sphere
        Vec3d center_pre_image = map.applyInverseMap(center);

        std::vector<Vec3d> coordinate_units;
        coordinate_units.push_back(Vec3d(1,0,0));
        coordinate_units.push_back(Vec3d(0,1,0));
        coordinate_units.push_back(Vec3d(0,0,1));

        Vec3d& out_min = out.min();
        Vec3d& out_max = out.max();
        for (int direction = 0; direction < 3; ++direction) {
            Vec3d temp  = map.applyIJT(coordinate_units[direction]);
            double offset =
                radius * sqrt(temp.x()*temp.x() + temp.y()*temp.y() + temp.z()*temp.z());
            out_min(direction) = center_pre_image(direction) - offset;
            out_max(direction) = center_pre_image(direction) + offset;
        }

    } else {
        // This is some unknown map type.  In this case, we form an axis-aligned
        // bounding box for the sphere in world space and find the pre-images of
        // the corners in index space.  From these corners we compute an axis-aligned
        // bounding box in index space.
        BBoxd bounding_box(center - radius*Vec3d(1,1,1), center + radius*Vec3d(1,1,1));
        calculateBounds<MapType>(map, bounding_box, out);
    }
}


namespace { // anonymous namespace for this helper function

/// @brief Find the intersection of a line passing through the point
/// (<I>x</I>=0,&nbsp;<I>z</I>=&minus;1/<I>g</I>) with the circle
/// (<I>x</I> &minus; <I>xo</I>)&sup2; + (<I>z</I> &minus; <I>zo</I>)&sup2; = <I>r</I>&sup2;
/// at a point tangent to the circle.
/// @return 0 if the focal point (0, -1/<I>g</I>) is inside the circle,
/// 1 if the focal point touches the circle, or 2 when both points are found.
inline int
findTangentPoints(const double g, const double xo, const double zo,
    const double r, double& xp, double& zp, double& xm, double& zm)
{
    double x2 = xo * xo;
    double r2 = r * r;
    double xd = g * xo;
    double xd2 = xd*xd;
    double zd = g * zo + 1.;
    double zd2 = zd*zd;
    double rd2 = r2*g*g;

    double distA = xd2 + zd2;
    double distB = distA - rd2;

    if (distB > 0) {
        double discriminate = sqrt(distB);

        xp = xo - xo*rd2/distA + r * zd *discriminate / distA;
        xm = xo - xo*rd2/distA - r * zd *discriminate / distA;

        zp = (zo*zd2 + zd*g*(x2 - r2) - xo*xo*g - r*xd*discriminate) / distA;
        zm = (zo*zd2 + zd*g*(x2 - r2) - xo*xo*g + r*xd*discriminate) / distA;

        return 2;

    } if (0 >= distB && distB >= -1e-9) {
        // the circle touches the focal point (x=0, z = -1/g)
        xp = 0;    xm = 0;
        zp = -1/g; zm = -1/g;

        return 1;
    }

    return 0;
}

} // end anonymous namespace


/// @brief Calculate an axis-aligned bounding box in index space
/// from a spherical bounding box in world space.
/// @note This specialization is optimized for a frustum map
template<>
inline void
calculateBounds<math::NonlinearFrustumMap>(const math::NonlinearFrustumMap& frustum,
    const Vec3d& center, const Real radius, BBoxd& out)
{
    // The frustum is a nonlinear map followed by a uniform scale, rotation, translation.
    // First we invert the translation, rotation and scale to find the spherical pre-image
    // of the sphere in "local" coordinates where the frustum is aligned with the near plane
    // on the z=0 plane and the "camera" is located at (x=0, y=0, z=-1/g).

    // check that the internal map has no shear.
    const math::AffineMap& secondMap = frustum.secondMap();
    // test if the linear part has shear or non-uniform scaling
    if (!frustum.hasSimpleAffine()) {

        // In this case, we form an axis-aligned bounding box for sphere in world space
        // and find the pre_images of the corners in voxel space.  From these corners we
        // compute an axis-algined bounding box in voxel-spae
        BBoxd bounding_box(center - radius*Vec3d(1,1,1), center + radius*Vec3d(1,1,1));
        calculateBounds<math::NonlinearFrustumMap>(frustum, bounding_box, out);
        return;
    }

    // for convenience
    Vec3d& out_min = out.min();
    Vec3d& out_max = out.max();

    Vec3d centerLS = secondMap.applyInverseMap(center);
    Vec3d voxelSize = secondMap.voxelSize();

    // all the voxels have the same size since we know this is a simple affine map
    double radiusLS = radius / voxelSize(0);

    double gamma = frustum.getGamma();
    double xp;
    double zp;
    double xm;
    double zm;
    int soln_number;

    // the bounding box in index space for the points in the frustum
    const BBoxd&  bbox = frustum.getBBox();
    // initialize min and max
    const double x_min = bbox.min().x();
    const double y_min = bbox.min().y();
    const double z_min = bbox.min().z();

    const double x_max = bbox.max().x();
    const double y_max = bbox.max().y();
    const double z_max = bbox.max().z();

    out_min.x() = x_min;
    out_max.x() = x_max;
    out_min.y() = y_min;
    out_max.y() = y_max;

    Vec3d extreme;
    Vec3d extreme2;
    Vec3d pre_image;
    // find the x-range
    soln_number = findTangentPoints(gamma, centerLS.x(), centerLS.z(), radiusLS, xp, zp, xm, zm);
    if (soln_number == 2) {
        extreme.x() = xp;
        extreme.y() = centerLS.y();
        extreme.z() = zp;

        // location in world space of the tangent point
        extreme2 = secondMap.applyMap(extreme);
        // convert back to voxel space
        pre_image = frustum.applyInverseMap(extreme2);
        out_max.x() = std::max(x_min, std::min(x_max, pre_image.x()));

        extreme.x() = xm;
        extreme.y() = centerLS.y();
        extreme.z() = zm;
        // location in world space of the tangent point
        extreme2 = secondMap.applyMap(extreme);

        // convert back to voxel space
        pre_image = frustum.applyInverseMap(extreme2);
        out_min.x() = std::max(x_min, std::min(x_max, pre_image.x()));

    } else if (soln_number == 1) {
        // the circle was tangent at the focal point
    } else if (soln_number == 0) {
        // the focal point was inside the circle
    }

    // find the y-range
    soln_number = findTangentPoints(gamma, centerLS.y(), centerLS.z(), radiusLS, xp, zp, xm, zm);
    if (soln_number == 2) {
        extreme.x() = centerLS.x();
        extreme.y() = xp;
        extreme.z() = zp;

        // location in world space of the tangent point
        extreme2 = secondMap.applyMap(extreme);
        // convert back to voxel space
        pre_image = frustum.applyInverseMap(extreme2);
        out_max.y() = std::max(y_min, std::min(y_max, pre_image.y()));

        extreme.x() = centerLS.x();
        extreme.y() = xm;
        extreme.z() = zm;
        extreme2 = secondMap.applyMap(extreme);

        // convert back to voxel space
        pre_image = frustum.applyInverseMap(extreme2);
        out_min.y() = std::max(y_min, std::min(y_max, pre_image.y()));

    } else if (soln_number == 1) {
        // the circle was tangent at the focal point
    } else if (soln_number == 0) {
        // the focal point was inside the circle
    }

    // the near and far
    // the closest point.  The front of the frustum is at 0 in index space
    double near_dist = std::max(centerLS.z() - radiusLS, 0.);
    // the farthest point.  The back of the frustum is at mDepth in index space
    double far_dist = std::min(centerLS.z() + radiusLS, frustum.getDepth() );

    Vec3d near_point(0.f, 0.f, near_dist);
    Vec3d far_point(0.f, 0.f, far_dist);

    out_min.z() = std::max(z_min, frustum.applyInverseMap(secondMap.applyMap(near_point)).z());
    out_max.z() = std::min(z_max, frustum.applyInverseMap(secondMap.applyMap(far_point)).z());

}

} // namespace util
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_UTIL_MAPSUTIL_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
