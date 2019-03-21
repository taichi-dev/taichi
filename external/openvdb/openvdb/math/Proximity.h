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

#ifndef OPENVDB_MATH_PROXIMITY_HAS_BEEN_INCLUDED
#define OPENVDB_MATH_PROXIMITY_HAS_BEEN_INCLUDED

#include <openvdb/Types.h>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace math {

/// @brief Closest Point on Triangle to Point. Given a triangle @c abc and a point @c p,
/// return the point on @c abc closest to @c p and the corresponding barycentric coordinates.
///
/// @details Algorithms from "Real-Time Collision Detection" pg 136 to 142 by Christer Ericson.
/// The closest point is obtained by first determining which of the triangles'
/// Voronoi feature regions @c p is in and then computing the orthogonal projection
/// of @c p onto the corresponding feature.
///
/// @param a    The triangle's first vertex point.
/// @param b    The triangle's second vertex point.
/// @param c    The triangle's third vertex point.
/// @param p    Point to compute the closest point on @c abc for.
/// @param uvw  Barycentric coordinates, computed and returned.
OPENVDB_API Vec3d
closestPointOnTriangleToPoint(
    const Vec3d& a, const Vec3d& b, const Vec3d& c, const Vec3d& p, Vec3d& uvw);


/// @brief  Closest Point on Line Segment to Point. Given segment @c ab and point @c p,
/// return the point on @c ab closest to @c p and @c t the parametric distance to @c b.
///
/// @param a    The segment's first vertex point.
/// @param b    The segment's second vertex point.
/// @param p    Point to compute the closest point on @c ab for.
/// @param t    Parametric distance to @c b.
OPENVDB_API Vec3d
closestPointOnSegmentToPoint(
    const Vec3d& a, const Vec3d& b, const Vec3d& p, double& t);

} // namespace math
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_MESH_TO_VOLUME_UTIL_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
