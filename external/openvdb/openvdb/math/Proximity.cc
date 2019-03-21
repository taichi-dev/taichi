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

#include "Proximity.h"

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace math {


OPENVDB_API Vec3d
closestPointOnTriangleToPoint(
    const Vec3d& a, const Vec3d& b, const Vec3d& c, const Vec3d& p, Vec3d& uvw)
{
    uvw.setZero();

    // degenerate triangle, singular
    if ((isApproxEqual(a, b) && isApproxEqual(a, c))) {
        uvw[0] = 1.0;
        return a;
    }

    Vec3d ab = b - a, ac = c - a, ap = p - a;
    double d1 = ab.dot(ap), d2 = ac.dot(ap);

    // degenerate triangle edges
    if (isApproxEqual(a, b)) {

        double t = 0.0;
        Vec3d cp = closestPointOnSegmentToPoint(a, c, p, t);

        uvw[0] = 1.0 - t;
        uvw[2] = t;

        return cp;

    } else if (isApproxEqual(a, c) || isApproxEqual(b, c)) {

        double t = 0.0;
        Vec3d cp = closestPointOnSegmentToPoint(a, b, p, t);
        uvw[0] = 1.0 - t;
        uvw[1] = t;
        return cp;
    }

    if (d1 <= 0.0 && d2 <= 0.0) {
        uvw[0] = 1.0;
        return a; // barycentric coordinates (1,0,0)
    }

    // Check if P in vertex region outside B
    Vec3d bp = p - b;
    double d3 = ab.dot(bp), d4 = ac.dot(bp);
    if (d3 >= 0.0 && d4 <= d3) {
        uvw[1] = 1.0;
        return b; // barycentric coordinates (0,1,0)
    }

    // Check if P in edge region of AB, if so return projection of P onto AB
    double vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
        uvw[1] = d1 / (d1 - d3);
        uvw[0] = 1.0 - uvw[1];
        return a + uvw[1] * ab; // barycentric coordinates (1-v,v,0)
    }

    // Check if P in vertex region outside C
    Vec3d cp = p - c;
    double d5 = ab.dot(cp), d6 = ac.dot(cp);
    if (d6 >= 0.0 && d5 <= d6) {
        uvw[2] = 1.0;
        return c; // barycentric coordinates (0,0,1)
    }

    // Check if P in edge region of AC, if so return projection of P onto AC
    double vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
        uvw[2] = d2 / (d2 - d6);
        uvw[0] = 1.0 - uvw[2];
        return a + uvw[2] * ac; // barycentric coordinates (1-w,0,w)
    }

    // Check if P in edge region of BC, if so return projection of P onto BC
    double va = d3*d6 - d5*d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
        uvw[2] = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        uvw[1] = 1.0 - uvw[2];
        return b + uvw[2] * (c - b); // barycentric coordinates (0,1-w,w)
    }

    // P inside face region. Compute Q through its barycentric coordinates (u,v,w)
    double denom = 1.0 / (va + vb + vc);
    uvw[2] = vc * denom;
    uvw[1] = vb * denom;
    uvw[0] = 1.0 - uvw[1] - uvw[2];

    return a + ab*uvw[1] + ac*uvw[2]; // = u*a + v*b + w*c , u= va*denom = 1.0-v-w
}


OPENVDB_API Vec3d
closestPointOnSegmentToPoint(const Vec3d& a, const Vec3d& b, const Vec3d& p, double& t)
{
    Vec3d ab = b - a;
    t = (p - a).dot(ab);

    if (t <= 0.0) {
        // c projects outside the [a,b] interval, on the a side.
        t = 0.0;
        return a;
    } else {

        // always nonnegative since denom = ||ab||^2
        double denom = ab.dot(ab);

        if (t >= denom) {
            // c projects outside the [a,b] interval, on the b side.
            t = 1.0;
            return b;
        } else {
            // c projects inside the [a,b] interval.
            t = t / denom;
            return a + (ab * t);
        }
    }
}

} // namespace math
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
