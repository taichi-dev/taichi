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

#ifndef OPENVDB_UNITTEST_UTIL_HAS_BEEN_INCLUDED
#define OPENVDB_UNITTEST_UTIL_HAS_BEEN_INCLUDED

#include <openvdb/openvdb.h>
#include <openvdb/math/Math.h> // for math::Random01
#include <openvdb/tools/Prune.h>// for pruneLevelSet
#include <sstream>

namespace unittest_util {

enum SphereMode { SPHERE_DENSE, SPHERE_DENSE_NARROW_BAND, SPHERE_SPARSE_NARROW_BAND };

/// @brief Generates the signed distance to a sphere located at @a center
/// and with a specified @a radius (both in world coordinates). Only voxels
/// in the domain [0,0,0] -> @a dim are considered. Also note that the
/// level set is either dense, dense narrow-band or sparse narrow-band.
///
/// @note This method is VERY SLOW and should only be used for debugging purposes!
/// However it works for any transform and even with open level sets.
/// A faster approch for closed narrow band generation is to only set voxels
/// sparsely and then use grid::signedFloodFill to define the sign
/// of the background values and tiles! This is implemented in openvdb/tools/LevelSetSphere.h
template<class GridType>
inline void
makeSphere(const openvdb::Coord& dim, const openvdb::Vec3f& center, float radius,
           GridType& grid, SphereMode mode)
{
    typedef typename GridType::ValueType ValueT;
    const ValueT
        zero = openvdb::zeroVal<ValueT>(),
        outside = grid.background(),
        inside = -outside;

    typename GridType::Accessor acc = grid.getAccessor();
    openvdb::Coord xyz;
    for (xyz[0]=0; xyz[0]<dim[0]; ++xyz[0]) {
        for (xyz[1]=0; xyz[1]<dim[1]; ++xyz[1]) {
            for (xyz[2]=0; xyz[2]<dim[2]; ++xyz[2]) {
                const openvdb::Vec3R p =  grid.transform().indexToWorld(xyz);
                const float dist = float((p-center).length() - radius);
                ValueT val = ValueT(zero + dist);
                switch (mode) {
                case SPHERE_DENSE:
                    acc.setValue(xyz, val);
                    break;
                case SPHERE_DENSE_NARROW_BAND:
                    acc.setValue(xyz, val < inside ? inside : outside < val ? outside : val);
                    break;
                case SPHERE_SPARSE_NARROW_BAND:
                    if (val < inside)
                        acc.setValueOff(xyz, inside);
                    else if (outside < val)
                        acc.setValueOff(xyz, outside);
                    else
                        acc.setValue(xyz, val);
                }
            }
        }
    }
    //if (mode == SPHERE_SPARSE_NARROW_BAND) grid.tree().prune();
    if (mode == SPHERE_SPARSE_NARROW_BAND) openvdb::tools::pruneLevelSet(grid.tree());
}

// Template specialization for boolean trees (mostly a dummy implementation)
template<>
inline void
makeSphere<openvdb::BoolGrid>(const openvdb::Coord& dim, const openvdb::Vec3f& center,
                              float radius, openvdb::BoolGrid& grid, SphereMode)
{
    openvdb::BoolGrid::Accessor acc = grid.getAccessor();
    openvdb::Coord xyz;
    for (xyz[0]=0; xyz[0]<dim[0]; ++xyz[0]) {
        for (xyz[1]=0; xyz[1]<dim[1]; ++xyz[1]) {
            for (xyz[2]=0; xyz[2]<dim[2]; ++xyz[2]) {
                const openvdb::Vec3R p =  grid.transform().indexToWorld(xyz);
                const float dist = static_cast<float>((p-center).length() - radius);
                if (dist <= 0) acc.setValue(xyz, true);
            }
        }
    }
}

// This method will soon be replaced by the one above!!!!!
template<class GridType>
inline void
makeSphere(const openvdb::Coord& dim, const openvdb::Vec3f& center, float radius,
           GridType &grid, float dx, SphereMode mode)
{
    grid.setTransform(openvdb::math::Transform::createLinearTransform(/*voxel size=*/dx));
    makeSphere<GridType>(dim, center, radius, grid, mode);
}

// Generate random points by uniformly distributing points
// on a unit-sphere.
inline void genPoints(const int numPoints, std::vector<openvdb::Vec3R>& points)
{
    // init
    openvdb::math::Random01 randNumber(0);
    const int n = int(std::sqrt(double(numPoints)));
    const double xScale = (2.0 * M_PI) / double(n);
    const double yScale = M_PI / double(n);
    
    double x, y, theta, phi;
    openvdb::Vec3R pos;
    
    points.reserve(n*n);
    
    // loop over a [0 to n) x [0 to n) grid.
    for (int a = 0; a < n; ++a) {
        for (int b = 0; b < n; ++b) {
            
            // jitter, move to random pos. inside the current cell
            x = double(a) + randNumber();
            y = double(b) + randNumber();
            
            // remap to a lat/long map
            theta = y * yScale; // [0 to PI]
            phi   = x * xScale; // [0 to 2PI]
            
            // convert to cartesian coordinates on a unit sphere.
            // spherical coordinate triplet (r=1, theta, phi)
            pos[0] = std::sin(theta)*std::cos(phi);
            pos[1] = std::sin(theta)*std::sin(phi);
            pos[2] = std::cos(theta);
            
            points.push_back(pos);
        }
    }
}

// @todo makePlane

} // namespace unittest_util

#endif // OPENVDB_UNITTEST_UTIL_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
