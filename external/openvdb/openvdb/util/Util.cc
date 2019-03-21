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

#include "Util.h"
#include <limits>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace util {

const Index32 INVALID_IDX = std::numeric_limits<Index32>::max();

const Coord COORD_OFFSETS[26] =
{
    Coord( 1,  0,  0), /// Voxel-face adjacent neghbours
    Coord(-1,  0,  0), /// 0 to 5
    Coord( 0,  1,  0),
    Coord( 0, -1,  0),
    Coord( 0,  0,  1),
    Coord( 0,  0, -1),
    Coord( 1,  0, -1), /// Voxel-edge adjacent neghbours
    Coord(-1,  0, -1), /// 6 to 17
    Coord( 1,  0,  1),
    Coord(-1,  0,  1),
    Coord( 1,  1,  0),
    Coord(-1,  1,  0),
    Coord( 1, -1,  0),
    Coord(-1, -1,  0),
    Coord( 0, -1,  1),
    Coord( 0, -1, -1),
    Coord( 0,  1,  1),
    Coord( 0,  1, -1),
    Coord(-1, -1, -1), /// Voxel-corner adjacent neghbours
    Coord(-1, -1,  1), /// 18 to 25
    Coord( 1, -1,  1),
    Coord( 1, -1, -1),
    Coord(-1,  1, -1),
    Coord(-1,  1,  1),
    Coord( 1,  1,  1),
    Coord( 1,  1, -1)
};

} // namespace util
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
