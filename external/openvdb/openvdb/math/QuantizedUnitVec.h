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

#ifndef OPENVDB_MATH_QUANTIZED_UNIT_VEC_HAS_BEEN_INCLUDED
#define OPENVDB_MATH_QUANTIZED_UNIT_VEC_HAS_BEEN_INCLUDED

#include <openvdb/Platform.h>
#include <openvdb/version.h>
#include "Vec3.h"

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace math {

/// @brief Unit vector occupying only 16 bits
/// @details Stores two quantized components.  Based on the
/// "Higher Accuracy Quantized Normals" article from GameDev.Net LLC, 2000
class OPENVDB_API QuantizedUnitVec
{
public:
    template<typename T> static uint16_t pack(const Vec3<T>& vec);
    static Vec3s unpack(const uint16_t data);

    static void flipSignBits(uint16_t&);

private:
    QuantizedUnitVec() {}

    // bit masks
    static const uint16_t MASK_SLOTS = 0x1FFF; // 0001111111111111
    static const uint16_t MASK_XSLOT = 0x1F80; // 0001111110000000
    static const uint16_t MASK_YSLOT = 0x007F; // 0000000001111111
    static const uint16_t MASK_XSIGN = 0x8000; // 1000000000000000
    static const uint16_t MASK_YSIGN = 0x4000; // 0100000000000000
    static const uint16_t MASK_ZSIGN = 0x2000; // 0010000000000000

    // normalization weights, 32 kilobytes.
    static float sNormalizationWeights[MASK_SLOTS + 1];
}; // class QuantizedUnitVec


////////////////////////////////////////


template<typename T>
inline uint16_t
QuantizedUnitVec::pack(const Vec3<T>& vec)
{
    if (math::isZero(vec)) return 0;

    uint16_t data = 0;
    T x(vec[0]), y(vec[1]), z(vec[2]);

    // The sign of the three components are first stored using
    // 3-bits and can then safely be discarded.
    if (x < T(0.0)) { data |= MASK_XSIGN; x = -x; }
    if (y < T(0.0)) { data |= MASK_YSIGN; y = -y; }
    if (z < T(0.0)) { data |= MASK_ZSIGN; z = -z; }

    // The z component is discarded and x & y are quantized in
    // the 0 to 126 range.
    T w = T(126.0) / (x + y + z);
    uint16_t xbits = static_cast<uint16_t>((x * w));
    uint16_t ybits = static_cast<uint16_t>((y * w));

    // The remaining 13 bits in our 16 bit word are dividied into a
    // 6-bit x-slot and a 7-bit y-slot. Both the xbits and the ybits
    // can still be represented using (2^7 - 1) quantization levels.

    // If the xbits requre more than 6-bits, store the complement.
    // (xbits + ybits < 127, thus if xbits > 63 => ybits <= 63)
    if (xbits > 63) {
        xbits = static_cast<uint16_t>(127 - xbits);
        ybits = static_cast<uint16_t>(127 - ybits);
    }

    // Pack components into their respective slots.
    data = static_cast<uint16_t>(data | (xbits << 7));
    data = static_cast<uint16_t>(data | ybits);
    return data;
}


inline Vec3s
QuantizedUnitVec::unpack(const uint16_t data)
{
    const float w = sNormalizationWeights[data & MASK_SLOTS];

    uint16_t xbits = static_cast<uint16_t>((data & MASK_XSLOT) >> 7);
    uint16_t ybits = static_cast<uint16_t>(data & MASK_YSLOT);

    // Check if the complement components where stored and revert.
    if ((xbits + ybits) > 126) {
        xbits = static_cast<uint16_t>(127 - xbits);
        ybits = static_cast<uint16_t>(127 - ybits);
    }

    Vec3s vec(float(xbits) * w, float(ybits) * w, float(126 - xbits - ybits) * w);

    if (data & MASK_XSIGN) vec[0] = -vec[0];
    if (data & MASK_YSIGN) vec[1] = -vec[1];
    if (data & MASK_ZSIGN) vec[2] = -vec[2];
    return vec;
}


////////////////////////////////////////


inline void
QuantizedUnitVec::flipSignBits(uint16_t& v)
{
    v = static_cast<uint16_t>((v & MASK_SLOTS) | (~v & ~MASK_SLOTS));
}

} // namespace math
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_MATH_QUANTIZED_UNIT_VEC_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
