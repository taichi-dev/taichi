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

/// @file points/points.cc

#include "PointDataGrid.h"

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace points {

void
internal::initialize()
{
    // Register attribute arrays with no compression
    TypedAttributeArray<bool>::registerType();
    TypedAttributeArray<int16_t>::registerType();
    TypedAttributeArray<int32_t>::registerType();
    TypedAttributeArray<int64_t>::registerType();
    TypedAttributeArray<float>::registerType();
    TypedAttributeArray<double>::registerType();
    TypedAttributeArray<math::Vec3<int32_t>>::registerType();
    TypedAttributeArray<math::Vec3<float>>::registerType();
    TypedAttributeArray<math::Vec3<double>>::registerType();

    // Register attribute arrays with group and string attribute
    GroupAttributeArray::registerType();
    StringAttributeArray::registerType();

    // Register attribute arrays with matrix and quaternion attributes
    TypedAttributeArray<math::Mat3<float>>::registerType();
    TypedAttributeArray<math::Mat3<double>>::registerType();
    TypedAttributeArray<math::Mat4<float>>::registerType();
    TypedAttributeArray<math::Mat4<double>>::registerType();
    TypedAttributeArray<math::Quat<float>>::registerType();
    TypedAttributeArray<math::Quat<double>>::registerType();

    // Register attribute arrays with truncate compression
    TypedAttributeArray<float, TruncateCodec>::registerType();
    TypedAttributeArray<math::Vec3<float>, TruncateCodec>::registerType();

    // Register attribute arrays with fixed point compression
    TypedAttributeArray<math::Vec3<float>, FixedPointCodec<true>>::registerType();
    TypedAttributeArray<math::Vec3<float>, FixedPointCodec<false>>::registerType();
    TypedAttributeArray<math::Vec3<float>, FixedPointCodec<true, PositionRange>>::registerType();
    TypedAttributeArray<math::Vec3<float>, FixedPointCodec<false, PositionRange>>::registerType();
    TypedAttributeArray<math::Vec3<float>, FixedPointCodec<true, UnitRange>>::registerType();
    TypedAttributeArray<math::Vec3<float>, FixedPointCodec<false, UnitRange>>::registerType();

    // Register attribute arrays with unit vector compression
    TypedAttributeArray<math::Vec3<float>, UnitVecCodec>::registerType();

    // Register types associated with point data grids.
    Metadata::registerType(typeNameAsString<PointDataIndex32>(), Int32Metadata::createMetadata);
    Metadata::registerType(typeNameAsString<PointDataIndex64>(), Int64Metadata::createMetadata);
    PointDataGrid::registerGrid();
}


void
internal::uninitialize()
{
    AttributeArray::clearRegistry();
}

} // namespace points
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
