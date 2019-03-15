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

#ifndef OPENVDB_IO_COMPRESSION_HAS_BEEN_INCLUDED
#define OPENVDB_IO_COMPRESSION_HAS_BEEN_INCLUDED

#include <openvdb/Types.h>
#include <openvdb/math/Math.h> // for negative()
#include "io.h" // for getDataCompression(), etc.
#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <vector>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace io {

/// @brief OR-able bit flags for compression options on input and output streams
/// @details
/// <dl>
/// <dt><tt>COMPRESS_NONE</tt>
/// <dd>On write, don't compress data.<br>
///     On read, the input stream contains uncompressed data.
///
/// <dt><tt>COMPRESS_ZIP</tt>
/// <dd>When writing grids other than level sets or fog volumes, apply
///     ZLIB compression to internal and leaf node value buffers.<br>
///     When reading grids other than level sets or fog volumes, indicate that
///     the value buffers of internal and leaf nodes are ZLIB-compressed.<br>
///     ZLIB compresses well but is slow.
///
/// <dt><tt>COMPRESS_ACTIVE_MASK</tt>
/// <dd>When writing a grid of any class, don't output a node's inactive values
///     if it has two or fewer distinct values.  Instead, output minimal information
///     to permit the lossless reconstruction of inactive values.<br>
///     On read, nodes might have been stored without inactive values.
///     Where necessary, reconstruct inactive values from available information.
///
/// <dt><tt>COMPRESS_BLOSC</tt>
/// <dd>When writing grids other than level sets or fog volumes, apply
///     Blosc compression to internal and leaf node value buffers.<br>
///     When reading grids other than level sets or fog volumes, indicate that
///     the value buffers of internal and leaf nodes are Blosc-compressed.<br>
///     Blosc is much faster than ZLIB and produces comparable file sizes.
/// </dl>
enum {
    COMPRESS_NONE           = 0,
    COMPRESS_ZIP            = 0x1,
    COMPRESS_ACTIVE_MASK    = 0x2,
    COMPRESS_BLOSC          = 0x4
};

/// Return a string describing the given compression flags.
OPENVDB_API std::string compressionToString(uint32_t flags);


////////////////////////////////////////


/// @internal Per-node indicator byte that specifies what additional metadata
/// is stored to permit reconstruction of inactive values
enum {
    /*0*/ NO_MASK_OR_INACTIVE_VALS,     // no inactive vals, or all inactive vals are +background
    /*1*/ NO_MASK_AND_MINUS_BG,         // all inactive vals are -background
    /*2*/ NO_MASK_AND_ONE_INACTIVE_VAL, // all inactive vals have the same non-background val
    /*3*/ MASK_AND_NO_INACTIVE_VALS,    // mask selects between -background and +background
    /*4*/ MASK_AND_ONE_INACTIVE_VAL,    // mask selects between backgd and one other inactive val
    /*5*/ MASK_AND_TWO_INACTIVE_VALS,   // mask selects between two non-background inactive vals
    /*6*/ NO_MASK_AND_ALL_VALS          // > 2 inactive vals, so no mask compression at all
};


////////////////////////////////////////


/// @brief RealToHalf and its specializations define a mapping from
/// floating-point data types to analogous half float types.
template<typename T>
struct RealToHalf {
    enum { isReal = false }; // unless otherwise specified, type T is not a floating-point type
    using HalfT = T; // type T's half float analogue is T itself
    static HalfT convert(const T& val) { return val; }
};
template<> struct RealToHalf<float> {
    enum { isReal = true };
    using HalfT = half;
    static HalfT convert(float val) { return HalfT(val); }
};
template<> struct RealToHalf<double> {
    enum { isReal = true };
    using HalfT = half;
    // A half can only be constructed from a float, so cast the value to a float first.
    static HalfT convert(double val) { return HalfT(float(val)); }
};
template<> struct RealToHalf<Vec2s> {
    enum { isReal = true };
    using HalfT = Vec2H;
    static HalfT convert(const Vec2s& val) { return HalfT(val); }
};
template<> struct RealToHalf<Vec2d> {
    enum { isReal = true };
    using HalfT = Vec2H;
    // A half can only be constructed from a float, so cast the vector's elements to floats first.
    static HalfT convert(const Vec2d& val) { return HalfT(Vec2s(val)); }
};
template<> struct RealToHalf<Vec3s> {
    enum { isReal = true };
    using HalfT = Vec3H;
    static HalfT convert(const Vec3s& val) { return HalfT(val); }
};
template<> struct RealToHalf<Vec3d> {
    enum { isReal = true };
    using HalfT = Vec3H;
    // A half can only be constructed from a float, so cast the vector's elements to floats first.
    static HalfT convert(const Vec3d& val) { return HalfT(Vec3s(val)); }
};


/// Return the given value truncated to 16-bit float precision.
template<typename T>
inline T
truncateRealToHalf(const T& val)
{
    return T(RealToHalf<T>::convert(val));
}


////////////////////////////////////////


OPENVDB_API void zipToStream(std::ostream&, const char* data, size_t numBytes);
OPENVDB_API void unzipFromStream(std::istream&, char* data, size_t numBytes);
OPENVDB_API void bloscToStream(std::ostream&, const char* data, size_t valSize, size_t numVals);
OPENVDB_API void bloscFromStream(std::istream&, char* data, size_t numBytes);

/// @brief Read data from a stream.
/// @param is           the input stream
/// @param data         the contiguous array of data to read in
/// @param count        the number of elements to read in
/// @param compression  whether and how the data is compressed (either COMPRESS_NONE,
///                     COMPRESS_ZIP, COMPRESS_ACTIVE_MASK or COMPRESS_BLOSC)
/// @throw IoError if @a compression is COMPRESS_BLOSC but OpenVDB was compiled
/// without Blosc support.
/// @details This default implementation is instantiated only for types
/// whose size can be determined by the sizeof() operator.
template<typename T>
inline void
readData(std::istream& is, T* data, Index count, uint32_t compression)
{
    if (compression & COMPRESS_BLOSC) {
        bloscFromStream(is, reinterpret_cast<char*>(data), sizeof(T) * count);
    } else if (compression & COMPRESS_ZIP) {
        unzipFromStream(is, reinterpret_cast<char*>(data), sizeof(T) * count);
    } else {
        if (data == nullptr) {
            assert(!getStreamMetadataPtr(is) || getStreamMetadataPtr(is)->seekable());
            is.seekg(sizeof(T) * count, std::ios_base::cur);
        } else {
            is.read(reinterpret_cast<char*>(data), sizeof(T) * count);
        }
    }
}

/// Specialization for std::string input
template<>
inline void
readData<std::string>(std::istream& is, std::string* data, Index count, uint32_t /*compression*/)
{
    for (Index i = 0; i < count; ++i) {
        size_t len = 0;
        is >> len;
        //data[i].resize(len);
        //is.read(&(data[i][0]), len);

        std::string buffer(len+1, ' ');
        is.read(&buffer[0], len+1);
        if (data != nullptr) data[i].assign(buffer, 0, len);
    }
}

/// HalfReader wraps a static function, read(), that is analogous to readData(), above,
/// except that it is partially specialized for floating-point types in order to promote
/// 16-bit half float values to full float.  A wrapper class is required because
/// only classes, not functions, can be partially specialized.
template<bool IsReal, typename T> struct HalfReader;
/// Partial specialization for non-floating-point types (no half to float promotion)
template<typename T>
struct HalfReader</*IsReal=*/false, T> {
    static inline void read(std::istream& is, T* data, Index count, uint32_t compression) {
        readData(is, data, count, compression);
    }
};
/// Partial specialization for floating-point types
template<typename T>
struct HalfReader</*IsReal=*/true, T> {
    using HalfT = typename RealToHalf<T>::HalfT;
    static inline void read(std::istream& is, T* data, Index count, uint32_t compression) {
        if (count < 1) return;
        if (data == nullptr) {
            // seek mode - pass through null pointer
            readData<HalfT>(is, nullptr, count, compression);
        } else {
            std::vector<HalfT> halfData(count); // temp buffer into which to read half float values
            readData<HalfT>(is, reinterpret_cast<HalfT*>(&halfData[0]), count, compression);
            // Copy half float values from the temporary buffer to the full float output array.
            std::copy(halfData.begin(), halfData.end(), data);
        }
    }
};


/// Write data to a stream.
/// @param os           the output stream
/// @param data         the contiguous array of data to write
/// @param count        the number of elements to write out
/// @param compression  whether and how to compress the data (either COMPRESS_NONE,
///                     COMPRESS_ZIP, COMPRESS_ACTIVE_MASK or COMPRESS_BLOSC)
/// @throw IoError if @a compression is COMPRESS_BLOSC but OpenVDB was compiled
/// without Blosc support.
/// @details This default implementation is instantiated only for types
/// whose size can be determined by the sizeof() operator.
template<typename T>
inline void
writeData(std::ostream &os, const T *data, Index count, uint32_t compression)
{
    if (compression & COMPRESS_BLOSC) {
        bloscToStream(os, reinterpret_cast<const char*>(data), sizeof(T), count);
    } else if (compression & COMPRESS_ZIP) {
        zipToStream(os, reinterpret_cast<const char*>(data), sizeof(T) * count);
    } else {
        os.write(reinterpret_cast<const char*>(data), sizeof(T) * count);
    }
}

/// Specialization for std::string output
template<>
inline void
writeData<std::string>(std::ostream& os, const std::string* data, Index count,
    uint32_t /*compression*/) ///< @todo add compression
{
    for (Index i = 0; i < count; ++i) {
        const size_t len = data[i].size();
        os << len;
        os.write(data[i].c_str(), len+1);
        //os.write(&(data[i][0]), len );
    }
}

/// HalfWriter wraps a static function, write(), that is analogous to writeData(), above,
/// except that it is partially specialized for floating-point types in order to quantize
/// floating-point values to 16-bit half float.  A wrapper class is required because
/// only classes, not functions, can be partially specialized.
template<bool IsReal, typename T> struct HalfWriter;
/// Partial specialization for non-floating-point types (no float to half quantization)
template<typename T>
struct HalfWriter</*IsReal=*/false, T> {
    static inline void write(std::ostream& os, const T* data, Index count, uint32_t compression) {
        writeData(os, data, count, compression);
    }
};
/// Partial specialization for floating-point types
template<typename T>
struct HalfWriter</*IsReal=*/true, T> {
    using HalfT = typename RealToHalf<T>::HalfT;
    static inline void write(std::ostream& os, const T* data, Index count, uint32_t compression) {
        if (count < 1) return;
        // Convert full float values to half float, then output the half float array.
        std::vector<HalfT> halfData(count);
        for (Index i = 0; i < count; ++i) halfData[i] = RealToHalf<T>::convert(data[i]);
        writeData<HalfT>(os, reinterpret_cast<const HalfT*>(&halfData[0]), count, compression);
    }
};
#ifdef _MSC_VER
/// Specialization to avoid double to float warnings in MSVC
template<>
struct HalfWriter</*IsReal=*/true, double> {
    using HalfT = RealToHalf<double>::HalfT;
    static inline void write(std::ostream& os, const double* data, Index count,
        uint32_t compression)
    {
        if (count < 1) return;
        // Convert full float values to half float, then output the half float array.
        std::vector<HalfT> halfData(count);
        for (Index i = 0; i < count; ++i) halfData[i] = RealToHalf<double>::convert(data[i]);
        writeData<HalfT>(os, reinterpret_cast<const HalfT*>(&halfData[0]), count, compression);
    }
};
#endif // _MSC_VER


////////////////////////////////////////


/// Populate the given buffer with @a destCount values of type @c ValueT
/// read from the given stream, taking into account that the stream might
/// have been compressed via one of several supported schemes.
/// [Mainly for internal use]
/// @param is         a stream from which to read data (possibly compressed,
///                   depending on the stream's compression settings)
/// @param destBuf    a buffer into which to read values of type @c ValueT
/// @param destCount  the number of values to be stored in the buffer
/// @param valueMask  a bitmask (typically, a node's value mask) indicating
///                   which positions in the buffer correspond to active values
/// @param fromHalf   if true, read 16-bit half floats from the input stream
///                   and convert them to full floats
template<typename ValueT, typename MaskT>
inline void
readCompressedValues(std::istream& is, ValueT* destBuf, Index destCount,
    const MaskT& valueMask, bool fromHalf)
{
    // Get the stream's compression settings.
    const uint32_t compression = getDataCompression(is);
    const bool maskCompressed = compression & COMPRESS_ACTIVE_MASK;

    const bool seek = (destBuf == nullptr);
    assert(!seek || (!getStreamMetadataPtr(is) || getStreamMetadataPtr(is)->seekable()));

    int8_t metadata = NO_MASK_AND_ALL_VALS;
    if (getFormatVersion(is) >= OPENVDB_FILE_VERSION_NODE_MASK_COMPRESSION) {
        // Read the flag that specifies what, if any, additional metadata
        // (selection mask and/or inactive value(s)) is saved.
        if (seek && !maskCompressed) {
            is.seekg(/*bytes=*/1, std::ios_base::cur);
        } else {
            is.read(reinterpret_cast<char*>(&metadata), /*bytes=*/1);
        }
    }

    ValueT background = zeroVal<ValueT>();
    if (const void* bgPtr = getGridBackgroundValuePtr(is)) {
        background = *static_cast<const ValueT*>(bgPtr);
    }
    ValueT inactiveVal1 = background;
    ValueT inactiveVal0 =
        ((metadata == NO_MASK_OR_INACTIVE_VALS) ? background : math::negative(background));

    if (metadata == NO_MASK_AND_ONE_INACTIVE_VAL ||
        metadata == MASK_AND_ONE_INACTIVE_VAL ||
        metadata == MASK_AND_TWO_INACTIVE_VALS)
    {
        // Read one of at most two distinct inactive values.
        if (seek) {
            is.seekg(/*bytes=*/sizeof(ValueT), std::ios_base::cur);
        } else {
            is.read(reinterpret_cast<char*>(&inactiveVal0), /*bytes=*/sizeof(ValueT));
        }
        if (metadata == MASK_AND_TWO_INACTIVE_VALS) {
            // Read the second of two distinct inactive values.
            if (seek) {
                is.seekg(/*bytes=*/sizeof(ValueT), std::ios_base::cur);
            } else {
                is.read(reinterpret_cast<char*>(&inactiveVal1), /*bytes=*/sizeof(ValueT));
            }
        }
    }

    MaskT selectionMask;
    if (metadata == MASK_AND_NO_INACTIVE_VALS ||
        metadata == MASK_AND_ONE_INACTIVE_VAL ||
        metadata == MASK_AND_TWO_INACTIVE_VALS)
    {
        // For use in mask compression (only), read the bitmask that selects
        // between two distinct inactive values.
        if (seek) {
            is.seekg(/*bytes=*/selectionMask.memUsage(), std::ios_base::cur);
        } else {
            selectionMask.load(is);
        }
    }

    ValueT* tempBuf = destBuf;
    std::unique_ptr<ValueT[]> scopedTempBuf;

    Index tempCount = destCount;

    if (maskCompressed && metadata != NO_MASK_AND_ALL_VALS
        && getFormatVersion(is) >= OPENVDB_FILE_VERSION_NODE_MASK_COMPRESSION)
    {
        tempCount = valueMask.countOn();
        if (!seek && tempCount != destCount) {
            // If this node has inactive voxels, allocate a temporary buffer
            // into which to read just the active values.
            scopedTempBuf.reset(new ValueT[tempCount]);
            tempBuf = scopedTempBuf.get();
        }
    }

    // Read in the buffer.
    if (fromHalf) {
        HalfReader<RealToHalf<ValueT>::isReal, ValueT>::read(
            is, (seek ? nullptr : tempBuf), tempCount, compression);
    } else {
        readData<ValueT>(is, (seek ? nullptr : tempBuf), tempCount, compression);
    }

    // If mask compression is enabled and the number of active values read into
    // the temp buffer is smaller than the size of the destination buffer,
    // then there are missing (inactive) values.
    if (!seek && maskCompressed && tempCount != destCount) {
        // Restore inactive values, using the background value and, if available,
        // the inside/outside mask.  (For fog volumes, the destination buffer is assumed
        // to be initialized to background value zero, so inactive values can be ignored.)
        for (Index destIdx = 0, tempIdx = 0; destIdx < MaskT::SIZE; ++destIdx) {
            if (valueMask.isOn(destIdx)) {
                // Copy a saved active value into this node's buffer.
                destBuf[destIdx] = tempBuf[tempIdx];
                ++tempIdx;
            } else {
                // Reconstruct an unsaved inactive value and copy it into this node's buffer.
                destBuf[destIdx] = (selectionMask.isOn(destIdx) ? inactiveVal1 : inactiveVal0);
            }
        }
    }
}


/// Write @a srcCount values of type @c ValueT to the given stream, optionally
/// after compressing the values via one of several supported schemes.
/// [Mainly for internal use]
/// @param os         a stream to which to write data (possibly compressed, depending
///                   on the stream's compression settings)
/// @param srcBuf     a buffer containing values of type @c ValueT to be written
/// @param srcCount   the number of values stored in the buffer
/// @param valueMask  a bitmask (typically, a node's value mask) indicating
///                   which positions in the buffer correspond to active values
/// @param childMask  a bitmask (typically, a node's child mask) indicating
///                   which positions in the buffer correspond to child node pointers
/// @param toHalf     if true, convert floating-point values to 16-bit half floats
template<typename ValueT, typename MaskT>
inline void
writeCompressedValues(std::ostream& os, ValueT* srcBuf, Index srcCount,
    const MaskT& valueMask, const MaskT& childMask, bool toHalf)
{
    struct Local {
        // Comparison function for values
        static inline bool eq(const ValueT& a, const ValueT& b) {
            return math::isExactlyEqual(a, b);
        }
    };

    // Get the stream's compression settings.
    const uint32_t compress = getDataCompression(os);
    const bool maskCompress = compress & COMPRESS_ACTIVE_MASK;

    Index tempCount = srcCount;
    ValueT* tempBuf = srcBuf;
    std::unique_ptr<ValueT[]> scopedTempBuf;

    int8_t metadata = NO_MASK_AND_ALL_VALS;

    if (!maskCompress) {
        os.write(reinterpret_cast<const char*>(&metadata), /*bytes=*/1);
    } else {
        // A valid level set's inactive values are either +background (outside)
        // or -background (inside), and a fog volume's inactive values are all zero.
        // Rather than write out all of these values, we can store just the active values
        // (given that the value mask specifies their positions) and, if necessary,
        // an inside/outside bitmask.

        const ValueT zero = zeroVal<ValueT>();
        ValueT background = zero;
        if (const void* bgPtr = getGridBackgroundValuePtr(os)) {
            background = *static_cast<const ValueT*>(bgPtr);
        }

        /// @todo Consider all values, not just inactive values?
        ValueT inactiveVal[2] = { background, background };
        int numUniqueInactiveVals = 0;
        for (typename MaskT::OffIterator it = valueMask.beginOff();
            numUniqueInactiveVals < 3 && it; ++it)
        {
            const Index32 idx = it.pos();

            // Skip inactive values that are actually child node pointers.
            if (childMask.isOn(idx)) continue;

            const ValueT& val = srcBuf[idx];
            const bool unique = !(
                (numUniqueInactiveVals > 0 && Local::eq(val, inactiveVal[0])) ||
                (numUniqueInactiveVals > 1 && Local::eq(val, inactiveVal[1]))
            );
            if (unique) {
                if (numUniqueInactiveVals < 2) inactiveVal[numUniqueInactiveVals] = val;
                ++numUniqueInactiveVals;
            }
        }

        metadata = NO_MASK_OR_INACTIVE_VALS;

        if (numUniqueInactiveVals == 1) {
            if (!Local::eq(inactiveVal[0], background)) {
                if (Local::eq(inactiveVal[0], math::negative(background))) {
                    metadata = NO_MASK_AND_MINUS_BG;
                } else {
                    metadata = NO_MASK_AND_ONE_INACTIVE_VAL;
                }
            }
        } else if (numUniqueInactiveVals == 2) {
            metadata = NO_MASK_OR_INACTIVE_VALS;
            if (!Local::eq(inactiveVal[0], background) && !Local::eq(inactiveVal[1], background)) {
                // If neither inactive value is equal to the background, both values
                // need to be saved, along with a mask that selects between them.
                metadata = MASK_AND_TWO_INACTIVE_VALS;

            } else if (Local::eq(inactiveVal[1], background)) {
                if (Local::eq(inactiveVal[0], math::negative(background))) {
                    // If the second inactive value is equal to the background and
                    // the first is equal to -background, neither value needs to be saved,
                    // but save a mask that selects between -background and +background.
                    metadata = MASK_AND_NO_INACTIVE_VALS;
                } else {
                    // If the second inactive value is equal to the background, only
                    // the first value needs to be saved, along with a mask that selects
                    // between it and the background.
                    metadata = MASK_AND_ONE_INACTIVE_VAL;
                }
            } else if (Local::eq(inactiveVal[0], background)) {
                if (Local::eq(inactiveVal[1], math::negative(background))) {
                    // If the first inactive value is equal to the background and
                    // the second is equal to -background, neither value needs to be saved,
                    // but save a mask that selects between -background and +background.
                    metadata = MASK_AND_NO_INACTIVE_VALS;
                    std::swap(inactiveVal[0], inactiveVal[1]);
                } else {
                    // If the first inactive value is equal to the background, swap it
                    // with the second value and save only that value, along with a mask
                    // that selects between it and the background.
                    std::swap(inactiveVal[0], inactiveVal[1]);
                    metadata = MASK_AND_ONE_INACTIVE_VAL;
                }
            }
        } else if (numUniqueInactiveVals > 2) {
            metadata = NO_MASK_AND_ALL_VALS;
        }

        os.write(reinterpret_cast<const char*>(&metadata), /*bytes=*/1);

        if (metadata == NO_MASK_AND_ONE_INACTIVE_VAL ||
            metadata == MASK_AND_ONE_INACTIVE_VAL ||
            metadata == MASK_AND_TWO_INACTIVE_VALS)
        {
            if (!toHalf) {
                // Write one of at most two distinct inactive values.
                os.write(reinterpret_cast<const char*>(&inactiveVal[0]), sizeof(ValueT));
                if (metadata == MASK_AND_TWO_INACTIVE_VALS) {
                    // Write the second of two distinct inactive values.
                    os.write(reinterpret_cast<const char*>(&inactiveVal[1]), sizeof(ValueT));
                }
            } else {
                // Write one of at most two distinct inactive values.
                ValueT truncatedVal = static_cast<ValueT>(truncateRealToHalf(inactiveVal[0]));
                os.write(reinterpret_cast<const char*>(&truncatedVal), sizeof(ValueT));
                if (metadata == MASK_AND_TWO_INACTIVE_VALS) {
                    // Write the second of two distinct inactive values.
                    truncatedVal = truncateRealToHalf(inactiveVal[1]);
                    os.write(reinterpret_cast<const char*>(&truncatedVal), sizeof(ValueT));
                }
            }
        }

        if (metadata == NO_MASK_AND_ALL_VALS) {
            // If there are more than two unique inactive values, the entire input buffer
            // needs to be saved (both active and inactive values).
            /// @todo Save the selection mask as long as most of the inactive values
            /// are one of two values?
        } else {
            // Create a new array to hold just the active values.
            scopedTempBuf.reset(new ValueT[srcCount]);
            tempBuf = scopedTempBuf.get();

            if (metadata == NO_MASK_OR_INACTIVE_VALS ||
                metadata == NO_MASK_AND_MINUS_BG ||
                metadata == NO_MASK_AND_ONE_INACTIVE_VAL)
            {
                // Copy active values to the contiguous array.
                tempCount = 0;
                for (typename MaskT::OnIterator it = valueMask.beginOn(); it; ++it, ++tempCount) {
                    tempBuf[tempCount] = srcBuf[it.pos()];
                }
            } else {
                // Copy active values to a new, contiguous array and populate a bitmask
                // that selects between two distinct inactive values.
                MaskT selectionMask;
                tempCount = 0;
                for (Index srcIdx = 0; srcIdx < srcCount; ++srcIdx) {
                    if (valueMask.isOn(srcIdx)) { // active value
                        tempBuf[tempCount] = srcBuf[srcIdx];
                        ++tempCount;
                    } else { // inactive value
                        if (Local::eq(srcBuf[srcIdx], inactiveVal[1])) {
                            selectionMask.setOn(srcIdx); // inactive value 1
                        } // else inactive value 0
                    }
                }
                assert(tempCount == valueMask.countOn());

                // Write out the mask that selects between two inactive values.
                selectionMask.save(os);
            }
        }
    }

    // Write out the buffer.
    if (toHalf) {
        HalfWriter<RealToHalf<ValueT>::isReal, ValueT>::write(os, tempBuf, tempCount, compress);
    } else {
        writeData(os, tempBuf, tempCount, compress);
    }
}

} // namespace io
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_IO_COMPRESSION_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
