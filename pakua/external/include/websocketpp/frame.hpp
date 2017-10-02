/*
 * Copyright (c) 2014, Peter Thorson. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the WebSocket++ Project nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL PETER THORSON BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

#ifndef WEBSOCKETPP_FRAME_HPP
#define WEBSOCKETPP_FRAME_HPP

#include <algorithm>
#include <string>

#include <websocketpp/common/system_error.hpp>
#include <websocketpp/common/network.hpp>

#include <websocketpp/utilities.hpp>

namespace websocketpp {
/// Data structures and utility functions for manipulating WebSocket frames
/**
 * namespace frame provides a number of data structures and utility functions
 * for reading, writing, and manipulating binary encoded WebSocket frames.
 */
namespace frame {

/// Minimum length of a WebSocket frame header.
static unsigned int const BASIC_HEADER_LENGTH = 2;
/// Maximum length of a WebSocket header
static unsigned int const MAX_HEADER_LENGTH = 14;
/// Maximum length of the variable portion of the WebSocket header
static unsigned int const MAX_EXTENDED_HEADER_LENGTH = 12;

/// Two byte conversion union
union uint16_converter {
    uint16_t i;
    uint8_t  c[2];
};

/// Four byte conversion union
union uint32_converter {
    uint32_t i;
    uint8_t c[4];
};

/// Eight byte conversion union
union uint64_converter {
    uint64_t i;
    uint8_t  c[8];
};

/// Constants and utility functions related to WebSocket opcodes
/**
 * WebSocket Opcodes are 4 bits. See RFC6455 section 5.2.
 */
namespace opcode {
    enum value {
        continuation = 0x0,
        text = 0x1,
        binary = 0x2,
        rsv3 = 0x3,
        rsv4 = 0x4,
        rsv5 = 0x5,
        rsv6 = 0x6,
        rsv7 = 0x7,
        close = 0x8,
        ping = 0x9,
        pong = 0xA,
        control_rsvb = 0xB,
        control_rsvc = 0xC,
        control_rsvd = 0xD,
        control_rsve = 0xE,
        control_rsvf = 0xF,

        CONTINUATION = 0x0,
        TEXT = 0x1,
        BINARY = 0x2,
        RSV3 = 0x3,
        RSV4 = 0x4,
        RSV5 = 0x5,
        RSV6 = 0x6,
        RSV7 = 0x7,
        CLOSE = 0x8,
        PING = 0x9,
        PONG = 0xA,
        CONTROL_RSVB = 0xB,
        CONTROL_RSVC = 0xC,
        CONTROL_RSVD = 0xD,
        CONTROL_RSVE = 0xE,
        CONTROL_RSVF = 0xF
    };

    /// Check if an opcode is reserved
    /**
     * @param v The opcode to test.
     * @return Whether or not the opcode is reserved.
     */
    inline bool reserved(value v) {
        return (v >= rsv3 && v <= rsv7) ||
               (v >= control_rsvb && v <= control_rsvf);
    }

    /// Check if an opcode is invalid
    /**
     * Invalid opcodes are negative or require greater than 4 bits to store.
     *
     * @param v The opcode to test.
     * @return Whether or not the opcode is invalid.
     */
    inline bool invalid(value v) {
        return (v > 0xF || v < 0);
    }

    /// Check if an opcode is for a control frame
    /**
     * @param v The opcode to test.
     * @return Whether or not the opcode is a control opcode.
     */
    inline bool is_control(value v) {
        return v >= 0x8;
    }
}

/// Constants related to frame and payload limits
namespace limits {
    /// Minimum length of a WebSocket frame header.
    static unsigned int const basic_header_length = 2;

    /// Maximum length of a WebSocket header
    static unsigned int const max_header_length = 14;

    /// Maximum length of the variable portion of the WebSocket header
    static unsigned int const max_extended_header_length = 12;

    /// Maximum size of a basic WebSocket payload
    static uint8_t const payload_size_basic = 125;

    /// Maximum size of an extended WebSocket payload (basic payload = 126)
    static uint16_t const payload_size_extended = 0xFFFF; // 2^16, 65535

    /// Maximum size of a jumbo WebSocket payload (basic payload = 127)
    static uint64_t const payload_size_jumbo = 0x7FFFFFFFFFFFFFFFLL;//2^63

    /// Maximum size of close frame reason
    /**
     * This is payload_size_basic - 2 bytes (as first two bytes are used for
     * the close code
     */
    static uint8_t const close_reason_size = 123;
}


// masks for fields in the basic header
static uint8_t const BHB0_OPCODE = 0x0F;
static uint8_t const BHB0_RSV3 = 0x10;
static uint8_t const BHB0_RSV2 = 0x20;
static uint8_t const BHB0_RSV1 = 0x40;
static uint8_t const BHB0_FIN = 0x80;

static uint8_t const BHB1_PAYLOAD = 0x7F;
static uint8_t const BHB1_MASK = 0x80;

static uint8_t const payload_size_code_16bit = 0x7E; // 126
static uint8_t const payload_size_code_64bit = 0x7F; // 127

typedef uint32_converter masking_key_type;

/// The constant size component of a WebSocket frame header
struct basic_header {
    basic_header() : b0(0x00),b1(0x00) {}

    basic_header(uint8_t p0, uint8_t p1) : b0(p0), b1(p1) {}

    basic_header(opcode::value op, uint64_t size, bool fin, bool mask,
        bool rsv1 = false, bool rsv2 = false, bool rsv3 = false) : b0(0x00),
        b1(0x00)
    {
        if (fin) {
            b0 |= BHB0_FIN;
        }
        if (rsv1) {
            b0 |= BHB0_RSV1;
        }
        if (rsv2) {
            b0 |= BHB0_RSV2;
        }
        if (rsv3) {
            b0 |= BHB0_RSV3;
        }
        b0 |= (op & BHB0_OPCODE);

        if (mask) {
            b1 |= BHB1_MASK;
        }

        uint8_t basic_value;

        if (size <= limits::payload_size_basic) {
            basic_value = static_cast<uint8_t>(size);
        } else if (size <= limits::payload_size_extended) {
            basic_value = payload_size_code_16bit;
        } else {
            basic_value = payload_size_code_64bit;
        }


        b1 |= basic_value;
    }

    uint8_t b0;
    uint8_t b1;
};

/// The variable size component of a WebSocket frame header
struct extended_header {
    extended_header() {
        std::fill_n(this->bytes,MAX_EXTENDED_HEADER_LENGTH,0x00);
    }

    extended_header(uint64_t payload_size) {
        std::fill_n(this->bytes,MAX_EXTENDED_HEADER_LENGTH,0x00);

        copy_payload(payload_size);
    }

    extended_header(uint64_t payload_size, uint32_t masking_key) {
        std::fill_n(this->bytes,MAX_EXTENDED_HEADER_LENGTH,0x00);

        // Copy payload size
        int offset = copy_payload(payload_size);

        // Copy Masking Key
        uint32_converter temp32;
        temp32.i = masking_key;
        std::copy(temp32.c,temp32.c+4,bytes+offset);
    }

    uint8_t bytes[MAX_EXTENDED_HEADER_LENGTH];
private:
    int copy_payload(uint64_t payload_size) {
        int payload_offset = 0;

        if (payload_size <= limits::payload_size_basic) {
            payload_offset = 8;
        } else if (payload_size <= limits::payload_size_extended) {
            payload_offset = 6;
        }

        uint64_converter temp64;
        temp64.i = lib::net::_htonll(payload_size);
        std::copy(temp64.c+payload_offset,temp64.c+8,bytes);

        return 8-payload_offset;
    }
};

bool get_fin(basic_header const &h);
void set_fin(basic_header &h, bool value);
bool get_rsv1(basic_header const &h);
void set_rsv1(basic_header &h, bool value);
bool get_rsv2(basic_header const &h);
void set_rsv2(basic_header &h, bool value);
bool get_rsv3(basic_header const &h);
void set_rsv3(basic_header &h, bool value);
opcode::value get_opcode(basic_header const &h);
bool get_masked(basic_header const &h);
void set_masked(basic_header &h, bool value);
uint8_t get_basic_size(basic_header const &);
size_t get_header_len(basic_header const &);
unsigned int get_masking_key_offset(basic_header const &);

std::string write_header(basic_header const &, extended_header const &);
masking_key_type get_masking_key(basic_header const &, extended_header const &);
uint16_t get_extended_size(extended_header const &);
uint64_t get_jumbo_size(extended_header const &);
uint64_t get_payload_size(basic_header const &, extended_header const &);

size_t prepare_masking_key(masking_key_type const & key);
size_t circshift_prepared_key(size_t prepared_key, size_t offset);

// Functions for performing xor based masking and unmasking
template <typename input_iter, typename output_iter>
void byte_mask(input_iter b, input_iter e, output_iter o, masking_key_type
    const & key, size_t key_offset = 0);
template <typename iter_type>
void byte_mask(iter_type b, iter_type e, masking_key_type const & key,
    size_t key_offset = 0);
void word_mask_exact(uint8_t * input, uint8_t * output, size_t length,
    masking_key_type const & key);
void word_mask_exact(uint8_t * data, size_t length, masking_key_type const &
    key);
size_t word_mask_circ(uint8_t * input, uint8_t * output, size_t length,
    size_t prepared_key);
size_t word_mask_circ(uint8_t * data, size_t length, size_t prepared_key);

/// Check whether the frame's FIN bit is set.
/**
 * @param [in] h The basic header to extract from.
 * @return True if the header's fin bit is set.
 */
inline bool get_fin(basic_header const & h) {
    return ((h.b0 & BHB0_FIN) == BHB0_FIN);
}

/// Set the frame's FIN bit
/**
 * @param [out] h Header to set.
 * @param [in] value Value to set it to.
 */
inline void set_fin(basic_header & h, bool value) {
    h.b0 = (value ? h.b0 | BHB0_FIN : h.b0 & ~BHB0_FIN);
}

/// check whether the frame's RSV1 bit is set
/**
 * @param [in] h The basic header to extract from.
 * @return True if the header's RSV1 bit is set.
 */
inline bool get_rsv1(const basic_header &h) {
    return ((h.b0 & BHB0_RSV1) == BHB0_RSV1);
}

/// Set the frame's RSV1 bit
/**
 * @param [out] h Header to set.
 * @param [in] value Value to set it to.
 */
inline void set_rsv1(basic_header &h, bool value) {
    h.b0 = (value ? h.b0 | BHB0_RSV1 : h.b0 & ~BHB0_RSV1);
}

/// check whether the frame's RSV2 bit is set
/**
 * @param [in] h The basic header to extract from.
 * @return True if the header's RSV2 bit is set.
 */
inline bool get_rsv2(const basic_header &h) {
    return ((h.b0 & BHB0_RSV2) == BHB0_RSV2);
}

/// Set the frame's RSV2 bit
/**
 * @param [out] h Header to set.
 * @param [in] value Value to set it to.
 */
inline void set_rsv2(basic_header &h, bool value) {
    h.b0 = (value ? h.b0 | BHB0_RSV2 : h.b0 & ~BHB0_RSV2);
}

/// check whether the frame's RSV3 bit is set
/**
 * @param [in] h The basic header to extract from.
 * @return True if the header's RSV3 bit is set.
 */
inline bool get_rsv3(const basic_header &h) {
    return ((h.b0 & BHB0_RSV3) == BHB0_RSV3);
}

/// Set the frame's RSV3 bit
/**
 * @param [out] h Header to set.
 * @param [in] value Value to set it to.
 */
inline void set_rsv3(basic_header &h, bool value) {
    h.b0 = (value ? h.b0 | BHB0_RSV3 : h.b0 & ~BHB0_RSV3);
}

/// Extract opcode from basic header
/**
 * @param [in] h The basic header to extract from.
 * @return The opcode value of the header.
 */
inline opcode::value get_opcode(const basic_header &h) {
    return opcode::value(h.b0 & BHB0_OPCODE);
}

/// check whether the frame is masked
/**
 * @param [in] h The basic header to extract from.
 * @return True if the header mask bit is set.
 */
inline bool get_masked(basic_header const & h) {
    return ((h.b1 & BHB1_MASK) == BHB1_MASK);
}

/// Set the frame's MASK bit
/**
 * @param [out] h Header to set.
 * @param value Value to set it to.
 */
inline void set_masked(basic_header & h, bool value) {
    h.b1 = (value ? h.b1 | BHB1_MASK : h.b1 & ~BHB1_MASK);
}

/// Extracts the raw payload length specified in the basic header
/**
 * A basic WebSocket frame header contains a 7 bit value that represents the
 * payload size. There are two reserved values that are used to indicate that
 * the actual payload size will not fit in 7 bits and that the full payload
 * size is included in a separate field. The values are as follows:
 *
 * PAYLOAD_SIZE_CODE_16BIT (0x7E) indicates that the actual payload is less
 * than 16 bit
 *
 * PAYLOAD_SIZE_CODE_64BIT (0x7F) indicates that the actual payload is less
 * than 63 bit
 *
 * @param [in] h Basic header to read value from.
 * @return The exact size encoded in h.
 */
inline uint8_t get_basic_size(const basic_header &h) {
    return h.b1 & BHB1_PAYLOAD;
}

/// Calculates the full length of the header based on the first bytes.
/**
 * A WebSocket frame header always has at least two bytes. Encoded within the
 * first two bytes is all the information necessary to calculate the full
 * (variable) header length. get_header_len() calculates the full header
 * length for the given two byte basic header.
 *
 * @param h Basic frame header to extract size from.
 * @return Full length of the extended header.
 */
inline size_t get_header_len(basic_header const & h) {
    // TODO: check extensions?

    // masking key offset represents the space used for the extended length
    // fields
    size_t size = BASIC_HEADER_LENGTH + get_masking_key_offset(h);

    // If the header is masked there is a 4 byte masking key
    if (get_masked(h)) {
        size += 4;
    }

    return size;
}

/// Calculate the offset location of the masking key within the extended header
/**
 * Calculate the offset location of the masking key within the extended header
 * using information from its corresponding basic header
 *
 * @param h Corresponding basic header to calculate from.
 *
 * @return byte offset of the first byte of the masking key
 */
inline unsigned int get_masking_key_offset(const basic_header &h) {
    if (get_basic_size(h) == payload_size_code_16bit) {
        return 2;
    } else if (get_basic_size(h) == payload_size_code_64bit) {
        return 8;
    } else {
        return 0;
    }
}

/// Generate a properly sized contiguous string that encodes a full frame header
/**
 * Copy the basic header h and extended header e into a properly sized
 * contiguous frame header string for the purposes of writing out to the wire.
 *
 * @param h The basic header to include
 * @param e The extended header to include
 *
 * @return A contiguous string containing h and e
 */
inline std::string prepare_header(const basic_header &h, const
    extended_header &e)
{
    std::string ret;

    ret.push_back(char(h.b0));
    ret.push_back(char(h.b1));
    ret.append(
        reinterpret_cast<const char*>(e.bytes),
        get_header_len(h)-BASIC_HEADER_LENGTH
    );

    return ret;
}

/// Extract the masking key from a frame header
/**
 * Note that while read and written as an integer at times, this value is not
 * an integer and should never be interpreted as one. Big and little endian
 * machines will generate and store masking keys differently without issue as
 * long as the integer values remain irrelivant.
 *
 * @param h The basic header to extract from
 * @param e The extended header to extract from
 *
 * @return The masking key as an integer.
 */
inline masking_key_type get_masking_key(const basic_header &h, const
    extended_header &e)
{
    masking_key_type temp32;

    if (!get_masked(h)) {
        temp32.i = 0;
    } else {
        unsigned int offset = get_masking_key_offset(h);
        std::copy(e.bytes+offset,e.bytes+offset+4,temp32.c);
    }

    return temp32;
}

/// Extract the extended size field from an extended header
/**
 * It is the responsibility of the caller to verify that e is a valid extended
 * header. This function assumes that e contains an extended payload size.
 *
 * @param e The extended header to extract from
 *
 * @return The size encoded in the extended header in host byte order
 */
inline uint16_t get_extended_size(const extended_header &e) {
    uint16_converter temp16;
    std::copy(e.bytes,e.bytes+2,temp16.c);
    return ntohs(temp16.i);
}

/// Extract the jumbo size field from an extended header
/**
 * It is the responsibility of the caller to verify that e is a valid extended
 * header. This function assumes that e contains a jumbo payload size.
 *
 * @param e The extended header to extract from
 *
 * @return The size encoded in the extended header in host byte order
 */
inline uint64_t get_jumbo_size(const extended_header &e) {
    uint64_converter temp64;
    std::copy(e.bytes,e.bytes+8,temp64.c);
    return lib::net::_ntohll(temp64.i);
}

/// Extract the full payload size field from a WebSocket header
/**
 * It is the responsibility of the caller to verify that h and e together
 * represent a valid WebSocket frame header. This function assumes only that h
 * and e are valid. It uses information in the basic header to determine where
 * to look for the payload_size
 *
 * @param h The basic header to extract from
 * @param e The extended header to extract from
 *
 * @return The size encoded in the combined header in host byte order.
 */
inline uint64_t get_payload_size(const basic_header &h, const
    extended_header &e)
{
    uint8_t val = get_basic_size(h);

    if (val <= limits::payload_size_basic) {
        return val;
    } else if (val == payload_size_code_16bit) {
        return get_extended_size(e);
    } else {
        return get_jumbo_size(e);
    }
}

/// Extract a masking key into a value the size of a machine word.
/**
 * Machine word size must be 4 or 8.
 *
 * @param key Masking key to extract from
 *
 * @return prepared key as a machine word
 */
inline size_t prepare_masking_key(const masking_key_type& key) {
    size_t low_bits = static_cast<size_t>(key.i);

    if (sizeof(size_t) == 8) {
        uint64_t high_bits = static_cast<size_t>(key.i);
        return static_cast<size_t>((high_bits << 32) | low_bits);
    } else {
        return low_bits;
    }
}

/// circularly shifts the supplied prepared masking key by offset bytes
/**
 * Prepared_key must be the output of prepare_masking_key with the associated
 * restrictions on the machine word size. offset must be greater than or equal
 * to zero and less than sizeof(size_t).
 */
inline size_t circshift_prepared_key(size_t prepared_key, size_t offset) {
    if (lib::net::is_little_endian()) {
        size_t temp = prepared_key << (sizeof(size_t)-offset)*8;
        return (prepared_key >> offset*8) | temp;
    } else {
        size_t temp = prepared_key >> (sizeof(size_t)-offset)*8;
        return (prepared_key << offset*8) | temp;
    }
}

/// Byte by byte mask/unmask
/**
 * Iterator based byte by byte masking and unmasking for WebSocket payloads.
 * Performs masking in place using the supplied key offset by the supplied
 * offset number of bytes.
 *
 * This function is simple and can be done in place on input with arbitrary
 * lengths and does not vary based on machine word size. It is slow.
 *
 * @param b Beginning iterator to start masking
 *
 * @param e Ending iterator to end masking
 *
 * @param o Beginning iterator to store masked results
 *
 * @param key 32 bit key to mask with.
 *
 * @param key_offset offset value to start masking at.
 */
template <typename input_iter, typename output_iter>
void byte_mask(input_iter first, input_iter last, output_iter result,
    masking_key_type const & key, size_t key_offset)
{
    size_t key_index = key_offset%4;
    while (first != last) {
        *result = *first ^ key.c[key_index++];
        key_index %= 4;
        ++result;
        ++first;
    }
}

/// Byte by byte mask/unmask (in place)
/**
 * Iterator based byte by byte masking and unmasking for WebSocket payloads.
 * Performs masking in place using the supplied key offset by the supplied
 * offset number of bytes.
 *
 * This function is simple and can be done in place on input with arbitrary
 * lengths and does not vary based on machine word size. It is slow.
 *
 * @param b Beginning iterator to start masking
 *
 * @param e Ending iterator to end masking
 *
 * @param key 32 bit key to mask with.
 *
 * @param key_offset offset value to start masking at.
 */
template <typename iter_type>
void byte_mask(iter_type b, iter_type e, masking_key_type const & key,
    size_t key_offset)
{
    byte_mask(b,e,b,key,key_offset);
}

/// Exact word aligned mask/unmask
/**
 * Balanced combination of byte by byte and circular word by word masking.
 * Best used to mask complete messages at once. Has much higher setup costs than
 * word_mask_circ but works with exact sized buffers.
 *
 * Buffer based word by word masking and unmasking for WebSocket payloads.
 * Masking is done in word by word chunks with the remainder not divisible by
 * the word size done byte by byte.
 *
 * input and output must both be at least length bytes. Exactly length bytes
 * will be written.
 *
 * @param input buffer to mask or unmask
 *
 * @param output buffer to store the output. May be the same as input.
 *
 * @param length length of data buffer
 *
 * @param key Masking key to use
 */
inline void word_mask_exact(uint8_t* input, uint8_t* output, size_t length,
    const masking_key_type& key)
{
    size_t prepared_key = prepare_masking_key(key);
    size_t n = length/sizeof(size_t);
    size_t* input_word = reinterpret_cast<size_t*>(input);
    size_t* output_word = reinterpret_cast<size_t*>(output);

    for (size_t i = 0; i < n; i++) {
        output_word[i] = input_word[i] ^ prepared_key;
    }

    for (size_t i = n*sizeof(size_t); i < length; i++) {
        output[i] = input[i] ^ key.c[i%4];
    }
}

/// Exact word aligned mask/unmask (in place)
/**
 * In place version of word_mask_exact
 *
 * @see word_mask_exact
 *
 * @param data buffer to read and write from
 *
 * @param length length of data buffer
 *
 * @param key Masking key to use
 */
inline void word_mask_exact(uint8_t* data, size_t length, const
    masking_key_type& key)
{
    word_mask_exact(data,data,length,key);
}

/// Circular word aligned mask/unmask
/**
 * Performs a circular mask/unmask in word sized chunks using pre-prepared keys
 * that store state between calls. Best for providing streaming masking or
 * unmasking of small chunks at a time of a larger message. Requires that the
 * underlying allocated size of the data buffer be a multiple of the word size.
 * Data in the buffer after `length` will be overwritten only with the same
 * values that were originally present.
 *
 * Buffer based word by word masking and unmasking for WebSocket payloads.
 * Performs masking in place using the supplied key. Casts the data buffer to
 * an array of size_t's and performs masking word by word. The underlying
 * buffer size must be a muliple of the word size.
 *
 * word_mask returns a copy of prepared_key circularly shifted based on the
 * length value. The returned value may be fed back into word_mask when more
 * data is available.
 *
 * input and output must both have length at least:
 *    ceil(length/sizeof(size_t))*sizeof(size_t)
 * Exactly that many bytes will be written, although only exactly length bytes
 * will be changed (trailing bytes will be replaced without masking)
 *
 * @param data Character buffer to mask
 *
 * @param length Length of data
 *
 * @param prepared_key Prepared key to use.
 *
 * @return the prepared_key shifted to account for the input length
 */
inline size_t word_mask_circ(uint8_t * input, uint8_t * output, size_t length,
    size_t prepared_key)
{
    size_t n = length / sizeof(size_t); // whole words
    size_t l = length - (n * sizeof(size_t)); // remaining bytes
    size_t * input_word = reinterpret_cast<size_t *>(input);
    size_t * output_word = reinterpret_cast<size_t *>(output);

    // mask word by word
    for (size_t i = 0; i < n; i++) {
        output_word[i] = input_word[i] ^ prepared_key;
    }

    // mask partial word at the end
    size_t start = length - l;
    uint8_t * byte_key = reinterpret_cast<uint8_t *>(&prepared_key);
    for (size_t i = 0; i < l; ++i) {
        output[start+i] = input[start+i] ^ byte_key[i];
    }

    return circshift_prepared_key(prepared_key,l);
}

/// Circular word aligned mask/unmask (in place)
/**
 * In place version of word_mask_circ
 *
 * @see word_mask_circ
 *
 * @param data Character buffer to read from and write to
 *
 * @param length Length of data
 *
 * @param prepared_key Prepared key to use.
 *
 * @return the prepared_key shifted to account for the input length
 */
inline size_t word_mask_circ(uint8_t* data, size_t length, size_t prepared_key){
    return word_mask_circ(data,data,length,prepared_key);
}

/// Circular byte aligned mask/unmask
/**
 * Performs a circular mask/unmask in byte sized chunks using pre-prepared keys
 * that store state between calls. Best for providing streaming masking or
 * unmasking of small chunks at a time of a larger message. Requires that the
 * underlying allocated size of the data buffer be a multiple of the word size.
 * Data in the buffer after `length` will be overwritten only with the same
 * values that were originally present.
 *
 * word_mask returns a copy of prepared_key circularly shifted based on the
 * length value. The returned value may be fed back into byte_mask when more
 * data is available.
 *
 * @param data Character buffer to mask
 *
 * @param length Length of data
 *
 * @param prepared_key Prepared key to use.
 *
 * @return the prepared_key shifted to account for the input length
 */
inline size_t byte_mask_circ(uint8_t * input, uint8_t * output, size_t length,
    size_t prepared_key)
{
    uint32_converter key;
    key.i = prepared_key;

    for (size_t i = 0; i < length; ++i) {
        output[i] = input[i] ^ key.c[i % 4];
    }

    return circshift_prepared_key(prepared_key,length % 4);
}

/// Circular byte aligned mask/unmask (in place)
/**
 * In place version of byte_mask_circ
 *
 * @see byte_mask_circ
 *
 * @param data Character buffer to read from and write to
 *
 * @param length Length of data
 *
 * @param prepared_key Prepared key to use.
 *
 * @return the prepared_key shifted to account for the input length
 */
inline size_t byte_mask_circ(uint8_t* data, size_t length, size_t prepared_key){
    return byte_mask_circ(data,data,length,prepared_key);
}

} // namespace frame
} // namespace websocketpp

#endif //WEBSOCKETPP_FRAME_HPP
