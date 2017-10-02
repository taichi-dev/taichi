/*
 * The following code is adapted from code originally written by Bjoern
 * Hoehrmann <bjoern@hoehrmann.de>. See
 * http://bjoern.hoehrmann.de/utf-8/decoder/dfa/ for details.
 *
 * The original license:
 *
 * Copyright (c) 2008-2009 Bjoern Hoehrmann <bjoern@hoehrmann.de>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
*/

#ifndef UTF8_VALIDATOR_HPP
#define UTF8_VALIDATOR_HPP

#include <websocketpp/common/stdint.hpp>

#include <string>

namespace websocketpp {
namespace utf8_validator {

/// State that represents a valid utf8 input sequence
static unsigned int const utf8_accept = 0;
/// State that represents an invalid utf8 input sequence
static unsigned int const utf8_reject = 1;

/// Lookup table for the UTF8 decode state machine
static uint8_t const utf8d[] = {
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, // 00..1f
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, // 20..3f
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, // 40..5f
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, // 60..7f
  1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9, // 80..9f
  7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7, // a0..bf
  8,8,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2, // c0..df
  0xa,0x3,0x3,0x3,0x3,0x3,0x3,0x3,0x3,0x3,0x3,0x3,0x3,0x4,0x3,0x3, // e0..ef
  0xb,0x6,0x6,0x6,0x5,0x8,0x8,0x8,0x8,0x8,0x8,0x8,0x8,0x8,0x8,0x8, // f0..ff
  0x0,0x1,0x2,0x3,0x5,0x8,0x7,0x1,0x1,0x1,0x4,0x6,0x1,0x1,0x1,0x1, // s0..s0
  1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,0,1,0,1,1,1,1,1,1, // s1..s2
  1,2,1,1,1,1,1,2,1,2,1,1,1,1,1,1,1,1,1,1,1,1,1,2,1,1,1,1,1,1,1,1, // s3..s4
  1,2,1,1,1,1,1,1,1,2,1,1,1,1,1,1,1,1,1,1,1,1,1,3,1,3,1,1,1,1,1,1, // s5..s6
  1,3,1,1,1,1,1,3,1,3,1,1,1,1,1,1,1,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1, // s7..s8
};

/// Decode the next byte of a UTF8 sequence
/**
 * @param [out] state The decoder state to advance
 * @param [out] codep The codepoint to fill in
 * @param [in] byte The byte to input
 * @return The ending state of the decode operation
 */
inline uint32_t decode(uint32_t * state, uint32_t * codep, uint8_t byte) {
  uint32_t type = utf8d[byte];

  *codep = (*state != utf8_accept) ?
    (byte & 0x3fu) | (*codep << 6) :
    (0xff >> type) & (byte);

  *state = utf8d[256 + *state*16 + type];
  return *state;
}

/// Provides streaming UTF8 validation functionality
class validator {
public:
    /// Construct and initialize the validator
    validator() : m_state(utf8_accept),m_codepoint(0) {}

    /// Advance the state of the validator with the next input byte
    /**
     * @param byte The byte to advance the validation state with
     * @return Whether or not the byte resulted in a validation error.
     */
    bool consume (uint8_t byte) {
        if (utf8_validator::decode(&m_state,&m_codepoint,byte) == utf8_reject) {
            return false;
        }
        return true;
    }

    /// Advance validator state with input from an iterator pair
    /**
     * @param begin Input iterator to the start of the input range
     * @param end Input iterator to the end of the input range
     * @return Whether or not decoding the bytes resulted in a validation error.
     */
    template <typename iterator_type>
    bool decode (iterator_type begin, iterator_type end) {
        for (iterator_type it = begin; it != end; ++it) {
            unsigned int result = utf8_validator::decode(
                &m_state,
                &m_codepoint,
                static_cast<uint8_t>(*it)
            );

            if (result == utf8_reject) {
                return false;
            }
        }
        return true;
    }

    /// Return whether the input sequence ended on a valid utf8 codepoint
    /**
     * @return Whether or not the input sequence ended on a valid codepoint.
     */
    bool complete() {
        return m_state == utf8_accept;
    }

    /// Reset the validator to decode another message
    void reset() {
        m_state = utf8_accept;
        m_codepoint = 0;
    }
private:
    uint32_t    m_state;
    uint32_t    m_codepoint;
};

/// Validate a UTF8 string
/**
 * convenience function that creates a validator, validates a complete string
 * and returns the result.
 */
inline bool validate(std::string const & s) {
    validator v;
    if (!v.decode(s.begin(),s.end())) {
        return false;
    }
    return v.complete();
}

} // namespace utf8_validator
} // namespace websocketpp

#endif // UTF8_VALIDATOR_HPP
