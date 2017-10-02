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

#ifndef WEBSOCKETPP_COMMON_NETWORK_HPP
#define WEBSOCKETPP_COMMON_NETWORK_HPP

// For ntohs and htons
#if defined(_WIN32)
    #include <winsock2.h>
#else
    //#include <arpa/inet.h>
    #include <netinet/in.h>
#endif

#include <websocketpp/common/stdint.hpp>

namespace websocketpp {
namespace lib {
namespace net {

inline bool is_little_endian() {
    short int val = 0x1;
    char *ptr = reinterpret_cast<char *>(&val);
    return (ptr[0] == 1);
}

#define TYP_INIT 0
#define TYP_SMLE 1
#define TYP_BIGE 2

/// Convert 64 bit value to network byte order
/**
 * This method is prefixed to avoid conflicts with operating system level
 * macros for this functionality.
 *
 * TODO: figure out if it would be beneficial to use operating system level
 * macros for this.
 *
 * @param src The integer in host byte order
 * @return src converted to network byte order
 */
inline uint64_t _htonll(uint64_t src) {
    static int typ = TYP_INIT;
    unsigned char c;
    union {
        uint64_t ull;
        unsigned char c[8];
    } x;
    if (typ == TYP_INIT) {
        x.ull = 0x01;
        typ = (x.c[7] == 0x01ULL) ? TYP_BIGE : TYP_SMLE;
    }
    if (typ == TYP_BIGE)
        return src;
    x.ull = src;
    c = x.c[0]; x.c[0] = x.c[7]; x.c[7] = c;
    c = x.c[1]; x.c[1] = x.c[6]; x.c[6] = c;
    c = x.c[2]; x.c[2] = x.c[5]; x.c[5] = c;
    c = x.c[3]; x.c[3] = x.c[4]; x.c[4] = c;
    return x.ull;
}

/// Convert 64 bit value to host byte order
/**
 * This method is prefixed to avoid conflicts with operating system level
 * macros for this functionality.
 *
 * TODO: figure out if it would be beneficial to use operating system level
 * macros for this.
 *
 * @param src The integer in network byte order
 * @return src converted to host byte order
 */
inline uint64_t _ntohll(uint64_t src) {
    return _htonll(src);
}

} // net
} // lib
} // websocketpp

#endif // WEBSOCKETPP_COMMON_NETWORK_HPP
