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

#ifndef WEBSOCKETPP_UTILITIES_IMPL_HPP
#define WEBSOCKETPP_UTILITIES_IMPL_HPP

#include <algorithm>
#include <string>

namespace websocketpp {
namespace utility {

inline std::string to_lower(std::string const & in) {
    std::string out = in;
    std::transform(out.begin(),out.end(),out.begin(),::tolower);
    return out;
}

inline std::string to_hex(std::string const & input) {
    std::string output;
    std::string hex = "0123456789ABCDEF";

    for (size_t i = 0; i < input.size(); i++) {
        output += hex[(input[i] & 0xF0) >> 4];
        output += hex[input[i] & 0x0F];
        output += " ";
    }

    return output;
}

inline std::string to_hex(uint8_t const * input, size_t length) {
    std::string output;
    std::string hex = "0123456789ABCDEF";

    for (size_t i = 0; i < length; i++) {
        output += hex[(input[i] & 0xF0) >> 4];
        output += hex[input[i] & 0x0F];
        output += " ";
    }

    return output;
}

inline std::string to_hex(const char* input,size_t length) {
    return to_hex(reinterpret_cast<const uint8_t*>(input),length);
}

inline std::string string_replace_all(std::string subject, std::string const &
    search, std::string const & replace)
{
    size_t pos = 0;
    while((pos = subject.find(search, pos)) != std::string::npos) {
         subject.replace(pos, search.length(), replace);
         pos += replace.length();
    }
    return subject;
}

} // namespace utility
} // namespace websocketpp

#endif // WEBSOCKETPP_UTILITIES_IMPL_HPP
