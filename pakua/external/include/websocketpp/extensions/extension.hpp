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

#ifndef WEBSOCKETPP_EXTENSION_HPP
#define WEBSOCKETPP_EXTENSION_HPP

#include <websocketpp/common/cpp11.hpp>
#include <websocketpp/common/system_error.hpp>

#include <string>
#include <vector>

namespace websocketpp {

/**
 * Some generic information about extensions
 *
 * Each extension object has an implemented flag. It can be retrieved by calling
 * is_implemented(). This compile time flag indicates whether or not the object
 * in question actually implements the extension or if it is a placeholder stub
 *
 * Each extension object also has an enabled flag. It can be retrieved by
 * calling is_enabled(). This runtime flag indicates whether or not the
 * extension has been negotiated for this connection.
 */
namespace extensions {

namespace error {
enum value {
    /// Catch all
    general = 1,

    /// Extension disabled
    disabled
};

class category : public lib::error_category {
public:
    category() {}

    const char *name() const _WEBSOCKETPP_NOEXCEPT_TOKEN_ {
        return "websocketpp.extension";
    }

    std::string message(int value) const {
        switch(value) {
            case general:
                return "Generic extension error";
            case disabled:
                return "Use of methods from disabled extension";
            default:
                return "Unknown permessage-compress error";
        }
    }
};

inline lib::error_category const & get_category() {
    static category instance;
    return instance;
}

inline lib::error_code make_error_code(error::value e) {
    return lib::error_code(static_cast<int>(e), get_category());
}

} // namespace error
} // namespace extensions
} // namespace websocketpp

_WEBSOCKETPP_ERROR_CODE_ENUM_NS_START_
template<> struct is_error_code_enum
    <websocketpp::extensions::error::value>
{
    static const bool value = true;
};
_WEBSOCKETPP_ERROR_CODE_ENUM_NS_END_

#endif // WEBSOCKETPP_EXTENSION_HPP
