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

#ifndef WEBSOCKETPP_EXTENSION_PERMESSAGE_DEFLATE_DISABLED_HPP
#define WEBSOCKETPP_EXTENSION_PERMESSAGE_DEFLATE_DISABLED_HPP

#include <websocketpp/common/cpp11.hpp>
#include <websocketpp/common/system_error.hpp>

#include <websocketpp/http/constants.hpp>
#include <websocketpp/extensions/extension.hpp>

#include <map>
#include <string>
#include <utility>

namespace websocketpp {
namespace extensions {
namespace permessage_deflate {

/// Stub class for use when disabling permessage_deflate extension
/**
 * This class is a stub that implements the permessage_deflate interface
 * with minimal dependencies. It is used to disable permessage_deflate
 * functionality at compile time without loading any unnecessary code.
 */
template <typename config>
class disabled {
    typedef std::pair<lib::error_code,std::string> err_str_pair;

public:
    /// Negotiate extension
    /**
     * The disabled extension always fails the negotiation with a disabled
     * error.
     *
     * @param offer Attribute from client's offer
     * @return Status code and value to return to remote endpoint
     */
    err_str_pair negotiate(http::attribute_list const &) {
        return make_pair(make_error_code(error::disabled),std::string());
    }

    /// Initialize state
    /**
     * For the disabled extension state initialization is a no-op.
     *
     * @param is_server True to initialize as a server, false for a client.
     * @return A code representing the error that occurred, if any
     */
    lib::error_code init(bool) {
        return lib::error_code();
    }

    /// Returns true if the extension is capable of providing
    /// permessage_deflate functionality
    bool is_implemented() const {
        return false;
    }

    /// Returns true if permessage_deflate functionality is active for this
    /// connection
    bool is_enabled() const {
        return false;
    }

    /// Generate extension offer
    /**
     * Creates an offer string to include in the Sec-WebSocket-Extensions
     * header of outgoing client requests.
     *
     * @return A WebSocket extension offer string for this extension
     */
    std::string generate_offer() const {
        return "";
    }

    /// Compress bytes
    /**
     * @param [in] in String to compress
     * @param [out] out String to append compressed bytes to
     * @return Error or status code
     */
    lib::error_code compress(std::string const &, std::string &) {
        return make_error_code(error::disabled);
    }

    /// Decompress bytes
    /**
     * @param buf Byte buffer to decompress
     * @param len Length of buf
     * @param out String to append decompressed bytes to
     * @return Error or status code
     */
    lib::error_code decompress(uint8_t const *, size_t, std::string &) {
        return make_error_code(error::disabled);
    }
};

} // namespace permessage_deflate
} // namespace extensions
} // namespace websocketpp

#endif // WEBSOCKETPP_EXTENSION_PERMESSAGE_DEFLATE_DISABLED_HPP
