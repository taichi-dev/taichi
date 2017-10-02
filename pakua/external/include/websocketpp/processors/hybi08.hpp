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

#ifndef WEBSOCKETPP_PROCESSOR_HYBI08_HPP
#define WEBSOCKETPP_PROCESSOR_HYBI08_HPP

#include <websocketpp/processors/hybi13.hpp>

#include <string>
#include <vector>

namespace websocketpp {
namespace processor {

/// Processor for Hybi Draft version 08
/**
 * The primary difference between 08 and 13 is a different origin header name
 */
template <typename config>
class hybi08 : public hybi13<config> {
public:
    typedef hybi08<config> type;
    typedef typename config::request_type request_type;

    typedef typename config::con_msg_manager_type::ptr msg_manager_ptr;
    typedef typename config::rng_type rng_type;

    explicit hybi08(bool secure, bool p_is_server, msg_manager_ptr manager, rng_type& rng)
      : hybi13<config>(secure, p_is_server, manager, rng) {}

    /// Fill in a set of request headers for a client connection request
    /**
     * The Hybi 08 processor only implements incoming connections so this will
     * always return an error.
     *
     * @param [out] req  Set of headers to fill in
     * @param [in] uri The uri being connected to
     * @param [in] subprotocols The list of subprotocols to request
     */
    lib::error_code client_handshake_request(request_type &, uri_ptr,
        std::vector<std::string> const &) const
    {
        return error::make_error_code(error::no_protocol_support);
    }

    int get_version() const {
        return 8;
    }

    std::string const & get_origin(request_type const & r) const {
        return r.get_header("Sec-WebSocket-Origin");
    }
private:
};

} // namespace processor
} // namespace websocketpp

#endif //WEBSOCKETPP_PROCESSOR_HYBI08_HPP
