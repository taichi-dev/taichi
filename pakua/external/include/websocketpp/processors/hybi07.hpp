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

#ifndef WEBSOCKETPP_PROCESSOR_HYBI07_HPP
#define WEBSOCKETPP_PROCESSOR_HYBI07_HPP

#include <websocketpp/processors/hybi08.hpp>

#include <string>
#include <vector>

namespace websocketpp {
namespace processor {

/// Processor for Hybi Draft version 07
/**
 * The primary difference between 07 and 08 is a version number.
 */
template <typename config>
class hybi07 : public hybi08<config> {
public:
    typedef typename config::request_type request_type;

    typedef typename config::con_msg_manager_type::ptr msg_manager_ptr;
    typedef typename config::rng_type rng_type;

    explicit hybi07(bool secure, bool p_is_server, msg_manager_ptr manager, rng_type& rng)
      : hybi08<config>(secure, p_is_server, manager, rng) {}

    /// Fill in a set of request headers for a client connection request
    /**
     * The Hybi 07 processor only implements incoming connections so this will
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
        return 7;
    }
private:
};

} // namespace processor
} // namespace websocketpp

#endif //WEBSOCKETPP_PROCESSOR_HYBI07_HPP
