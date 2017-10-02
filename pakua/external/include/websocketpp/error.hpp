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

#ifndef WEBSOCKETPP_ERROR_HPP
#define WEBSOCKETPP_ERROR_HPP

#include <exception>
#include <string>
#include <utility>

#include <websocketpp/common/cpp11.hpp>
#include <websocketpp/common/system_error.hpp>

namespace websocketpp {

/// Combination error code / string type for returning two values
typedef std::pair<lib::error_code,std::string> err_str_pair;

/// Library level error codes
namespace error {
enum value {
    /// Catch-all library error
    general = 1,

    /// send attempted when endpoint write queue was full
    send_queue_full,

    /// Attempted an operation using a payload that was improperly formatted
    /// ex: invalid UTF8 encoding on a text message.
    payload_violation,

    /// Attempted to open a secure connection with an insecure endpoint
    endpoint_not_secure,

    /// Attempted an operation that required an endpoint that is no longer
    /// available. This is usually because the endpoint went out of scope
    /// before a connection that it created.
    endpoint_unavailable,

    /// An invalid uri was supplied
    invalid_uri,

    /// The endpoint is out of outgoing message buffers
    no_outgoing_buffers,

    /// The endpoint is out of incoming message buffers
    no_incoming_buffers,

    /// The connection was in the wrong state for this operation
    invalid_state,

    /// Unable to parse close code
    bad_close_code,

    /// Close code is in a reserved range
    reserved_close_code,

    /// Close code is invalid
    invalid_close_code,

    /// Invalid UTF-8
    invalid_utf8,

    /// Invalid subprotocol
    invalid_subprotocol,

    /// An operation was attempted on a connection that did not exist or was
    /// already deleted.
    bad_connection,

    /// Unit testing utility error code
    test,

    /// Connection creation attempted failed
    con_creation_failed,

    /// Selected subprotocol was not requested by the client
    unrequested_subprotocol,

    /// Attempted to use a client specific feature on a server endpoint
    client_only,

    /// Attempted to use a server specific feature on a client endpoint
    server_only,

    /// HTTP connection ended
    http_connection_ended,

    /// WebSocket opening handshake timed out
    open_handshake_timeout,

    /// WebSocket close handshake timed out
    close_handshake_timeout,

    /// Invalid port in URI
    invalid_port,

    /// An async accept operation failed because the underlying transport has been
    /// requested to not listen for new connections anymore.
    async_accept_not_listening,

    /// The requested operation was canceled
    operation_canceled,

    /// Connection rejected
    rejected,

    /// Upgrade Required. This happens if an HTTP request is made to a
    /// WebSocket++ server that doesn't implement an http handler
    upgrade_required,

    /// Invalid WebSocket protocol version
    invalid_version,

    /// Unsupported WebSocket protocol version
    unsupported_version,

    /// HTTP parse error
    http_parse_error,
    
    /// Extension negotiation failed
    extension_neg_failed
}; // enum value


class category : public lib::error_category {
public:
    category() {}

    char const * name() const _WEBSOCKETPP_NOEXCEPT_TOKEN_ {
        return "websocketpp";
    }

    std::string message(int value) const {
        switch(value) {
            case error::general:
                return "Generic error";
            case error::send_queue_full:
                return "send queue full";
            case error::payload_violation:
                return "payload violation";
            case error::endpoint_not_secure:
                return "endpoint not secure";
            case error::endpoint_unavailable:
                return "endpoint not available";
            case error::invalid_uri:
                return "invalid uri";
            case error::no_outgoing_buffers:
                return "no outgoing message buffers";
            case error::no_incoming_buffers:
                return "no incoming message buffers";
            case error::invalid_state:
                return "invalid state";
            case error::bad_close_code:
                return "Unable to extract close code";
            case error::invalid_close_code:
                return "Extracted close code is in an invalid range";
            case error::reserved_close_code:
                return "Extracted close code is in a reserved range";
            case error::invalid_utf8:
                return "Invalid UTF-8";
            case error::invalid_subprotocol:
                return "Invalid subprotocol";
            case error::bad_connection:
                return "Bad Connection";
            case error::test:
                return "Test Error";
            case error::con_creation_failed:
                return "Connection creation attempt failed";
            case error::unrequested_subprotocol:
                return "Selected subprotocol was not requested by the client";
            case error::client_only:
                return "Feature not available on server endpoints";
            case error::server_only:
                return "Feature not available on client endpoints";
            case error::http_connection_ended:
                return "HTTP connection ended";
            case error::open_handshake_timeout:
                return "The opening handshake timed out";
            case error::close_handshake_timeout:
                return "The closing handshake timed out";
            case error::invalid_port:
                return "Invalid URI port";
            case error::async_accept_not_listening:
                return "Async Accept not listening";
            case error::operation_canceled:
                return "Operation canceled";
            case error::rejected:
                return "Connection rejected";
            case error::upgrade_required:
                return "Upgrade required";
            case error::invalid_version:
                return "Invalid version";
            case error::unsupported_version:
                return "Unsupported version";
            case error::http_parse_error:
                return "HTTP parse error";
            case error::extension_neg_failed:
                return "Extension negotiation failed";
            default:
                return "Unknown";
        }
    }
};

inline const lib::error_category& get_category() {
    static category instance;
    return instance;
}

inline lib::error_code make_error_code(error::value e) {
    return lib::error_code(static_cast<int>(e), get_category());
}

} // namespace error
} // namespace websocketpp

_WEBSOCKETPP_ERROR_CODE_ENUM_NS_START_
template<> struct is_error_code_enum<websocketpp::error::value>
{
    static bool const value = true;
};
_WEBSOCKETPP_ERROR_CODE_ENUM_NS_END_

namespace websocketpp {

class exception : public std::exception {
public:
    exception(std::string const & msg, lib::error_code ec = make_error_code(error::general))
      : m_msg(msg.empty() ? ec.message() : msg), m_code(ec)
    {}

    explicit exception(lib::error_code ec)
      : m_msg(ec.message()), m_code(ec)
    {}

    ~exception() throw() {}

    virtual char const * what() const throw() {
        return m_msg.c_str();
    }

    lib::error_code code() const throw() {
        return m_code;
    }

    const std::string m_msg;
    lib::error_code m_code;
};

} // namespace websocketpp

#endif // WEBSOCKETPP_ERROR_HPP
