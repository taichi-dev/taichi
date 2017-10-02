/*
 * Copyright (c) 2015, Peter Thorson. All rights reserved.
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

#ifndef WEBSOCKETPP_TRANSPORT_ASIO_SOCKET_BASE_HPP
#define WEBSOCKETPP_TRANSPORT_ASIO_SOCKET_BASE_HPP

#include <websocketpp/common/asio.hpp>
#include <websocketpp/common/memory.hpp>
#include <websocketpp/common/functional.hpp>
#include <websocketpp/common/system_error.hpp>
#include <websocketpp/common/cpp11.hpp>
#include <websocketpp/common/connection_hdl.hpp>

#include <string>

// Interface that sockets/security policies must implement

/*
 * Endpoint Interface
 *
 * bool is_secure() const;
 * @return Whether or not the endpoint creates secure connections
 *
 * lib::error_code init(socket_con_ptr scon);
 * Called by the transport after a new connection is created to initialize
 * the socket component of the connection.
 * @param scon Pointer to the socket component of the connection
 * @return Error code (empty on success)
 */


// Connection
// TODO
// set_hostname(std::string hostname)
// pre_init(init_handler);
// post_init(init_handler);

namespace websocketpp {
namespace transport {
namespace asio {
namespace socket {

typedef lib::function<void(lib::asio::error_code const &)> shutdown_handler;

/**
 * The transport::asio::socket::* classes are a set of security/socket related
 * policies and support code for the ASIO transport types.
 */

/// Errors related to asio transport sockets
namespace error {
    enum value {
        /// Catch-all error for security policy errors that don't fit in other
        /// categories
        security = 1,

        /// Catch-all error for socket component errors that don't fit in other
        /// categories
        socket,

        /// A function was called in a state that it was illegal to do so.
        invalid_state,

        /// The application was prompted to provide a TLS context and it was
        /// empty or otherwise invalid
        invalid_tls_context,

        /// TLS Handshake Timeout
        tls_handshake_timeout,

        /// pass_through from underlying library
        pass_through,

        /// Required tls_init handler not present
        missing_tls_init_handler,

        /// TLS Handshake Failed
        tls_handshake_failed,
        
        /// Failed to set TLS SNI hostname
        tls_failed_sni_hostname
    };
} // namespace error

/// Error category related to asio transport socket policies
class socket_category : public lib::error_category {
public:
    char const * name() const _WEBSOCKETPP_NOEXCEPT_TOKEN_ {
        return "websocketpp.transport.asio.socket";
    }

    std::string message(int value) const {
        switch(value) {
            case error::security:
                return "Security policy error";
            case error::socket:
                return "Socket component error";
            case error::invalid_state:
                return "Invalid state";
            case error::invalid_tls_context:
                return "Invalid or empty TLS context supplied";
            case error::tls_handshake_timeout:
                return "TLS handshake timed out";
            case error::pass_through:
                return "Pass through from socket policy";
            case error::missing_tls_init_handler:
                return "Required tls_init handler not present.";
            case error::tls_handshake_failed:
                return "TLS handshake failed";
            case error::tls_failed_sni_hostname:
                return "Failed to set TLS SNI hostname";
            default:
                return "Unknown";
        }
    }
};

inline lib::error_category const & get_socket_category() {
    static socket_category instance;
    return instance;
}

inline lib::error_code make_error_code(error::value e) {
    return lib::error_code(static_cast<int>(e), get_socket_category());
}

/// Type of asio transport socket policy initialization handlers
typedef lib::function<void(const lib::error_code&)> init_handler;

} // namespace socket
} // namespace asio
} // namespace transport
} // namespace websocketpp

#endif // WEBSOCKETPP_TRANSPORT_ASIO_SOCKET_BASE_HPP
