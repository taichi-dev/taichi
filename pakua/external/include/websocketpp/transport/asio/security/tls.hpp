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

#ifndef WEBSOCKETPP_TRANSPORT_SECURITY_TLS_HPP
#define WEBSOCKETPP_TRANSPORT_SECURITY_TLS_HPP

#include <websocketpp/transport/asio/security/base.hpp>

#include <websocketpp/uri.hpp>

#include <websocketpp/common/asio_ssl.hpp>
#include <websocketpp/common/asio.hpp>
#include <websocketpp/common/connection_hdl.hpp>
#include <websocketpp/common/functional.hpp>
#include <websocketpp/common/memory.hpp>

#include <sstream>
#include <string>

namespace websocketpp {
namespace transport {
namespace asio {
/// A socket policy for the asio transport that implements a TLS encrypted
/// socket by wrapping with an asio::ssl::stream
namespace tls_socket {

/// The signature of the socket_init_handler for this socket policy
typedef lib::function<void(connection_hdl,lib::asio::ssl::stream<
    lib::asio::ip::tcp::socket>&)> socket_init_handler;
/// The signature of the tls_init_handler for this socket policy
typedef lib::function<lib::shared_ptr<lib::asio::ssl::context>(connection_hdl)>
    tls_init_handler;

/// TLS enabled Asio connection socket component
/**
 * transport::asio::tls_socket::connection implements a secure connection socket
 * component that uses Asio's ssl::stream to wrap an ip::tcp::socket.
 */
class connection : public lib::enable_shared_from_this<connection> {
public:
    /// Type of this connection socket component
    typedef connection type;
    /// Type of a shared pointer to this connection socket component
    typedef lib::shared_ptr<type> ptr;

    /// Type of the ASIO socket being used
    typedef lib::asio::ssl::stream<lib::asio::ip::tcp::socket> socket_type;
    /// Type of a shared pointer to the ASIO socket being used
    typedef lib::shared_ptr<socket_type> socket_ptr;
    /// Type of a pointer to the ASIO io_service being used
    typedef lib::asio::io_service * io_service_ptr;
    /// Type of a pointer to the ASIO io_service strand being used
    typedef lib::shared_ptr<lib::asio::io_service::strand> strand_ptr;
    /// Type of a shared pointer to the ASIO TLS context being used
    typedef lib::shared_ptr<lib::asio::ssl::context> context_ptr;

    explicit connection() {
        //std::cout << "transport::asio::tls_socket::connection constructor"
        //          << std::endl;
    }

    /// Get a shared pointer to this component
    ptr get_shared() {
        return shared_from_this();
    }

    /// Check whether or not this connection is secure
    /**
     * @return Whether or not this connection is secure
     */
    bool is_secure() const {
        return true;
    }

    /// Retrieve a pointer to the underlying socket
    /**
     * This is used internally. It can also be used to set socket options, etc
     */
    socket_type::lowest_layer_type & get_raw_socket() {
        return m_socket->lowest_layer();
    }

    /// Retrieve a pointer to the layer below the ssl stream
    /**
     * This is used internally.
     */
    socket_type::next_layer_type & get_next_layer() {
        return m_socket->next_layer();
    }

    /// Retrieve a pointer to the wrapped socket
    /**
     * This is used internally.
     */
    socket_type & get_socket() {
        return *m_socket;
    }

    /// Set the socket initialization handler
    /**
     * The socket initialization handler is called after the socket object is
     * created but before it is used. This gives the application a chance to
     * set any ASIO socket options it needs.
     *
     * @param h The new socket_init_handler
     */
    void set_socket_init_handler(socket_init_handler h) {
        m_socket_init_handler = h;
    }

    /// Set TLS init handler
    /**
     * The tls init handler is called when needed to request a TLS context for
     * the library to use. A TLS init handler must be set and it must return a
     * valid TLS context in order for this endpoint to be able to initialize
     * TLS connections
     *
     * @param h The new tls_init_handler
     */
    void set_tls_init_handler(tls_init_handler h) {
        m_tls_init_handler = h;
    }

    /// Get the remote endpoint address
    /**
     * The iostream transport has no information about the ultimate remote
     * endpoint. It will return the string "iostream transport". To indicate
     * this.
     *
     * TODO: allow user settable remote endpoint addresses if this seems useful
     *
     * @return A string identifying the address of the remote endpoint
     */
    std::string get_remote_endpoint(lib::error_code & ec) const {
        std::stringstream s;

        lib::asio::error_code aec;
        lib::asio::ip::tcp::endpoint ep = m_socket->lowest_layer().remote_endpoint(aec);

        if (aec) {
            ec = error::make_error_code(error::pass_through);
            s << "Error getting remote endpoint: " << aec
               << " (" << aec.message() << ")";
            return s.str();
        } else {
            ec = lib::error_code();
            s << ep;
            return s.str();
        }
    }
protected:
    /// Perform one time initializations
    /**
     * init_asio is called once immediately after construction to initialize
     * Asio components to the io_service
     *
     * @param service A pointer to the endpoint's io_service
     * @param strand A pointer to the connection's strand
     * @param is_server Whether or not the endpoint is a server or not.
     */
    lib::error_code init_asio (io_service_ptr service, strand_ptr strand,
        bool is_server)
    {
        if (!m_tls_init_handler) {
            return socket::make_error_code(socket::error::missing_tls_init_handler);
        }
        m_context = m_tls_init_handler(m_hdl);

        if (!m_context) {
            return socket::make_error_code(socket::error::invalid_tls_context);
        }
        m_socket = lib::make_shared<socket_type>(
            _WEBSOCKETPP_REF(*service),lib::ref(*m_context));

        m_io_service = service;
        m_strand = strand;
        m_is_server = is_server;

        return lib::error_code();
    }

    /// Set hostname hook
    /**
     * Called by the transport as a connection is being established to provide
     * the hostname being connected to to the security/socket layer.
     *
     * This socket policy uses the hostname to set the appropriate TLS SNI
     * header.
     *
     * @since 0.6.0
     *
     * @param u The uri to set
     */
    void set_uri(uri_ptr u) {
        m_uri = u;
    }

    /// Pre-initialize security policy
    /**
     * Called by the transport after a new connection is created to initialize
     * the socket component of the connection. This method is not allowed to
     * write any bytes to the wire. This initialization happens before any
     * proxies or other intermediate wrappers are negotiated.
     *
     * @param callback Handler to call back with completion information
     */
    void pre_init(init_handler callback) {
        // TODO: is this the best way to check whether this function is 
        //       available in the version of OpenSSL being used?
        // TODO: consider case where host is an IP address
#if OPENSSL_VERSION_NUMBER >= 0x90812f
        if (!m_is_server) {
            // For clients on systems with a suitable OpenSSL version, set the
            // TLS SNI hostname header so connecting to TLS servers using SNI
            // will work.
            long res = SSL_set_tlsext_host_name(
                get_socket().native_handle(), m_uri->get_host().c_str());
            if (!(1 == res)) {
                callback(socket::make_error_code(socket::error::tls_failed_sni_hostname));
            }
        }
#endif

        if (m_socket_init_handler) {
            m_socket_init_handler(m_hdl,get_socket());
        }

        callback(lib::error_code());
    }

    /// Post-initialize security policy
    /**
     * Called by the transport after all intermediate proxies have been
     * negotiated. This gives the security policy the chance to talk with the
     * real remote endpoint for a bit before the websocket handshake.
     *
     * @param callback Handler to call back with completion information
     */
    void post_init(init_handler callback) {
        m_ec = socket::make_error_code(socket::error::tls_handshake_timeout);

        // TLS handshake
        if (m_strand) {
            m_socket->async_handshake(
                get_handshake_type(),
                m_strand->wrap(lib::bind(
                    &type::handle_init, get_shared(),
                    callback,
                    lib::placeholders::_1
                ))
            );
        } else {
            m_socket->async_handshake(
                get_handshake_type(),
                lib::bind(
                    &type::handle_init, get_shared(),
                    callback,
                    lib::placeholders::_1
                )
            );
        }
    }

    /// Sets the connection handle
    /**
     * The connection handle is passed to any handlers to identify the
     * connection
     *
     * @param hdl The new handle
     */
    void set_handle(connection_hdl hdl) {
        m_hdl = hdl;
    }

    void handle_init(init_handler callback,lib::asio::error_code const & ec) {
        if (ec) {
            m_ec = socket::make_error_code(socket::error::tls_handshake_failed);
        } else {
            m_ec = lib::error_code();
        }

        callback(m_ec);
    }

    lib::error_code get_ec() const {
        return m_ec;
    }

    /// Cancel all async operations on this socket
    /**
     * Attempts to cancel all async operations on this socket and reports any
     * failures.
     *
     * NOTE: Windows XP and earlier do not support socket cancellation.
     *
     * @return The error that occurred, if any.
     */
    lib::asio::error_code cancel_socket() {
        lib::asio::error_code ec;
        get_raw_socket().cancel(ec);
        return ec;
    }

    void async_shutdown(socket::shutdown_handler callback) {
        if (m_strand) {
            m_socket->async_shutdown(m_strand->wrap(callback));
        } else {
            m_socket->async_shutdown(callback);
        }
    }

    /// Translate any security policy specific information about an error code
    /**
     * Translate_ec takes an Asio error code and attempts to convert its value
     * to an appropriate websocketpp error code. In the case that the Asio and
     * Websocketpp error types are the same (such as using boost::asio and
     * boost::system_error or using standalone asio and std::system_error the
     * code will be passed through natively.
     *
     * In the case of a mismatch (boost::asio with std::system_error) a
     * translated code will be returned. Any error that is determined to be
     * related to TLS but does not have a more specific websocketpp error code
     * is returned under the catch all error `tls_error`. Non-TLS related errors
     * are returned as the transport generic error `pass_through`
     *
     * @since 0.3.0
     *
     * @param ec The error code to translate_ec
     * @return The translated error code
     */
    template <typename ErrorCodeType>
    lib::error_code translate_ec(ErrorCodeType ec) {
        if (ec.category() == lib::asio::error::get_ssl_category()) {
            if (ERR_GET_REASON(ec.value()) == SSL_R_SHORT_READ) {
                return make_error_code(transport::error::tls_short_read);
            } else {
                // We know it is a TLS related error, but otherwise don't know
                // more. Pass through as TLS generic.
                return make_error_code(transport::error::tls_error);
            }
        } else {
            // We don't know any more information about this error so pass
            // through
            return make_error_code(transport::error::pass_through);
        }
    }
    
    /// Overload of translate_ec to catch cases where lib::error_code is the
    /// same type as lib::asio::error_code
    lib::error_code translate_ec(lib::error_code ec) {
        // Normalize the tls_short_read error as it is used by the library and 
        // needs a consistent value. All other errors pass through natively.
        // TODO: how to get the SSL category from std::error?
        /*if (ec.category() == lib::asio::error::get_ssl_category()) {
            if (ERR_GET_REASON(ec.value()) == SSL_R_SHORT_READ) {
                return make_error_code(transport::error::tls_short_read);
            }
        }*/
        return ec;
    }
private:
    socket_type::handshake_type get_handshake_type() {
        if (m_is_server) {
            return lib::asio::ssl::stream_base::server;
        } else {
            return lib::asio::ssl::stream_base::client;
        }
    }

    io_service_ptr      m_io_service;
    strand_ptr          m_strand;
    context_ptr         m_context;
    socket_ptr          m_socket;
    uri_ptr             m_uri;
    bool                m_is_server;

    lib::error_code     m_ec;

    connection_hdl      m_hdl;
    socket_init_handler m_socket_init_handler;
    tls_init_handler    m_tls_init_handler;
};

/// TLS enabled Asio endpoint socket component
/**
 * transport::asio::tls_socket::endpoint implements a secure endpoint socket
 * component that uses Asio's ssl::stream to wrap an ip::tcp::socket.
 */
class endpoint {
public:
    /// The type of this endpoint socket component
    typedef endpoint type;

    /// The type of the corresponding connection socket component
    typedef connection socket_con_type;
    /// The type of a shared pointer to the corresponding connection socket
    /// component.
    typedef socket_con_type::ptr socket_con_ptr;

    explicit endpoint() {}

    /// Checks whether the endpoint creates secure connections
    /**
     * @return Whether or not the endpoint creates secure connections
     */
    bool is_secure() const {
        return true;
    }

    /// Set socket init handler
    /**
     * The socket init handler is called after a connection's socket is created
     * but before it is used. This gives the end application an opportunity to
     * set asio socket specific parameters.
     *
     * @param h The new socket_init_handler
     */
    void set_socket_init_handler(socket_init_handler h) {
        m_socket_init_handler = h;
    }

    /// Set TLS init handler
    /**
     * The tls init handler is called when needed to request a TLS context for
     * the library to use. A TLS init handler must be set and it must return a
     * valid TLS context in order for this endpoint to be able to initialize
     * TLS connections
     *
     * @param h The new tls_init_handler
     */
    void set_tls_init_handler(tls_init_handler h) {
        m_tls_init_handler = h;
    }
protected:
    /// Initialize a connection
    /**
     * Called by the transport after a new connection is created to initialize
     * the socket component of the connection.
     *
     * @param scon Pointer to the socket component of the connection
     *
     * @return Error code (empty on success)
     */
    lib::error_code init(socket_con_ptr scon) {
        scon->set_socket_init_handler(m_socket_init_handler);
        scon->set_tls_init_handler(m_tls_init_handler);
        return lib::error_code();
    }

private:
    socket_init_handler m_socket_init_handler;
    tls_init_handler m_tls_init_handler;
};

} // namespace tls_socket
} // namespace asio
} // namespace transport
} // namespace websocketpp

#endif // WEBSOCKETPP_TRANSPORT_SECURITY_TLS_HPP
