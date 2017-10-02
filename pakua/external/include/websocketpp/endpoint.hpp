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

#ifndef WEBSOCKETPP_ENDPOINT_HPP
#define WEBSOCKETPP_ENDPOINT_HPP

#include <websocketpp/connection.hpp>

#include <websocketpp/logger/levels.hpp>
#include <websocketpp/version.hpp>

#include <string>

namespace websocketpp {

/// Creates and manages connections associated with a WebSocket endpoint
template <typename connection, typename config>
class endpoint : public config::transport_type, public config::endpoint_base {
public:
    // Import appropriate types from our helper class
    // See endpoint_types for more details.
    typedef endpoint<connection,config> type;

    /// Type of the transport component of this endpoint
    typedef typename config::transport_type transport_type;
    /// Type of the concurrency component of this endpoint
    typedef typename config::concurrency_type concurrency_type;

    /// Type of the connections that this endpoint creates
    typedef connection connection_type;
    /// Shared pointer to connection_type
    typedef typename connection_type::ptr connection_ptr;
    /// Weak pointer to connection type
    typedef typename connection_type::weak_ptr connection_weak_ptr;

    /// Type of the transport component of the connections that this endpoint
    /// creates
    typedef typename transport_type::transport_con_type transport_con_type;
    /// Type of a shared pointer to the transport component of the connections
    /// that this endpoint creates.
    typedef typename transport_con_type::ptr transport_con_ptr;

    /// Type of message_handler
    typedef typename connection_type::message_handler message_handler;
    /// Type of message pointers that this endpoint uses
    typedef typename connection_type::message_ptr message_ptr;

    /// Type of error logger
    typedef typename config::elog_type elog_type;
    /// Type of access logger
    typedef typename config::alog_type alog_type;

    /// Type of our concurrency policy's scoped lock object
    typedef typename concurrency_type::scoped_lock_type scoped_lock_type;
    /// Type of our concurrency policy's mutex object
    typedef typename concurrency_type::mutex_type mutex_type;

    /// Type of RNG
    typedef typename config::rng_type rng_type;

    // TODO: organize these
    typedef typename connection_type::termination_handler termination_handler;

    // This would be ideal. Requires C++11 though
    //friend connection;

    explicit endpoint(bool p_is_server)
      : m_alog(config::alog_level, log::channel_type_hint::access)
      , m_elog(config::elog_level, log::channel_type_hint::error)
      , m_user_agent(::websocketpp::user_agent)
      , m_open_handshake_timeout_dur(config::timeout_open_handshake)
      , m_close_handshake_timeout_dur(config::timeout_close_handshake)
      , m_pong_timeout_dur(config::timeout_pong)
      , m_max_message_size(config::max_message_size)
      , m_max_http_body_size(config::max_http_body_size)
      , m_is_server(p_is_server)
    {
        m_alog.set_channels(config::alog_level);
        m_elog.set_channels(config::elog_level);

        m_alog.write(log::alevel::devel, "endpoint constructor");

        transport_type::init_logging(&m_alog, &m_elog);
    }


    /// Destructor
    ~endpoint<connection,config>() {}

    #ifdef _WEBSOCKETPP_DEFAULT_DELETE_FUNCTIONS_
        // no copy constructor because endpoints are not copyable
        endpoint(endpoint &) = delete;
    
        // no copy assignment operator because endpoints are not copyable
        endpoint & operator=(endpoint const &) = delete;
    #endif // _WEBSOCKETPP_DEFAULT_DELETE_FUNCTIONS_

    #ifdef _WEBSOCKETPP_MOVE_SEMANTICS_
        /// Move constructor
        endpoint(endpoint && o) 
         : config::transport_type(std::move(o))
         , config::endpoint_base(std::move(o))
         , m_alog(std::move(o.m_alog))
         , m_elog(std::move(o.m_elog))
         , m_user_agent(std::move(o.m_user_agent))
         , m_open_handler(std::move(o.m_open_handler))
         
         , m_close_handler(std::move(o.m_close_handler))
         , m_fail_handler(std::move(o.m_fail_handler))
         , m_ping_handler(std::move(o.m_ping_handler))
         , m_pong_handler(std::move(o.m_pong_handler))
         , m_pong_timeout_handler(std::move(o.m_pong_timeout_handler))
         , m_interrupt_handler(std::move(o.m_interrupt_handler))
         , m_http_handler(std::move(o.m_http_handler))
         , m_validate_handler(std::move(o.m_validate_handler))
         , m_message_handler(std::move(o.m_message_handler))

         , m_open_handshake_timeout_dur(o.m_open_handshake_timeout_dur)
         , m_close_handshake_timeout_dur(o.m_close_handshake_timeout_dur)
         , m_pong_timeout_dur(o.m_pong_timeout_dur)
         , m_max_message_size(o.m_max_message_size)
         , m_max_http_body_size(o.m_max_http_body_size)

         , m_rng(std::move(o.m_rng))
         , m_is_server(o.m_is_server)         
        {}

    #ifdef _WEBSOCKETPP_DEFAULT_DELETE_FUNCTIONS_
        // no move assignment operator because of const member variables
        endpoint & operator=(endpoint &&) = delete;
    #endif // _WEBSOCKETPP_DEFAULT_DELETE_FUNCTIONS_

    #endif // _WEBSOCKETPP_MOVE_SEMANTICS_


    /// Returns the user agent string that this endpoint will use
    /**
     * Returns the user agent string that this endpoint will use when creating
     * new connections.
     *
     * The default value for this version is stored in websocketpp::user_agent
     *
     * @return The user agent string.
     */
    std::string get_user_agent() const {
        scoped_lock_type guard(m_mutex);
        return m_user_agent;
    }

    /// Sets the user agent string that this endpoint will use
    /**
     * Sets the identifier that this endpoint will use when creating new
     * connections. Changing this value will only affect future connections.
     * For client endpoints this will be sent as the "User-Agent" header in
     * outgoing requests. For server endpoints this will be sent in the "Server"
     * response header.
     *
     * Setting this value to the empty string will suppress the use of the
     * Server and User-Agent headers. This is typically done to hide
     * implementation details for security purposes.
     *
     * For best results set this before accepting or opening connections.
     *
     * The default value for this version is stored in websocketpp::user_agent
     *
     * This can be overridden on an individual connection basis by setting a
     * custom "Server" header during the validate handler or "User-Agent"
     * header on a connection before calling connect().
     *
     * @param ua The string to set the user agent to.
     */
    void set_user_agent(std::string const & ua) {
        scoped_lock_type guard(m_mutex);
        m_user_agent = ua;
    }

    /// Returns whether or not this endpoint is a server.
    /**
     * @return Whether or not this endpoint is a server
     */
    bool is_server() const {
        return m_is_server;
    }

    /********************************/
    /* Pass-through logging adaptor */
    /********************************/

    /// Set Access logging channel
    /**
     * Set the access logger's channel value. The value is a number whose
     * interpretation depends on the logging policy in use.
     *
     * @param channels The channel value(s) to set
     */
    void set_access_channels(log::level channels) {
        m_alog.set_channels(channels);
    }

    /// Clear Access logging channels
    /**
     * Clear the access logger's channel value. The value is a number whose
     * interpretation depends on the logging policy in use.
     *
     * @param channels The channel value(s) to clear
     */
    void clear_access_channels(log::level channels) {
        m_alog.clear_channels(channels);
    }

    /// Set Error logging channel
    /**
     * Set the error logger's channel value. The value is a number whose
     * interpretation depends on the logging policy in use.
     *
     * @param channels The channel value(s) to set
     */
    void set_error_channels(log::level channels) {
        m_elog.set_channels(channels);
    }

    /// Clear Error logging channels
    /**
     * Clear the error logger's channel value. The value is a number whose
     * interpretation depends on the logging policy in use.
     *
     * @param channels The channel value(s) to clear
     */
    void clear_error_channels(log::level channels) {
        m_elog.clear_channels(channels);
    }

    /// Get reference to access logger
    /**
     * @return A reference to the access logger
     */
    alog_type & get_alog() {
        return m_alog;
    }

    /// Get reference to error logger
    /**
     * @return A reference to the error logger
     */
    elog_type & get_elog() {
        return m_elog;
    }

    /*************************/
    /* Set Handler functions */
    /*************************/

    void set_open_handler(open_handler h) {
        m_alog.write(log::alevel::devel,"set_open_handler");
        scoped_lock_type guard(m_mutex);
        m_open_handler = h;
    }
    void set_close_handler(close_handler h) {
        m_alog.write(log::alevel::devel,"set_close_handler");
        scoped_lock_type guard(m_mutex);
        m_close_handler = h;
    }
    void set_fail_handler(fail_handler h) {
        m_alog.write(log::alevel::devel,"set_fail_handler");
        scoped_lock_type guard(m_mutex);
        m_fail_handler = h;
    }
    void set_ping_handler(ping_handler h) {
        m_alog.write(log::alevel::devel,"set_ping_handler");
        scoped_lock_type guard(m_mutex);
        m_ping_handler = h;
    }
    void set_pong_handler(pong_handler h) {
        m_alog.write(log::alevel::devel,"set_pong_handler");
        scoped_lock_type guard(m_mutex);
        m_pong_handler = h;
    }
    void set_pong_timeout_handler(pong_timeout_handler h) {
        m_alog.write(log::alevel::devel,"set_pong_timeout_handler");
        scoped_lock_type guard(m_mutex);
        m_pong_timeout_handler = h;
    }
    void set_interrupt_handler(interrupt_handler h) {
        m_alog.write(log::alevel::devel,"set_interrupt_handler");
        scoped_lock_type guard(m_mutex);
        m_interrupt_handler = h;
    }
    void set_http_handler(http_handler h) {
        m_alog.write(log::alevel::devel,"set_http_handler");
        scoped_lock_type guard(m_mutex);
        m_http_handler = h;
    }
    void set_validate_handler(validate_handler h) {
        m_alog.write(log::alevel::devel,"set_validate_handler");
        scoped_lock_type guard(m_mutex);
        m_validate_handler = h;
    }
    void set_message_handler(message_handler h) {
        m_alog.write(log::alevel::devel,"set_message_handler");
        scoped_lock_type guard(m_mutex);
        m_message_handler = h;
    }

    //////////////////////////////////////////
    // Connection timeouts and other limits //
    //////////////////////////////////////////

    /// Set open handshake timeout
    /**
     * Sets the length of time the library will wait after an opening handshake
     * has been initiated before cancelling it. This can be used to prevent
     * excessive wait times for outgoing clients or excessive resource usage
     * from broken clients or DoS attacks on servers.
     *
     * Connections that time out will have their fail handlers called with the
     * open_handshake_timeout error code.
     *
     * The default value is specified via the compile time config value
     * 'timeout_open_handshake'. The default value in the core config
     * is 5000ms. A value of 0 will disable the timer entirely.
     *
     * To be effective, the transport you are using must support timers. See
     * the documentation for your transport policy for details about its
     * timer support.
     *
     * @param dur The length of the open handshake timeout in ms
     */
    void set_open_handshake_timeout(long dur) {
        scoped_lock_type guard(m_mutex);
        m_open_handshake_timeout_dur = dur;
    }

    /// Set close handshake timeout
    /**
     * Sets the length of time the library will wait after a closing handshake
     * has been initiated before cancelling it. This can be used to prevent
     * excessive wait times for outgoing clients or excessive resource usage
     * from broken clients or DoS attacks on servers.
     *
     * Connections that time out will have their close handlers called with the
     * close_handshake_timeout error code.
     *
     * The default value is specified via the compile time config value
     * 'timeout_close_handshake'. The default value in the core config
     * is 5000ms. A value of 0 will disable the timer entirely.
     *
     * To be effective, the transport you are using must support timers. See
     * the documentation for your transport policy for details about its
     * timer support.
     *
     * @param dur The length of the close handshake timeout in ms
     */
    void set_close_handshake_timeout(long dur) {
        scoped_lock_type guard(m_mutex);
        m_close_handshake_timeout_dur = dur;
    }

    /// Set pong timeout
    /**
     * Sets the length of time the library will wait for a pong response to a
     * ping. This can be used as a keepalive or to detect broken  connections.
     *
     * Pong responses that time out will have the pong timeout handler called.
     *
     * The default value is specified via the compile time config value
     * 'timeout_pong'. The default value in the core config
     * is 5000ms. A value of 0 will disable the timer entirely.
     *
     * To be effective, the transport you are using must support timers. See
     * the documentation for your transport policy for details about its
     * timer support.
     *
     * @param dur The length of the pong timeout in ms
     */
    void set_pong_timeout(long dur) {
        scoped_lock_type guard(m_mutex);
        m_pong_timeout_dur = dur;
    }

    /// Get default maximum message size
    /**
     * Get the default maximum message size that will be used for new 
     * connections created by this endpoint. The maximum message size determines
     * the point at which the connection will fail a connection with the 
     * message_too_big protocol error.
     *
     * The default is set by the max_message_size value from the template config
     *
     * @since 0.3.0
     */
    size_t get_max_message_size() const {
        return m_max_message_size;
    }
    
    /// Set default maximum message size
    /**
     * Set the default maximum message size that will be used for new 
     * connections created by this endpoint. Maximum message size determines the
     * point at which the connection will fail a connection with the
     * message_too_big protocol error.
     *
     * The default is set by the max_message_size value from the template config
     *
     * @since 0.3.0
     *
     * @param new_value The value to set as the maximum message size.
     */
    void set_max_message_size(size_t new_value) {
        m_max_message_size = new_value;
    }

    /// Get maximum HTTP message body size
    /**
     * Get maximum HTTP message body size. Maximum message body size determines
     * the point at which the connection will stop reading an HTTP request whose
     * body is too large.
     *
     * The default is set by the max_http_body_size value from the template
     * config
     *
     * @since 0.5.0
     *
     * @return The maximum HTTP message body size
     */
    size_t get_max_http_body_size() const {
        return m_max_http_body_size;
    }
    
    /// Set maximum HTTP message body size
    /**
     * Set maximum HTTP message body size. Maximum message body size determines
     * the point at which the connection will stop reading an HTTP request whose
     * body is too large.
     *
     * The default is set by the max_http_body_size value from the template
     * config
     *
     * @since 0.5.1
     *
     * @param new_value The value to set as the maximum message size.
     */
    void set_max_http_body_size(size_t new_value) {
        m_max_http_body_size = new_value;
    }

    /*************************************/
    /* Connection pass through functions */
    /*************************************/

    /**
     * These functions act as adaptors to their counterparts in connection. They
     * can produce one additional type of error, the bad_connection error, that
     * indicates that the conversion from connection_hdl to connection_ptr
     * failed due to the connection not existing anymore. Each method has a
     * default and an exception free varient.
     */

    void interrupt(connection_hdl hdl, lib::error_code & ec);
    void interrupt(connection_hdl hdl);

    /// Pause reading of new data (exception free)
    /**
     * Signals to the connection to halt reading of new data. While reading is 
     * paused, the connection will stop reading from its associated socket. In
     * turn this will result in TCP based flow control kicking in and slowing
     * data flow from the remote endpoint.
     *
     * This is useful for applications that push new requests to a queue to be 
     * processed by another thread and need a way to signal when their request
     * queue is full without blocking the network processing thread.
     *
     * Use `resume_reading()` to resume.
     *
     * If supported by the transport this is done asynchronously. As such
     * reading may not stop until the current read operation completes. 
     * Typically you can expect to receive no more bytes after initiating a read
     * pause than the size of the read buffer.
     *
     * If reading is paused for this connection already nothing is changed.
     */
    void pause_reading(connection_hdl hdl, lib::error_code & ec);
    
    /// Pause reading of new data
    void pause_reading(connection_hdl hdl);

    /// Resume reading of new data (exception free)
    /**
     * Signals to the connection to resume reading of new data after it was 
     * paused by `pause_reading()`.
     *
     * If reading is not paused for this connection already nothing is changed.
     */
    void resume_reading(connection_hdl hdl, lib::error_code & ec);

    /// Resume reading of new data
    void resume_reading(connection_hdl hdl);

    /// Send deferred HTTP Response
    /**
     * Sends an http response to an HTTP connection that was deferred. This will
     * send a complete response including all headers, status line, and body
     * text. The connection will be closed afterwards.
     *
     * Exception free variant
     *
     * @since 0.6.0
     *
     * @param hdl The connection to send the response on
     * @param ec A status code, zero on success, non-zero otherwise
     */
    void send_http_response(connection_hdl hdl, lib::error_code & ec);
        
    /// Send deferred HTTP Response (exception free)
    /**
     * Sends an http response to an HTTP connection that was deferred. This will
     * send a complete response including all headers, status line, and body
     * text. The connection will be closed afterwards.
     *
     * Exception variant
     *
     * @since 0.6.0
     *
     * @param hdl The connection to send the response on
     */
    void send_http_response(connection_hdl hdl);

    /// Create a message and add it to the outgoing send queue (exception free)
    /**
     * Convenience method to send a message given a payload string and an opcode
     *
     * @param [in] hdl The handle identifying the connection to send via.
     * @param [in] payload The payload string to generated the message with
     * @param [in] op The opcode to generated the message with.
     * @param [out] ec A code to fill in for errors
     */
    void send(connection_hdl hdl, std::string const & payload,
        frame::opcode::value op, lib::error_code & ec);
    /// Create a message and add it to the outgoing send queue
    /**
     * Convenience method to send a message given a payload string and an opcode
     *
     * @param [in] hdl The handle identifying the connection to send via.
     * @param [in] payload The payload string to generated the message with
     * @param [in] op The opcode to generated the message with.
     * @param [out] ec A code to fill in for errors
     */
    void send(connection_hdl hdl, std::string const & payload,
        frame::opcode::value op);

    void send(connection_hdl hdl, void const * payload, size_t len,
        frame::opcode::value op, lib::error_code & ec);
    void send(connection_hdl hdl, void const * payload, size_t len,
        frame::opcode::value op);

    void send(connection_hdl hdl, message_ptr msg, lib::error_code & ec);
    void send(connection_hdl hdl, message_ptr msg);

    void close(connection_hdl hdl, close::status::value const code,
        std::string const & reason, lib::error_code & ec);
    void close(connection_hdl hdl, close::status::value const code,
        std::string const & reason);

    /// Send a ping to a specific connection
    /**
     * @since 0.3.0-alpha3
     *
     * @param [in] hdl The connection_hdl of the connection to send to.
     * @param [in] payload The payload string to send.
     * @param [out] ec A reference to an error code to fill in
     */
    void ping(connection_hdl hdl, std::string const & payload,
        lib::error_code & ec);
    /// Send a ping to a specific connection
    /**
     * Exception variant of `ping`
     *
     * @since 0.3.0-alpha3
     *
     * @param [in] hdl The connection_hdl of the connection to send to.
     * @param [in] payload The payload string to send.
     */
    void ping(connection_hdl hdl, std::string const & payload);

    /// Send a pong to a specific connection
    /**
     * @since 0.3.0-alpha3
     *
     * @param [in] hdl The connection_hdl of the connection to send to.
     * @param [in] payload The payload string to send.
     * @param [out] ec A reference to an error code to fill in
     */
    void pong(connection_hdl hdl, std::string const & payload,
        lib::error_code & ec);
    /// Send a pong to a specific connection
    /**
     * Exception variant of `pong`
     *
     * @since 0.3.0-alpha3
     *
     * @param [in] hdl The connection_hdl of the connection to send to.
     * @param [in] payload The payload string to send.
     */
    void pong(connection_hdl hdl, std::string const & payload);

    /// Retrieves a connection_ptr from a connection_hdl (exception free)
    /**
     * Converting a weak pointer to shared_ptr is not thread safe because the
     * pointer could be deleted at any time.
     *
     * NOTE: This method may be called by handler to upgrade its handle to a
     * full connection_ptr. That full connection may then be used safely for the
     * remainder of the handler body. get_con_from_hdl and the resulting
     * connection_ptr are NOT safe to use outside the handler loop.
     *
     * @param hdl The connection handle to translate
     *
     * @return the connection_ptr. May be NULL if the handle was invalid.
     */
    connection_ptr get_con_from_hdl(connection_hdl hdl, lib::error_code & ec) {
        connection_ptr con = lib::static_pointer_cast<connection_type>(
            hdl.lock());
        if (!con) {
            ec = error::make_error_code(error::bad_connection);
        }
        return con;
    }

    /// Retrieves a connection_ptr from a connection_hdl (exception version)
    connection_ptr get_con_from_hdl(connection_hdl hdl) {
        lib::error_code ec;
        connection_ptr con = this->get_con_from_hdl(hdl,ec);
        if (ec) {
            throw exception(ec);
        }
        return con;
    }
protected:
    connection_ptr create_connection();

    alog_type m_alog;
    elog_type m_elog;
private:
    // dynamic settings
    std::string                 m_user_agent;

    open_handler                m_open_handler;
    close_handler               m_close_handler;
    fail_handler                m_fail_handler;
    ping_handler                m_ping_handler;
    pong_handler                m_pong_handler;
    pong_timeout_handler        m_pong_timeout_handler;
    interrupt_handler           m_interrupt_handler;
    http_handler                m_http_handler;
    validate_handler            m_validate_handler;
    message_handler             m_message_handler;

    long                        m_open_handshake_timeout_dur;
    long                        m_close_handshake_timeout_dur;
    long                        m_pong_timeout_dur;
    size_t                      m_max_message_size;
    size_t                      m_max_http_body_size;

    rng_type m_rng;

    // static settings
    bool const                  m_is_server;

    // endpoint state
    mutable mutex_type          m_mutex;
};

} // namespace websocketpp

#include <websocketpp/impl/endpoint_impl.hpp>

#endif // WEBSOCKETPP_ENDPOINT_HPP
