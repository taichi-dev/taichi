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

#ifndef WEBSOCKETPP_CONNECTION_HPP
#define WEBSOCKETPP_CONNECTION_HPP

#include <websocketpp/close.hpp>
#include <websocketpp/error.hpp>
#include <websocketpp/frame.hpp>

#include <websocketpp/logger/levels.hpp>
#include <websocketpp/processors/processor.hpp>
#include <websocketpp/transport/base/connection.hpp>
#include <websocketpp/http/constants.hpp>

#include <websocketpp/common/connection_hdl.hpp>
#include <websocketpp/common/cpp11.hpp>
#include <websocketpp/common/functional.hpp>

#include <queue>
#include <sstream>
#include <string>
#include <vector>

namespace websocketpp {

/// The type and function signature of an open handler
/**
 * The open handler is called once for every successful WebSocket connection
 * attempt. Either the fail handler or the open handler will be called for each
 * WebSocket connection attempt. HTTP Connections that did not attempt to
 * upgrade the connection to the WebSocket protocol will trigger the http
 * handler instead of fail/open.
 */
typedef lib::function<void(connection_hdl)> open_handler;

/// The type and function signature of a close handler
/**
 * The close handler is called once for every successfully established
 * connection after it is no longer capable of sending or receiving new messages
 *
 * The close handler will be called exactly once for every connection for which
 * the open handler was called.
 */
typedef lib::function<void(connection_hdl)> close_handler;

/// The type and function signature of a fail handler
/**
 * The fail handler is called once for every unsuccessful WebSocket connection
 * attempt. Either the fail handler or the open handler will be called for each
 * WebSocket connection attempt. HTTP Connections that did not attempt to
 * upgrade the connection to the WebSocket protocol will trigger the http
 * handler instead of fail/open.
 */
typedef lib::function<void(connection_hdl)> fail_handler;

/// The type and function signature of an interrupt handler
/**
 * The interrupt handler is called when a connection receives an interrupt
 * request from the application. Interrupts allow the application to trigger a
 * handler to be run in the absense of a WebSocket level handler trigger (like
 * a new message).
 *
 * This is typically used by another application thread to schedule some tasks
 * that can only be run from within the handler chain for thread safety reasons.
 */
typedef lib::function<void(connection_hdl)> interrupt_handler;

/// The type and function signature of a ping handler
/**
 * The ping handler is called when the connection receives a WebSocket ping
 * control frame. The string argument contains the ping payload. The payload is
 * a binary string up to 126 bytes in length. The ping handler returns a bool,
 * true if a pong response should be sent, false if the pong response should be
 * suppressed.
 */
typedef lib::function<bool(connection_hdl,std::string)> ping_handler;

/// The type and function signature of a pong handler
/**
 * The pong handler is called when the connection receives a WebSocket pong
 * control frame. The string argument contains the pong payload. The payload is
 * a binary string up to 126 bytes in length.
 */
typedef lib::function<void(connection_hdl,std::string)> pong_handler;

/// The type and function signature of a pong timeout handler
/**
 * The pong timeout handler is called when a ping goes unanswered by a pong for
 * longer than the locally specified timeout period.
 */
typedef lib::function<void(connection_hdl,std::string)> pong_timeout_handler;

/// The type and function signature of a validate handler
/**
 * The validate handler is called after a WebSocket handshake has been received
 * and processed but before it has been accepted. This gives the application a
 * chance to implement connection details specific policies for accepting
 * connections and the ability to negotiate extensions and subprotocols.
 *
 * The validate handler return value indicates whether or not the connection
 * should be accepted. Additional methods may be called during the function to
 * set response headers, set HTTP return/error codes, etc.
 */
typedef lib::function<bool(connection_hdl)> validate_handler;

/// The type and function signature of a http handler
/**
 * The http handler is called when an HTTP connection is made that does not
 * attempt to upgrade the connection to the WebSocket protocol. This allows
 * WebSocket++ servers to respond to these requests with regular HTTP responses.
 *
 * This can be used to deliver error pages & dashboards and to deliver static
 * files such as the base HTML & JavaScript for an otherwise single page
 * WebSocket application.
 *
 * Note: WebSocket++ is designed to be a high performance WebSocket server. It
 * is not tuned to provide a full featured, high performance, HTTP web server
 * solution. The HTTP handler is appropriate only for low volume HTTP traffic.
 * If you expect to serve high volumes of HTTP traffic a dedicated HTTP web
 * server is strongly recommended.
 *
 * The default HTTP handler will return a 426 Upgrade Required error. Custom
 * handlers may override the response status code to deliver any type of
 * response.
 */
typedef lib::function<void(connection_hdl)> http_handler;

//
typedef lib::function<void(lib::error_code const & ec, size_t bytes_transferred)> read_handler;
typedef lib::function<void(lib::error_code const & ec)> write_frame_handler;

// constants related to the default WebSocket protocol versions available
#ifdef _WEBSOCKETPP_INITIALIZER_LISTS_ // simplified C++11 version
    /// Container that stores the list of protocol versions supported
    /**
     * @todo Move this to configs to allow compile/runtime disabling or enabling
     * of protocol versions
     */
    static std::vector<int> const versions_supported = {0,7,8,13};
#else
    /// Helper array to get around lack of initializer lists pre C++11
    static int const helper[] = {0,7,8,13};
    /// Container that stores the list of protocol versions supported
    /**
     * @todo Move this to configs to allow compile/runtime disabling or enabling
     * of protocol versions
     */
    static std::vector<int> const versions_supported(helper,helper+4);
#endif

namespace session {
namespace state {
    // externally visible session state (states based on the RFC)
    enum value {
        connecting = 0,
        open = 1,
        closing = 2,
        closed = 3
    };
} // namespace state


namespace fail {
namespace status {
    enum value {
        GOOD = 0,           // no failure yet!
        SYSTEM = 1,         // system call returned error, check that code
        WEBSOCKET = 2,      // websocket close codes contain error
        UNKNOWN = 3,        // No failure information is available
        TIMEOUT_TLS = 4,    // TLS handshake timed out
        TIMEOUT_WS = 5      // WS handshake timed out
    };
} // namespace status
} // namespace fail

namespace internal_state {
    // More granular internal states. These are used for multi-threaded
    // connection synchronization and preventing values that are not yet or no
    // longer available from being used.

    enum value {
        USER_INIT = 0,
        TRANSPORT_INIT = 1,
        READ_HTTP_REQUEST = 2,
        WRITE_HTTP_REQUEST = 3,
        READ_HTTP_RESPONSE = 4,
        WRITE_HTTP_RESPONSE = 5,
        PROCESS_HTTP_REQUEST = 6,
        PROCESS_CONNECTION = 7
    };
} // namespace internal_state


namespace http_state {
    // states to keep track of the progress of http connections

    enum value {
        init = 0,
        deferred = 1,
        headers_written = 2,
        body_written = 3,
        closed = 4
    };
} // namespace http_state

} // namespace session

/// Represents an individual WebSocket connection
template <typename config>
class connection
 : public config::transport_type::transport_con_type
 , public config::connection_base
{
public:
    /// Type of this connection
    typedef connection<config> type;
    /// Type of a shared pointer to this connection
    typedef lib::shared_ptr<type> ptr;
    /// Type of a weak pointer to this connection
    typedef lib::weak_ptr<type> weak_ptr;

    /// Type of the concurrency component of this connection
    typedef typename config::concurrency_type concurrency_type;
    /// Type of the access logging policy
    typedef typename config::alog_type alog_type;
    /// Type of the error logging policy
    typedef typename config::elog_type elog_type;

    /// Type of the transport component of this connection
    typedef typename config::transport_type::transport_con_type
        transport_con_type;
    /// Type of a shared pointer to the transport component of this connection
    typedef typename transport_con_type::ptr transport_con_ptr;

    typedef lib::function<void(ptr)> termination_handler;

    typedef typename concurrency_type::scoped_lock_type scoped_lock_type;
    typedef typename concurrency_type::mutex_type mutex_type;

    typedef typename config::request_type request_type;
    typedef typename config::response_type response_type;

    typedef typename config::message_type message_type;
    typedef typename message_type::ptr message_ptr;

    typedef typename config::con_msg_manager_type con_msg_manager_type;
    typedef typename con_msg_manager_type::ptr con_msg_manager_ptr;

    /// Type of RNG
    typedef typename config::rng_type rng_type;

    typedef processor::processor<config> processor_type;
    typedef lib::shared_ptr<processor_type> processor_ptr;

    // Message handler (needs to know message type)
    typedef lib::function<void(connection_hdl,message_ptr)> message_handler;

    /// Type of a pointer to a transport timer handle
    typedef typename transport_con_type::timer_ptr timer_ptr;

    // Misc Convenience Types
    typedef session::internal_state::value istate_type;

private:
    enum terminate_status {
        failed = 1,
        closed,
        unknown
    };
public:

    explicit connection(bool p_is_server, std::string const & ua, alog_type& alog,
        elog_type& elog, rng_type & rng)
      : transport_con_type(p_is_server, alog, elog)
      , m_handle_read_frame(lib::bind(
            &type::handle_read_frame,
            this,
            lib::placeholders::_1,
            lib::placeholders::_2
        ))
      , m_write_frame_handler(lib::bind(
            &type::handle_write_frame,
            this,
            lib::placeholders::_1
        ))
      , m_user_agent(ua)
      , m_open_handshake_timeout_dur(config::timeout_open_handshake)
      , m_close_handshake_timeout_dur(config::timeout_close_handshake)
      , m_pong_timeout_dur(config::timeout_pong)
      , m_max_message_size(config::max_message_size)
      , m_state(session::state::connecting)
      , m_internal_state(session::internal_state::USER_INIT)
      , m_msg_manager(new con_msg_manager_type())
      , m_send_buffer_size(0)
      , m_write_flag(false)
      , m_read_flag(true)
      , m_is_server(p_is_server)
      , m_alog(alog)
      , m_elog(elog)
      , m_rng(rng)
      , m_local_close_code(close::status::abnormal_close)
      , m_remote_close_code(close::status::abnormal_close)
      , m_is_http(false)
      , m_http_state(session::http_state::init)
      , m_was_clean(false)
    {
        m_alog.write(log::alevel::devel,"connection constructor");
    }

    /// Get a shared pointer to this component
    ptr get_shared() {
        return lib::static_pointer_cast<type>(transport_con_type::get_shared());
    }

    ///////////////////////////
    // Set Handler Callbacks //
    ///////////////////////////

    /// Set open handler
    /**
     * The open handler is called after the WebSocket handshake is complete and
     * the connection is considered OPEN.
     *
     * @param h The new open_handler
     */
    void set_open_handler(open_handler h) {
        m_open_handler = h;
    }

    /// Set close handler
    /**
     * The close handler is called immediately after the connection is closed.
     *
     * @param h The new close_handler
     */
    void set_close_handler(close_handler h) {
        m_close_handler = h;
    }

    /// Set fail handler
    /**
     * The fail handler is called whenever the connection fails while the
     * handshake is bring processed.
     *
     * @param h The new fail_handler
     */
    void set_fail_handler(fail_handler h) {
        m_fail_handler = h;
    }

    /// Set ping handler
    /**
     * The ping handler is called whenever the connection receives a ping
     * control frame. The ping payload is included.
     *
     * The ping handler's return time controls whether or not a pong is
     * sent in response to this ping. Returning false will suppress the
     * return pong. If no ping handler is set a pong will be sent.
     *
     * @param h The new ping_handler
     */
    void set_ping_handler(ping_handler h) {
        m_ping_handler = h;
    }

    /// Set pong handler
    /**
     * The pong handler is called whenever the connection receives a pong
     * control frame. The pong payload is included.
     *
     * @param h The new pong_handler
     */
    void set_pong_handler(pong_handler h) {
        m_pong_handler = h;
    }

    /// Set pong timeout handler
    /**
     * If the transport component being used supports timers, the pong timeout
     * handler is called whenever a pong control frame is not received with the
     * configured timeout period after the application sends a ping.
     *
     * The config setting `timeout_pong` controls the length of the timeout
     * period. It is specified in milliseconds.
     *
     * This can be used to probe the health of the remote endpoint's WebSocket
     * implementation. This does not guarantee that the remote application
     * itself is still healthy but can be a useful diagnostic.
     *
     * Note: receipt of this callback doesn't mean the pong will never come.
     * This functionality will not suppress delivery of the pong in question
     * should it arrive after the timeout.
     *
     * @param h The new pong_timeout_handler
     */
    void set_pong_timeout_handler(pong_timeout_handler h) {
        m_pong_timeout_handler = h;
    }

    /// Set interrupt handler
    /**
     * The interrupt handler is called whenever the connection is manually
     * interrupted by the application.
     *
     * @param h The new interrupt_handler
     */
    void set_interrupt_handler(interrupt_handler h) {
        m_interrupt_handler = h;
    }

    /// Set http handler
    /**
     * The http handler is called after an HTTP request other than a WebSocket
     * upgrade request is received. It allows a WebSocket++ server to respond
     * to regular HTTP requests on the same port as it processes WebSocket
     * connections. This can be useful for hosting error messages, flash
     * policy files, status pages, and other simple HTTP responses. It is not
     * intended to be used as a primary web server.
     *
     * @param h The new http_handler
     */
    void set_http_handler(http_handler h) {
        m_http_handler = h;
    }

    /// Set validate handler
    /**
     * The validate handler is called after a WebSocket handshake has been
     * parsed but before a response is returned. It provides the application
     * a chance to examine the request and determine whether or not it wants
     * to accept the connection.
     *
     * Returning false from the validate handler will reject the connection.
     * If no validate handler is present, all connections will be allowed.
     *
     * @param h The new validate_handler
     */
    void set_validate_handler(validate_handler h) {
        m_validate_handler = h;
    }

    /// Set message handler
    /**
     * The message handler is called after a new message has been received.
     *
     * @param h The new message_handler
     */
    void set_message_handler(message_handler h) {
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
        m_pong_timeout_dur = dur;
    }

    /// Get maximum message size
    /**
     * Get maximum message size. Maximum message size determines the point at 
     * which the connection will fail with the message_too_big protocol error.
     *
     * The default is set by the endpoint that creates the connection.
     *
     * @since 0.3.0
     */
    size_t get_max_message_size() const {
        return m_max_message_size;
    }
    
    /// Set maximum message size
    /**
     * Set maximum message size. Maximum message size determines the point at 
     * which the connection will fail with the message_too_big protocol error. 
     * This value may be changed during the connection.
     *
     * The default is set by the endpoint that creates the connection.
     *
     * @since 0.3.0
     *
     * @param new_value The value to set as the maximum message size.
     */
    void set_max_message_size(size_t new_value) {
        m_max_message_size = new_value;
        if (m_processor) {
            m_processor->set_max_message_size(new_value);
        }
    }
    
    /// Get maximum HTTP message body size
    /**
     * Get maximum HTTP message body size. Maximum message body size determines
     * the point at which the connection will stop reading an HTTP request whose
     * body is too large.
     *
     * The default is set by the endpoint that creates the connection.
     *
     * @since 0.5.0
     *
     * @return The maximum HTTP message body size
     */
    size_t get_max_http_body_size() const {
        return m_request.get_max_body_size();
    }
    
    /// Set maximum HTTP message body size
    /**
     * Set maximum HTTP message body size. Maximum message body size determines
     * the point at which the connection will stop reading an HTTP request whose
     * body is too large.
     *
     * The default is set by the endpoint that creates the connection.
     *
     * @since 0.5.0
     *
     * @param new_value The value to set as the maximum message size.
     */
    void set_max_http_body_size(size_t new_value) {
        m_request.set_max_body_size(new_value);
    }

    //////////////////////////////////
    // Uncategorized public methods //
    //////////////////////////////////

    /// Get the size of the outgoing write buffer (in payload bytes)
    /**
     * Retrieves the number of bytes in the outgoing write buffer that have not
     * already been dispatched to the transport layer. This represents the bytes
     * that are presently cancelable without uncleanly ending the websocket
     * connection
     *
     * This method invokes the m_write_lock mutex
     *
     * @return The current number of bytes in the outgoing send buffer.
     */
    size_t get_buffered_amount() const;

    /// Get the size of the outgoing write buffer (in payload bytes)
    /**
     * @deprecated use `get_buffered_amount` instead
     */
    size_t buffered_amount() const {
        return get_buffered_amount();
    }

    ////////////////////
    // Action Methods //
    ////////////////////

    /// Create a message and then add it to the outgoing send queue
    /**
     * Convenience method to send a message given a payload string and
     * optionally an opcode. Default opcode is utf8 text.
     *
     * This method locks the m_write_lock mutex
     *
     * @param payload The payload string to generated the message with
     *
     * @param op The opcode to generated the message with. Default is
     * frame::opcode::text
     */
    lib::error_code send(std::string const & payload, frame::opcode::value op =
        frame::opcode::text);

    /// Send a message (raw array overload)
    /**
     * Convenience method to send a message given a raw array and optionally an
     * opcode. Default opcode is binary.
     *
     * This method locks the m_write_lock mutex
     *
     * @param payload A pointer to the array containing the bytes to send.
     *
     * @param len Length of the array.
     *
     * @param op The opcode to generated the message with. Default is
     * frame::opcode::binary
     */
    lib::error_code send(void const * payload, size_t len, frame::opcode::value
        op = frame::opcode::binary);

    /// Add a message to the outgoing send queue
    /**
     * If presented with a prepared message it is added without validation or
     * framing. If presented with an unprepared message it is validated, framed,
     * and then added
     *
     * Errors are returned via an exception
     * \todo make exception system_error rather than error_code
     *
     * This method invokes the m_write_lock mutex
     *
     * @param msg A message_ptr to the message to send.
     */
    lib::error_code send(message_ptr msg);

    /// Asyncronously invoke handler::on_inturrupt
    /**
     * Signals to the connection to asyncronously invoke the on_inturrupt
     * callback for this connection's handler once it is safe to do so.
     *
     * When the on_inturrupt handler callback is called it will be from
     * within the transport event loop with all the thread safety features
     * guaranteed by the transport to regular handlers
     *
     * Multiple inturrupt signals can be active at once on the same connection
     *
     * @return An error code
     */
    lib::error_code interrupt();
    
    /// Transport inturrupt callback
    void handle_interrupt();
    
    /// Pause reading of new data
    /**
     * Signals to the connection to halt reading of new data. While reading is paused, 
     * the connection will stop reading from its associated socket. In turn this will 
     * result in TCP based flow control kicking in and slowing data flow from the remote
     * endpoint.
     *
     * This is useful for applications that push new requests to a queue to be processed
     * by another thread and need a way to signal when their request queue is full without
     * blocking the network processing thread.
     *
     * Use `resume_reading()` to resume.
     *
     * If supported by the transport this is done asynchronously. As such reading may not
     * stop until the current read operation completes. Typically you can expect to
     * receive no more bytes after initiating a read pause than the size of the read 
     * buffer.
     *
     * If reading is paused for this connection already nothing is changed.
     */
    lib::error_code pause_reading();

    /// Pause reading callback
    void handle_pause_reading();

    /// Resume reading of new data
    /**
     * Signals to the connection to resume reading of new data after it was paused by
     * `pause_reading()`.
     *
     * If reading is not paused for this connection already nothing is changed.
     */
    lib::error_code resume_reading();

    /// Resume reading callback
    void handle_resume_reading();

    /// Send a ping
    /**
     * Initiates a ping with the given payload/
     *
     * There is no feedback directly from ping except in cases of immediately
     * detectable errors. Feedback will be provided via on_pong or
     * on_pong_timeout callbacks.
     *
     * Ping locks the m_write_lock mutex
     *
     * @param payload Payload to be used for the ping
     */
    void ping(std::string const & payload);

    /// exception free variant of ping
    void ping(std::string const & payload, lib::error_code & ec);

    /// Utility method that gets called back when the ping timer expires
    void handle_pong_timeout(std::string payload, lib::error_code const & ec);

    /// Send a pong
    /**
     * Initiates a pong with the given payload.
     *
     * There is no feedback from a pong once sent.
     *
     * Pong locks the m_write_lock mutex
     *
     * @param payload Payload to be used for the pong
     */
    void pong(std::string const & payload);

    /// exception free variant of pong
    void pong(std::string const & payload, lib::error_code & ec);

    /// Close the connection
    /**
     * Initiates the close handshake process.
     *
     * If close returns successfully the connection will be in the closing
     * state and no additional messages may be sent. All messages sent prior
     * to calling close will be written out before the connection is closed.
     *
     * If no reason is specified none will be sent. If no code is specified
     * then no code will be sent.
     *
     * The handler's on_close callback will be called once the close handshake
     * is complete.
     *
     * Reasons will be automatically truncated to the maximum length (123 bytes)
     * if necessary.
     *
     * @param code The close code to send
     * @param reason The close reason to send
     */
    void close(close::status::value const code, std::string const & reason);

    /// exception free variant of close
    void close(close::status::value const code, std::string const & reason,
        lib::error_code & ec);

    ////////////////////////////////////////////////
    // Pass-through access to the uri information //
    ////////////////////////////////////////////////

    /// Returns the secure flag from the connection URI
    /**
     * This value is available after the HTTP request has been fully read and
     * may be called from any thread.
     *
     * @return Whether or not the connection URI is flagged secure.
     */
    bool get_secure() const;

    /// Returns the host component of the connection URI
    /**
     * This value is available after the HTTP request has been fully read and
     * may be called from any thread.
     *
     * @return The host component of the connection URI
     */
    std::string const & get_host() const;

    /// Returns the resource component of the connection URI
    /**
     * This value is available after the HTTP request has been fully read and
     * may be called from any thread.
     *
     * @return The resource component of the connection URI
     */
    std::string const & get_resource() const;

    /// Returns the port component of the connection URI
    /**
     * This value is available after the HTTP request has been fully read and
     * may be called from any thread.
     *
     * @return The port component of the connection URI
     */
    uint16_t get_port() const;

    /// Gets the connection URI
    /**
     * This should really only be called by internal library methods unless you
     * really know what you are doing.
     *
     * @return A pointer to the connection's URI
     */
    uri_ptr get_uri() const;

    /// Sets the connection URI
    /**
     * This should really only be called by internal library methods unless you
     * really know what you are doing.
     *
     * @param uri The new URI to set
     */
    void set_uri(uri_ptr uri);

    /////////////////////////////
    // Subprotocol negotiation //
    /////////////////////////////

    /// Gets the negotated subprotocol
    /**
     * Retrieves the subprotocol that was negotiated during the handshake. This
     * method is valid in the open handler and later.
     *
     * @return The negotiated subprotocol
     */
    std::string const & get_subprotocol() const;

    /// Gets all of the subprotocols requested by the client
    /**
     * Retrieves the subprotocols that were requested during the handshake. This
     * method is valid in the validate handler and later.
     *
     * @return A vector of the requested subprotocol
     */
    std::vector<std::string> const & get_requested_subprotocols() const;

    /// Adds the given subprotocol string to the request list (exception free)
    /**
     * Adds a subprotocol to the list to send with the opening handshake. This
     * may be called multiple times to request more than one. If the server
     * supports one of these, it may choose one. If so, it will return it
     * in it's handshake reponse and the value will be available via
     * get_subprotocol(). Subprotocol requests should be added in order of
     * preference.
     *
     * @param request The subprotocol to request
     * @param ec A reference to an error code that will be filled in the case of
     * errors
     */
    void add_subprotocol(std::string const & request, lib::error_code & ec);

    /// Adds the given subprotocol string to the request list
    /**
     * Adds a subprotocol to the list to send with the opening handshake. This
     * may be called multiple times to request more than one. If the server
     * supports one of these, it may choose one. If so, it will return it
     * in it's handshake reponse and the value will be available via
     * get_subprotocol(). Subprotocol requests should be added in order of
     * preference.
     *
     * @param request The subprotocol to request
     */
    void add_subprotocol(std::string const & request);

    /// Select a subprotocol to use (exception free)
    /**
     * Indicates which subprotocol should be used for this connection. Valid
     * only during the validate handler callback. Subprotocol selected must have
     * been requested by the client. Consult get_requested_subprotocols() for a
     * list of valid subprotocols.
     *
     * This member function is valid on server endpoints/connections only
     *
     * @param value The subprotocol to select
     * @param ec A reference to an error code that will be filled in the case of
     * errors
     */
    void select_subprotocol(std::string const & value, lib::error_code & ec);

    /// Select a subprotocol to use
    /**
     * Indicates which subprotocol should be used for this connection. Valid
     * only during the validate handler callback. Subprotocol selected must have
     * been requested by the client. Consult get_requested_subprotocols() for a
     * list of valid subprotocols.
     *
     * This member function is valid on server endpoints/connections only
     *
     * @param value The subprotocol to select
     */
    void select_subprotocol(std::string const & value);

    /////////////////////////////////////////////////////////////
    // Pass-through access to the request and response objects //
    /////////////////////////////////////////////////////////////

    /// Retrieve a request header
    /**
     * Retrieve the value of a header from the handshake HTTP request.
     *
     * @param key Name of the header to get
     * @return The value of the header
     */
    std::string const & get_request_header(std::string const & key) const;

    /// Retrieve a request body
    /**
     * Retrieve the value of the request body. This value is typically used with
     * PUT and POST requests to upload files or other data. Only HTTP
     * connections will ever have bodies. WebSocket connection's will always
     * have blank bodies.
     *
     * @return The value of the request body.
     */
    std::string const & get_request_body() const;

    /// Retrieve a response header
    /**
     * Retrieve the value of a header from the handshake HTTP request.
     *
     * @param key Name of the header to get
     * @return The value of the header
     */
    std::string const & get_response_header(std::string const & key) const;

    /// Get response HTTP status code
    /**
     * Gets the response status code 
     *
     * @since 0.7.0
     *
     * @return The response status code sent
     */
    http::status_code::value get_response_code() const {
        return m_response.get_status_code();
    }

    /// Get response HTTP status message
    /**
     * Gets the response status message 
     *
     * @since 0.7.0
     *
     * @return The response status message sent
     */
    std::string const & get_response_msg() const {
        return m_response.get_status_msg();
    }
    
    /// Set response status code and message
    /**
     * Sets the response status code to `code` and looks up the corresponding
     * message for standard codes. Non-standard codes will be entered as Unknown
     * use set_status(status_code::value,std::string) overload to set both
     * values explicitly.
     *
     * This member function is valid only from the http() and validate() handler
     * callbacks.
     *
     * @param code Code to set
     * @param msg Message to set
     * @see websocketpp::http::response::set_status
     */
    void set_status(http::status_code::value code);

    /// Set response status code and message
    /**
     * Sets the response status code and message to independent custom values.
     * use set_status(status_code::value) to set the code and have the standard
     * message be automatically set.
     *
     * This member function is valid only from the http() and validate() handler
     * callbacks.
     *
     * @param code Code to set
     * @param msg Message to set
     * @see websocketpp::http::response::set_status
     */
    void set_status(http::status_code::value code, std::string const & msg);

    /// Set response body content
    /**
     * Set the body content of the HTTP response to the parameter string. Note
     * set_body will also set the Content-Length HTTP header to the appropriate
     * value. If you want the Content-Length header to be something else set it
     * to something else after calling set_body
     *
     * This member function is valid only from the http() and validate() handler
     * callbacks.
     *
     * @param value String data to include as the body content.
     * @see websocketpp::http::response::set_body
     */
    void set_body(std::string const & value);

    /// Append a header
    /**
     * If a header with this name already exists the value will be appended to
     * the existing header to form a comma separated list of values. Use
     * `connection::replace_header` to overwrite existing values.
     *
     * This member function is valid only from the http() and validate() handler
     * callbacks, or to a client connection before connect has been called.
     *
     * @param key Name of the header to set
     * @param val Value to add
     * @see replace_header
     * @see websocketpp::http::parser::append_header
     */
    void append_header(std::string const & key, std::string const & val);

    /// Replace a header
    /**
     * If a header with this name already exists the old value will be replaced
     * Use `connection::append_header` to append to a list of existing values.
     *
     * This member function is valid only from the http() and validate() handler
     * callbacks, or to a client connection before connect has been called.
     *
     * @param key Name of the header to set
     * @param val Value to set
     * @see append_header
     * @see websocketpp::http::parser::replace_header
     */
    void replace_header(std::string const & key, std::string const & val);

    /// Remove a header
    /**
     * Removes a header from the response.
     *
     * This member function is valid only from the http() and validate() handler
     * callbacks, or to a client connection before connect has been called.
     *
     * @param key The name of the header to remove
     * @see websocketpp::http::parser::remove_header
     */
    void remove_header(std::string const & key);

    /// Get request object
    /**
     * Direct access to request object. This can be used to call methods of the
     * request object that are not part of the standard request API that
     * connection wraps.
     *
     * Note use of this method involves using behavior specific to the
     * configured HTTP policy. Such behavior may not work with alternate HTTP
     * policies.
     *
     * @since 0.3.0-alpha3
     *
     * @return A const reference to the raw request object
     */
    request_type const & get_request() const {
        return m_request;
    }
    
    /// Get response object
    /**
     * Direct access to the HTTP response sent or received as a part of the
     * opening handshake. This can be used to call methods of the response
     * object that are not part of the standard request API that connection
     * wraps.
     *
     * Note use of this method involves using behavior specific to the
     * configured HTTP policy. Such behavior may not work with alternate HTTP
     * policies.
     *
     * @since 0.7.0
     *
     * @return A const reference to the raw response object
     */
    response_type const & get_response() const {
        return m_response;
    }
    
    /// Defer HTTP Response until later (Exception free)
    /**
     * Used in the http handler to defer the HTTP response for this connection
     * until later. Handshake timers will be canceled and the connection will be
     * left open until `send_http_response` or an equivalent is called.
     *
     * Warning: deferred connections won't time out and as a result can tie up
     * resources.
     *
     * @since 0.6.0
     *
     * @return A status code, zero on success, non-zero otherwise
     */
    lib::error_code defer_http_response();
    
    /// Send deferred HTTP Response (exception free)
    /**
     * Sends an http response to an HTTP connection that was deferred. This will
     * send a complete response including all headers, status line, and body
     * text. The connection will be closed afterwards.
     *
     * @since 0.6.0
     *
     * @param ec A status code, zero on success, non-zero otherwise
     */
    void send_http_response(lib::error_code & ec);
    
    /// Send deferred HTTP Response
    void send_http_response();
    
    // TODO HTTPNBIO: write_headers
    // function that processes headers + status so far and writes it to the wire
    // beginning the HTTP response body state. This method will ignore anything
    // in the response body.
    
    // TODO HTTPNBIO: write_body_message
    // queues the specified message_buffer for async writing
    
    // TODO HTTPNBIO: finish connection
    //
    
    // TODO HTTPNBIO: write_response
    // Writes the whole response, headers + body and closes the connection
    
    

    /////////////////////////////////////////////////////////////
    // Pass-through access to the other connection information //
    /////////////////////////////////////////////////////////////

    /// Get Connection Handle
    /**
     * The connection handle is a token that can be shared outside the
     * WebSocket++ core for the purposes of identifying a connection and
     * sending it messages.
     *
     * @return A handle to the connection
     */
    connection_hdl get_handle() const {
        return m_connection_hdl;
    }

    /// Get whether or not this connection is part of a server or client
    /**
     * @return whether or not the connection is attached to a server endpoint
     */
    bool is_server() const {
        return m_is_server;
    }

    /// Return the same origin policy origin value from the opening request.
    /**
     * This value is available after the HTTP request has been fully read and
     * may be called from any thread.
     *
     * @return The connection's origin value from the opening handshake.
     */
    std::string const & get_origin() const;

    /// Return the connection state.
    /**
     * Values can be connecting, open, closing, and closed
     *
     * @return The connection's current state.
     */
    session::state::value get_state() const;


    /// Get the WebSocket close code sent by this endpoint.
    /**
     * @return The WebSocket close code sent by this endpoint.
     */
    close::status::value get_local_close_code() const {
        return m_local_close_code;
    }

    /// Get the WebSocket close reason sent by this endpoint.
    /**
     * @return The WebSocket close reason sent by this endpoint.
     */
    std::string const & get_local_close_reason() const {
        return m_local_close_reason;
    }

    /// Get the WebSocket close code sent by the remote endpoint.
    /**
     * @return The WebSocket close code sent by the remote endpoint.
     */
    close::status::value get_remote_close_code() const {
        return m_remote_close_code;
    }

    /// Get the WebSocket close reason sent by the remote endpoint.
    /**
     * @return The WebSocket close reason sent by the remote endpoint.
     */
    std::string const & get_remote_close_reason() const {
        return m_remote_close_reason;
    }

    /// Get the internal error code for a closed/failed connection
    /**
     * Retrieves a machine readable detailed error code indicating the reason
     * that the connection was closed or failed. Valid only after the close or
     * fail handler is called.
     *
     * @return Error code indicating the reason the connection was closed or
     * failed
     */
    lib::error_code get_ec() const {
        return m_ec;
    }

    /// Get a message buffer
    /**
     * Warning: The API related to directly sending message buffers may change
     * before the 1.0 release. If you plan to use it, please keep an eye on any
     * breaking changes notifications in future release notes. Also if you have
     * any feedback about usage and capabilities now is a great time to provide
     * it.
     *
     * Message buffers are used to store message payloads and other message
     * metadata.
     *
     * The size parameter is a hint only. Your final payload does not need to
     * match it. There may be some performance benefits if the initial size
     * guess is equal to or slightly higher than the final payload size.
     *
     * @param op The opcode for the new message
     * @param size A hint to optimize the initial allocation of payload space.
     * @return A new message buffer
     */
    message_ptr get_message(websocketpp::frame::opcode::value op, size_t size)
        const
    {
        return m_msg_manager->get_message(op, size);
    }

    ////////////////////////////////////////////////////////////////////////
    // The remaining public member functions are for internal/policy use  //
    // only. Do not call from application code unless you understand what //
    // you are doing.                                                     //
    ////////////////////////////////////////////////////////////////////////

    

    void read_handshake(size_t num_bytes);

    void handle_read_handshake(lib::error_code const & ec,
        size_t bytes_transferred);
    void handle_read_http_response(lib::error_code const & ec,
        size_t bytes_transferred);

    
    void handle_write_http_response(lib::error_code const & ec);
    void handle_send_http_request(lib::error_code const & ec);

    void handle_open_handshake_timeout(lib::error_code const & ec);
    void handle_close_handshake_timeout(lib::error_code const & ec);

    void handle_read_frame(lib::error_code const & ec, size_t bytes_transferred);
    void read_frame();

    /// Get array of WebSocket protocol versions that this connection supports.
    std::vector<int> const & get_supported_versions() const;

    /// Sets the handler for a terminating connection. Should only be used
    /// internally by the endpoint class.
    void set_termination_handler(termination_handler new_handler);

    void terminate(lib::error_code const & ec);
    void handle_terminate(terminate_status tstat, lib::error_code const & ec);

    /// Checks if there are frames in the send queue and if there are sends one
    /**
     * \todo unit tests
     *
     * This method locks the m_write_lock mutex
     */
    void write_frame();

    /// Process the results of a frame write operation and start the next write
    /**
     * \todo unit tests
     *
     * This method locks the m_write_lock mutex
     *
     * @param terminate Whether or not to terminate the connection upon
     * completion of this write.
     *
     * @param ec A status code from the transport layer, zero on success,
     * non-zero otherwise.
     */
    void handle_write_frame(lib::error_code const & ec);
// protected:
    // This set of methods would really like to be protected, but doing so 
    // requires that the endpoint be able to friend the connection. This is 
    // allowed with C++11, but not prior versions

    /// Start the connection state machine
    void start();

    /// Set Connection Handle
    /**
     * The connection handle is a token that can be shared outside the
     * WebSocket++ core for the purposes of identifying a connection and
     * sending it messages.
     *
     * @param hdl A connection_hdl that the connection will use to refer
     * to itself.
     */
    void set_handle(connection_hdl hdl) {
        m_connection_hdl = hdl;
        transport_con_type::set_handle(hdl);
    }
protected:
    void handle_transport_init(lib::error_code const & ec);

    /// Set m_processor based on information in m_request. Set m_response
    /// status and return an error code indicating status.
    lib::error_code initialize_processor();

    /// Perform WebSocket handshake validation of m_request using m_processor.
    /// set m_response and return an error code indicating status.
    lib::error_code process_handshake_request();
private:
    

    /// Completes m_response, serializes it, and sends it out on the wire.
    void write_http_response(lib::error_code const & ec);

    /// Sends an opening WebSocket connect request
    void send_http_request();

    /// Alternate path for write_http_response in error conditions
    void write_http_response_error(lib::error_code const & ec);

    /// Process control message
    /**
     *
     */
    void process_control_frame(message_ptr msg);

    /// Send close acknowledgement
    /**
     * If no arguments are present no close code/reason will be specified.
     *
     * Note: the close code/reason values provided here may be overrided by
     * other settings (such as silent close).
     *
     * @param code The close code to send
     * @param reason The close reason to send
     * @return A status code, zero on success, non-zero otherwise
     */
    lib::error_code send_close_ack(close::status::value code =
        close::status::blank, std::string const & reason = std::string());

    /// Send close frame
    /**
     * If no arguments are present no close code/reason will be specified.
     *
     * Note: the close code/reason values provided here may be overrided by
     * other settings (such as silent close).
     *
     * The ack flag determines what to do in the case of a blank status and
     * whether or not to terminate the TCP connection after sending it.
     *
     * @param code The close code to send
     * @param reason The close reason to send
     * @param ack Whether or not this is an acknowledgement close frame
     * @return A status code, zero on success, non-zero otherwise
     */
    lib::error_code send_close_frame(close::status::value code =
        close::status::blank, std::string const & reason = std::string(), bool ack = false,
        bool terminal = false);

    /// Get a pointer to a new WebSocket protocol processor for a given version
    /**
     * @param version Version number of the WebSocket protocol to get a
     * processor for. Negative values indicate invalid/unknown versions and will
     * always return a null ptr
     *
     * @return A shared_ptr to a new instance of the appropriate processor or a
     * null ptr if there is no installed processor that matches the version
     * number.
     */
    processor_ptr get_processor(int version) const;

    /// Add a message to the write queue
    /**
     * Adds a message to the write queue and updates any associated shared state
     *
     * Must be called while holding m_write_lock
     *
     * @todo unit tests
     *
     * @param msg The message to push
     */
    void write_push(message_ptr msg);

    /// Pop a message from the write queue
    /**
     * Removes and returns a message from the write queue and updates any
     * associated shared state.
     *
     * Must be called while holding m_write_lock
     *
     * @todo unit tests
     *
     * @return the message_ptr at the front of the queue
     */
    message_ptr write_pop();

    /// Prints information about the incoming connection to the access log
    /**
     * Prints information about the incoming connection to the access log.
     * Includes: connection type, websocket version, remote endpoint, user agent
     * path, status code.
     */
    void log_open_result();

    /// Prints information about a connection being closed to the access log
    /**
     * Includes: local and remote close codes and reasons
     */
    void log_close_result();

    /// Prints information about a connection being failed to the access log
    /**
     * Includes: error code and message for why it was failed
     */
    void log_fail_result();
    
    /// Prints information about HTTP connections
    /**
     * Includes: TODO
     */
    void log_http_result();

    /// Prints information about an arbitrary error code on the specified channel
    template <typename error_type>
    void log_err(log::level l, char const * msg, error_type const & ec) {
        std::stringstream s;
        s << msg << " error: " << ec << " (" << ec.message() << ")";
        m_elog.write(l, s.str());
    }

    // internal handler functions
    read_handler            m_handle_read_frame;
    write_frame_handler     m_write_frame_handler;

    // static settings
    std::string const       m_user_agent;

    /// Pointer to the connection handle
    connection_hdl          m_connection_hdl;

    /// Handler objects
    open_handler            m_open_handler;
    close_handler           m_close_handler;
    fail_handler            m_fail_handler;
    ping_handler            m_ping_handler;
    pong_handler            m_pong_handler;
    pong_timeout_handler    m_pong_timeout_handler;
    interrupt_handler       m_interrupt_handler;
    http_handler            m_http_handler;
    validate_handler        m_validate_handler;
    message_handler         m_message_handler;

    /// constant values
    long                    m_open_handshake_timeout_dur;
    long                    m_close_handshake_timeout_dur;
    long                    m_pong_timeout_dur;
    size_t                  m_max_message_size;

    /// External connection state
    /**
     * Lock: m_connection_state_lock
     */
    session::state::value   m_state;

    /// Internal connection state
    /**
     * Lock: m_connection_state_lock
     */
    istate_type             m_internal_state;

    mutable mutex_type      m_connection_state_lock;

    /// The lock used to protect the message queue
    /**
     * Serializes access to the write queue as well as shared state within the
     * processor.
     */
    mutex_type              m_write_lock;

    // connection resources
    char                    m_buf[config::connection_read_buffer_size];
    size_t                  m_buf_cursor;
    termination_handler     m_termination_handler;
    con_msg_manager_ptr     m_msg_manager;
    timer_ptr               m_handshake_timer;
    timer_ptr               m_ping_timer;

    /// @todo this is not memory efficient. this value is not used after the
    /// handshake.
    std::string m_handshake_buffer;

    /// Pointer to the processor object for this connection
    /**
     * The processor provides functionality that is specific to the WebSocket
     * protocol version that the client has negotiated. It also contains all of
     * the state necessary to encode and decode the incoming and outgoing
     * WebSocket byte streams
     *
     * Use of the prepare_data_frame method requires lock: m_write_lock
     */
    processor_ptr           m_processor;

    /// Queue of unsent outgoing messages
    /**
     * Lock: m_write_lock
     */
    std::queue<message_ptr> m_send_queue;

    /// Size in bytes of the outstanding payloads in the write queue
    /**
     * Lock: m_write_lock
     */
    size_t m_send_buffer_size;

    /// buffer holding the various parts of the current message being writen
    /**
     * Lock m_write_lock
     */
    std::vector<transport::buffer> m_send_buffer;

    /// a list of pointers to hold on to the messages being written to keep them
    /// from going out of scope before the write is complete.
    std::vector<message_ptr> m_current_msgs;

    /// True if there is currently an outstanding transport write
    /**
     * Lock m_write_lock
     */
    bool m_write_flag;

    /// True if this connection is presently reading new data
    bool m_read_flag;

    // connection data
    request_type            m_request;
    response_type           m_response;
    uri_ptr                 m_uri;
    std::string             m_subprotocol;

    // connection data that might not be necessary to keep around for the life
    // of the whole connection.
    std::vector<std::string> m_requested_subprotocols;

    bool const              m_is_server;
    alog_type& m_alog;
    elog_type& m_elog;

    rng_type & m_rng;

    // Close state
    /// Close code that was sent on the wire by this endpoint
    close::status::value    m_local_close_code;

    /// Close reason that was sent on the wire by this endpoint
    std::string             m_local_close_reason;

    /// Close code that was received on the wire from the remote endpoint
    close::status::value    m_remote_close_code;

    /// Close reason that was received on the wire from the remote endpoint
    std::string             m_remote_close_reason;

    /// Detailed internal error code
    lib::error_code m_ec;
    
    /// A flag that gets set once it is determined that the connection is an
    /// HTTP connection and not a WebSocket one.
    bool m_is_http;
    
    /// A flag that gets set when the completion of an http connection is
    /// deferred until later.
    session::http_state::value m_http_state;

    bool m_was_clean;

    /// Whether or not this endpoint initiated the closing handshake.
    bool                    m_closed_by_me;

    /// ???
    bool                    m_failed_by_me;

    /// Whether or not this endpoint initiated the drop of the TCP connection
    bool                    m_dropped_by_me;
};

} // namespace websocketpp

#include <websocketpp/impl/connection_impl.hpp>

#endif // WEBSOCKETPP_CONNECTION_HPP
