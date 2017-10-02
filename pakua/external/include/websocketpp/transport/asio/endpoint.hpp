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

#ifndef WEBSOCKETPP_TRANSPORT_ASIO_HPP
#define WEBSOCKETPP_TRANSPORT_ASIO_HPP

#include <websocketpp/transport/base/endpoint.hpp>
#include <websocketpp/transport/asio/connection.hpp>
#include <websocketpp/transport/asio/security/none.hpp>

#include <websocketpp/uri.hpp>
#include <websocketpp/logger/levels.hpp>

#include <websocketpp/common/functional.hpp>

#include <sstream>
#include <string>

namespace websocketpp {
namespace transport {
namespace asio {

/// Asio based endpoint transport component
/**
 * transport::asio::endpoint implements an endpoint transport component using
 * Asio.
 */
template <typename config>
class endpoint : public config::socket_type {
public:
    /// Type of this endpoint transport component
    typedef endpoint<config> type;

    /// Type of the concurrency policy
    typedef typename config::concurrency_type concurrency_type;
    /// Type of the socket policy
    typedef typename config::socket_type socket_type;
    /// Type of the error logging policy
    typedef typename config::elog_type elog_type;
    /// Type of the access logging policy
    typedef typename config::alog_type alog_type;

    /// Type of the socket connection component
    typedef typename socket_type::socket_con_type socket_con_type;
    /// Type of a shared pointer to the socket connection component
    typedef typename socket_con_type::ptr socket_con_ptr;

    /// Type of the connection transport component associated with this
    /// endpoint transport component
    typedef asio::connection<config> transport_con_type;
    /// Type of a shared pointer to the connection transport component
    /// associated with this endpoint transport component
    typedef typename transport_con_type::ptr transport_con_ptr;

    /// Type of a pointer to the ASIO io_service being used
    typedef lib::asio::io_service * io_service_ptr;
    /// Type of a shared pointer to the acceptor being used
    typedef lib::shared_ptr<lib::asio::ip::tcp::acceptor> acceptor_ptr;
    /// Type of a shared pointer to the resolver being used
    typedef lib::shared_ptr<lib::asio::ip::tcp::resolver> resolver_ptr;
    /// Type of timer handle
    typedef lib::shared_ptr<lib::asio::steady_timer> timer_ptr;
    /// Type of a shared pointer to an io_service work object
    typedef lib::shared_ptr<lib::asio::io_service::work> work_ptr;

    // generate and manage our own io_service
    explicit endpoint()
      : m_io_service(NULL)
      , m_external_io_service(false)
      , m_listen_backlog(0)
      , m_reuse_addr(false)
      , m_state(UNINITIALIZED)
    {
        //std::cout << "transport::asio::endpoint constructor" << std::endl;
    }

    ~endpoint() {
        // clean up our io_service if we were initialized with an internal one.

        // Explicitly destroy local objects
        m_acceptor.reset();
        m_resolver.reset();
        m_work.reset();
        if (m_state != UNINITIALIZED && !m_external_io_service) {
            delete m_io_service;
        }
    }

    /// transport::asio objects are moveable but not copyable or assignable.
    /// The following code sets this situation up based on whether or not we
    /// have C++11 support or not
#ifdef _WEBSOCKETPP_DEFAULT_DELETE_FUNCTIONS_
    endpoint(const endpoint & src) = delete;
    endpoint& operator= (const endpoint & rhs) = delete;
#else
private:
    endpoint(const endpoint & src);
    endpoint & operator= (const endpoint & rhs);
public:
#endif // _WEBSOCKETPP_DEFAULT_DELETE_FUNCTIONS_

#ifdef _WEBSOCKETPP_MOVE_SEMANTICS_
    endpoint (endpoint && src)
      : config::socket_type(std::move(src))
      , m_tcp_pre_init_handler(src.m_tcp_pre_init_handler)
      , m_tcp_post_init_handler(src.m_tcp_post_init_handler)
      , m_io_service(src.m_io_service)
      , m_external_io_service(src.m_external_io_service)
      , m_acceptor(src.m_acceptor)
      , m_listen_backlog(lib::asio::socket_base::max_connections)
      , m_reuse_addr(src.m_reuse_addr)
      , m_elog(src.m_elog)
      , m_alog(src.m_alog)
      , m_state(src.m_state)
    {
        src.m_io_service = NULL;
        src.m_external_io_service = false;
        src.m_acceptor = NULL;
        src.m_state = UNINITIALIZED;
    }

    /*endpoint & operator= (const endpoint && rhs) {
        if (this != &rhs) {
            m_io_service = rhs.m_io_service;
            m_external_io_service = rhs.m_external_io_service;
            m_acceptor = rhs.m_acceptor;
            m_listen_backlog = rhs.m_listen_backlog;
            m_reuse_addr = rhs.m_reuse_addr;
            m_state = rhs.m_state;

            rhs.m_io_service = NULL;
            rhs.m_external_io_service = false;
            rhs.m_acceptor = NULL;
            rhs.m_listen_backlog = lib::asio::socket_base::max_connections;
            rhs.m_state = UNINITIALIZED;
            
            // TODO: this needs to be updated
        }
        return *this;
    }*/
#endif // _WEBSOCKETPP_MOVE_SEMANTICS_

    /// Return whether or not the endpoint produces secure connections.
    bool is_secure() const {
        return socket_type::is_secure();
    }

    /// initialize asio transport with external io_service (exception free)
    /**
     * Initialize the ASIO transport policy for this endpoint using the provided
     * io_service object. asio_init must be called exactly once on any endpoint
     * that uses transport::asio before it can be used.
     *
     * @param ptr A pointer to the io_service to use for asio events
     * @param ec Set to indicate what error occurred, if any.
     */
    void init_asio(io_service_ptr ptr, lib::error_code & ec) {
        if (m_state != UNINITIALIZED) {
            m_elog->write(log::elevel::library,
                "asio::init_asio called from the wrong state");
            using websocketpp::error::make_error_code;
            ec = make_error_code(websocketpp::error::invalid_state);
            return;
        }

        m_alog->write(log::alevel::devel,"asio::init_asio");

        m_io_service = ptr;
        m_external_io_service = true;
        m_acceptor = lib::make_shared<lib::asio::ip::tcp::acceptor>(
            lib::ref(*m_io_service));

        m_state = READY;
        ec = lib::error_code();
    }

    /// initialize asio transport with external io_service
    /**
     * Initialize the ASIO transport policy for this endpoint using the provided
     * io_service object. asio_init must be called exactly once on any endpoint
     * that uses transport::asio before it can be used.
     *
     * @param ptr A pointer to the io_service to use for asio events
     */
    void init_asio(io_service_ptr ptr) {
        lib::error_code ec;
        init_asio(ptr,ec);
        if (ec) { throw exception(ec); }
    }

    /// Initialize asio transport with internal io_service (exception free)
    /**
     * This method of initialization will allocate and use an internally managed
     * io_service.
     *
     * @see init_asio(io_service_ptr ptr)
     *
     * @param ec Set to indicate what error occurred, if any.
     */
    void init_asio(lib::error_code & ec) {
        // Use a smart pointer until the call is successful and ownership has 
        // successfully been taken. Use unique_ptr when available.
        // TODO: remove the use of auto_ptr when C++98/03 support is no longer
        //       necessary.
#ifdef _WEBSOCKETPP_CPP11_MEMORY_
        lib::unique_ptr<lib::asio::io_service> service(new lib::asio::io_service());
#else
        lib::auto_ptr<lib::asio::io_service> service(new lib::asio::io_service());
#endif
        init_asio(service.get(), ec);
        if( !ec ) service.release(); // Call was successful, transfer ownership
        m_external_io_service = false;
    }

    /// Initialize asio transport with internal io_service
    /**
     * This method of initialization will allocate and use an internally managed
     * io_service.
     *
     * @see init_asio(io_service_ptr ptr)
     */
    void init_asio() {
        // Use a smart pointer until the call is successful and ownership has 
        // successfully been taken. Use unique_ptr when available.
        // TODO: remove the use of auto_ptr when C++98/03 support is no longer
        //       necessary.
#ifdef _WEBSOCKETPP_CPP11_MEMORY_
        lib::unique_ptr<lib::asio::io_service> service(new lib::asio::io_service());
#else
        lib::auto_ptr<lib::asio::io_service> service(new lib::asio::io_service());
#endif
        init_asio( service.get() );
        // If control got this far without an exception, then ownership has successfully been taken
        service.release();
        m_external_io_service = false;
    }

    /// Sets the tcp pre init handler
    /**
     * The tcp pre init handler is called after the raw tcp connection has been
     * established but before any additional wrappers (proxy connects, TLS
     * handshakes, etc) have been performed.
     *
     * @since 0.3.0
     *
     * @param h The handler to call on tcp pre init.
     */
    void set_tcp_pre_init_handler(tcp_init_handler h) {
        m_tcp_pre_init_handler = h;
    }

    /// Sets the tcp pre init handler (deprecated)
    /**
     * The tcp pre init handler is called after the raw tcp connection has been
     * established but before any additional wrappers (proxy connects, TLS
     * handshakes, etc) have been performed.
     *
     * @deprecated Use set_tcp_pre_init_handler instead
     *
     * @param h The handler to call on tcp pre init.
     */
    void set_tcp_init_handler(tcp_init_handler h) {
        set_tcp_pre_init_handler(h);
    }

    /// Sets the tcp post init handler
    /**
     * The tcp post init handler is called after the tcp connection has been
     * established and all additional wrappers (proxy connects, TLS handshakes,
     * etc have been performed. This is fired before any bytes are read or any
     * WebSocket specific handshake logic has been performed.
     *
     * @since 0.3.0
     *
     * @param h The handler to call on tcp post init.
     */
    void set_tcp_post_init_handler(tcp_init_handler h) {
        m_tcp_post_init_handler = h;
    }

    /// Sets the maximum length of the queue of pending connections.
    /**
     * Sets the maximum length of the queue of pending connections. Increasing
     * this will allow WebSocket++ to queue additional incoming connections.
     * Setting it higher may prevent failed connections at high connection rates
     * but may cause additional latency.
     *
     * For this value to take effect you may need to adjust operating system
     * settings.
     *
     * New values affect future calls to listen only.
     *
     * A value of zero will use the operating system default. This is the
     * default value.
     *
     * @since 0.3.0
     *
     * @param backlog The maximum length of the queue of pending connections
     */
    void set_listen_backlog(int backlog) {
        m_listen_backlog = backlog;
    }

    /// Sets whether to use the SO_REUSEADDR flag when opening listening sockets
    /**
     * Specifies whether or not to use the SO_REUSEADDR TCP socket option. What
     * this flag does depends on your operating system. Please consult operating
     * system documentation for more details.
     *
     * New values affect future calls to listen only.
     *
     * The default is false.
     *
     * @since 0.3.0
     *
     * @param value Whether or not to use the SO_REUSEADDR option
     */
    void set_reuse_addr(bool value) {
        m_reuse_addr = value;
    }

    /// Retrieve a reference to the endpoint's io_service
    /**
     * The io_service may be an internal or external one. This may be used to
     * call methods of the io_service that are not explicitly wrapped by the
     * endpoint.
     *
     * This method is only valid after the endpoint has been initialized with
     * `init_asio`. No error will be returned if it isn't.
     *
     * @return A reference to the endpoint's io_service
     */
    lib::asio::io_service & get_io_service() {
        return *m_io_service;
    }
    
    /// Get local TCP endpoint
    /**
     * Extracts the local endpoint from the acceptor. This represents the
     * address that WebSocket++ is listening on.
     *
     * Sets a bad_descriptor error if the acceptor is not currently listening
     * or otherwise unavailable.
     * 
     * @since 0.7.0
     *
     * @param ec Set to indicate what error occurred, if any.
     * @return The local endpoint
     */
    lib::asio::ip::tcp::endpoint get_local_endpoint(lib::asio::error_code & ec) {
        if (m_acceptor) {
            return m_acceptor->local_endpoint(ec);
        } else {
            ec = lib::asio::error::make_error_code(lib::asio::error::bad_descriptor);
            return lib::asio::ip::tcp::endpoint();
        }
    }

    /// Set up endpoint for listening manually (exception free)
    /**
     * Bind the internal acceptor using the specified settings. The endpoint
     * must have been initialized by calling init_asio before listening.
     *
     * @param ep An endpoint to read settings from
     * @param ec Set to indicate what error occurred, if any.
     */
    void listen(lib::asio::ip::tcp::endpoint const & ep, lib::error_code & ec)
    {
        if (m_state != READY) {
            m_elog->write(log::elevel::library,
                "asio::listen called from the wrong state");
            using websocketpp::error::make_error_code;
            ec = make_error_code(websocketpp::error::invalid_state);
            return;
        }

        m_alog->write(log::alevel::devel,"asio::listen");

        lib::asio::error_code bec;

        m_acceptor->open(ep.protocol(),bec);
        if (!bec) {
            m_acceptor->set_option(lib::asio::socket_base::reuse_address(m_reuse_addr),bec);
        }
        if (!bec) {
            m_acceptor->bind(ep,bec);
        }
        if (!bec) {
            m_acceptor->listen(m_listen_backlog,bec);
        }
        if (bec) {
            if (m_acceptor->is_open()) {
                m_acceptor->close();
            }
            log_err(log::elevel::info,"asio listen",bec);
            ec = make_error_code(error::pass_through);
        } else {
            m_state = LISTENING;
            ec = lib::error_code();
        }
    }

    /// Set up endpoint for listening manually
    /**
     * Bind the internal acceptor using the settings specified by the endpoint e
     *
     * @param ep An endpoint to read settings from
     */
    void listen(lib::asio::ip::tcp::endpoint const & ep) {
        lib::error_code ec;
        listen(ep,ec);
        if (ec) { throw exception(ec); }
    }

    /// Set up endpoint for listening with protocol and port (exception free)
    /**
     * Bind the internal acceptor using the given internet protocol and port.
     * The endpoint must have been initialized by calling init_asio before
     * listening.
     *
     * Common options include:
     * - IPv6 with mapped IPv4 for dual stack hosts lib::asio::ip::tcp::v6()
     * - IPv4 only: lib::asio::ip::tcp::v4()
     *
     * @param internet_protocol The internet protocol to use.
     * @param port The port to listen on.
     * @param ec Set to indicate what error occurred, if any.
     */
    template <typename InternetProtocol>
    void listen(InternetProtocol const & internet_protocol, uint16_t port,
        lib::error_code & ec)
    {
        lib::asio::ip::tcp::endpoint ep(internet_protocol, port);
        listen(ep,ec);
    }

    /// Set up endpoint for listening with protocol and port
    /**
     * Bind the internal acceptor using the given internet protocol and port.
     * The endpoint must have been initialized by calling init_asio before
     * listening.
     *
     * Common options include:
     * - IPv6 with mapped IPv4 for dual stack hosts lib::asio::ip::tcp::v6()
     * - IPv4 only: lib::asio::ip::tcp::v4()
     *
     * @param internet_protocol The internet protocol to use.
     * @param port The port to listen on.
     */
    template <typename InternetProtocol>
    void listen(InternetProtocol const & internet_protocol, uint16_t port)
    {
        lib::asio::ip::tcp::endpoint ep(internet_protocol, port);
        listen(ep);
    }

    /// Set up endpoint for listening on a port (exception free)
    /**
     * Bind the internal acceptor using the given port. The IPv6 protocol with
     * mapped IPv4 for dual stack hosts will be used. If you need IPv4 only use
     * the overload that allows specifying the protocol explicitly.
     *
     * The endpoint must have been initialized by calling init_asio before
     * listening.
     *
     * @param port The port to listen on.
     * @param ec Set to indicate what error occurred, if any.
     */
    void listen(uint16_t port, lib::error_code & ec) {
        listen(lib::asio::ip::tcp::v6(), port, ec);
    }

    /// Set up endpoint for listening on a port
    /**
     * Bind the internal acceptor using the given port. The IPv6 protocol with
     * mapped IPv4 for dual stack hosts will be used. If you need IPv4 only use
     * the overload that allows specifying the protocol explicitly.
     *
     * The endpoint must have been initialized by calling init_asio before
     * listening.
     *
     * @param port The port to listen on.
     * @param ec Set to indicate what error occurred, if any.
     */
    void listen(uint16_t port) {
        listen(lib::asio::ip::tcp::v6(), port);
    }

    /// Set up endpoint for listening on a host and service (exception free)
    /**
     * Bind the internal acceptor using the given host and service. More details
     * about what host and service can be are available in the Asio
     * documentation for ip::basic_resolver_query::basic_resolver_query's
     * constructors.
     *
     * The endpoint must have been initialized by calling init_asio before
     * listening.
     *
     * @param host A string identifying a location. May be a descriptive name or
     * a numeric address string.
     * @param service A string identifying the requested service. This may be a
     * descriptive name or a numeric string corresponding to a port number.
     * @param ec Set to indicate what error occurred, if any.
     */
    void listen(std::string const & host, std::string const & service,
        lib::error_code & ec)
    {
        using lib::asio::ip::tcp;
        tcp::resolver r(*m_io_service);
        tcp::resolver::query query(host, service);
        tcp::resolver::iterator endpoint_iterator = r.resolve(query);
        tcp::resolver::iterator end;
        if (endpoint_iterator == end) {
            m_elog->write(log::elevel::library,
                "asio::listen could not resolve the supplied host or service");
            ec = make_error_code(error::invalid_host_service);
            return;
        }
        listen(*endpoint_iterator,ec);
    }

    /// Set up endpoint for listening on a host and service
    /**
     * Bind the internal acceptor using the given host and service. More details
     * about what host and service can be are available in the Asio
     * documentation for ip::basic_resolver_query::basic_resolver_query's
     * constructors.
     *
     * The endpoint must have been initialized by calling init_asio before
     * listening.
     *
     * @param host A string identifying a location. May be a descriptive name or
     * a numeric address string.
     * @param service A string identifying the requested service. This may be a
     * descriptive name or a numeric string corresponding to a port number.
     * @param ec Set to indicate what error occurred, if any.
     */
    void listen(std::string const & host, std::string const & service)
    {
        lib::error_code ec;
        listen(host,service,ec);
        if (ec) { throw exception(ec); }
    }

    /// Stop listening (exception free)
    /**
     * Stop listening and accepting new connections. This will not end any
     * existing connections.
     *
     * @since 0.3.0-alpha4
     * @param ec A status code indicating an error, if any.
     */
    void stop_listening(lib::error_code & ec) {
        if (m_state != LISTENING) {
            m_elog->write(log::elevel::library,
                "asio::listen called from the wrong state");
            using websocketpp::error::make_error_code;
            ec = make_error_code(websocketpp::error::invalid_state);
            return;
        }

        m_acceptor->close();
        m_state = READY;
        ec = lib::error_code();
    }

    /// Stop listening
    /**
     * Stop listening and accepting new connections. This will not end any
     * existing connections.
     *
     * @since 0.3.0-alpha4
     */
    void stop_listening() {
        lib::error_code ec;
        stop_listening(ec);
        if (ec) { throw exception(ec); }
    }

    /// Check if the endpoint is listening
    /**
     * @return Whether or not the endpoint is listening.
     */
    bool is_listening() const {
        return (m_state == LISTENING);
    }

    /// wraps the run method of the internal io_service object
    std::size_t run() {
        return m_io_service->run();
    }

    /// wraps the run_one method of the internal io_service object
    /**
     * @since 0.3.0-alpha4
     */
    std::size_t run_one() {
        return m_io_service->run_one();
    }

    /// wraps the stop method of the internal io_service object
    void stop() {
        m_io_service->stop();
    }

    /// wraps the poll method of the internal io_service object
    std::size_t poll() {
        return m_io_service->poll();
    }

    /// wraps the poll_one method of the internal io_service object
    std::size_t poll_one() {
        return m_io_service->poll_one();
    }

    /// wraps the reset method of the internal io_service object
    void reset() {
        m_io_service->reset();
    }

    /// wraps the stopped method of the internal io_service object
    bool stopped() const {
        return m_io_service->stopped();
    }

    /// Marks the endpoint as perpetual, stopping it from exiting when empty
    /**
     * Marks the endpoint as perpetual. Perpetual endpoints will not
     * automatically exit when they run out of connections to process. To stop
     * a perpetual endpoint call `end_perpetual`.
     *
     * An endpoint may be marked perpetual at any time by any thread. It must be
     * called either before the endpoint has run out of work or before it was
     * started
     *
     * @since 0.3.0
     */
    void start_perpetual() {
        m_work = lib::make_shared<lib::asio::io_service::work>(
            lib::ref(*m_io_service)
        );
    }

    /// Clears the endpoint's perpetual flag, allowing it to exit when empty
    /**
     * Clears the endpoint's perpetual flag. This will cause the endpoint's run
     * method to exit normally when it runs out of connections. If there are
     * currently active connections it will not end until they are complete.
     *
     * @since 0.3.0
     */
    void stop_perpetual() {
        m_work.reset();
    }

    /// Call back a function after a period of time.
    /**
     * Sets a timer that calls back a function after the specified period of
     * milliseconds. Returns a handle that can be used to cancel the timer.
     * A cancelled timer will return the error code error::operation_aborted
     * A timer that expired will return no error.
     *
     * @param duration Length of time to wait in milliseconds
     * @param callback The function to call back when the timer has expired
     * @return A handle that can be used to cancel the timer if it is no longer
     * needed.
     */
    timer_ptr set_timer(long duration, timer_handler callback) {
        timer_ptr new_timer = lib::make_shared<lib::asio::steady_timer>(
            *m_io_service,
             lib::asio::milliseconds(duration)
        );

        new_timer->async_wait(
            lib::bind(
                &type::handle_timer,
                this,
                new_timer,
                callback,
                lib::placeholders::_1
            )
        );

        return new_timer;
    }

    /// Timer handler
    /**
     * The timer pointer is included to ensure the timer isn't destroyed until
     * after it has expired.
     *
     * @param t Pointer to the timer in question
     * @param callback The function to call back
     * @param ec A status code indicating an error, if any.
     */
    void handle_timer(timer_ptr, timer_handler callback,
        lib::asio::error_code const & ec)
    {
        if (ec) {
            if (ec == lib::asio::error::operation_aborted) {
                callback(make_error_code(transport::error::operation_aborted));
            } else {
                m_elog->write(log::elevel::info,
                    "asio handle_timer error: "+ec.message());
                log_err(log::elevel::info,"asio handle_timer",ec);
                callback(make_error_code(error::pass_through));
            }
        } else {
            callback(lib::error_code());
        }
    }

    /// Accept the next connection attempt and assign it to con (exception free)
    /**
     * @param tcon The connection to accept into.
     * @param callback The function to call when the operation is complete.
     * @param ec A status code indicating an error, if any.
     */
    void async_accept(transport_con_ptr tcon, accept_handler callback,
        lib::error_code & ec)
    {
        if (m_state != LISTENING) {
            using websocketpp::error::make_error_code;
            ec = make_error_code(websocketpp::error::async_accept_not_listening);
            return;
        }

        m_alog->write(log::alevel::devel, "asio::async_accept");

        if (config::enable_multithreading) {
            m_acceptor->async_accept(
                tcon->get_raw_socket(),
                tcon->get_strand()->wrap(lib::bind(
                    &type::handle_accept,
                    this,
                    callback,
                    lib::placeholders::_1
                ))
            );
        } else {
            m_acceptor->async_accept(
                tcon->get_raw_socket(),
                lib::bind(
                    &type::handle_accept,
                    this,
                    callback,
                    lib::placeholders::_1
                )
            );
        }
    }

    /// Accept the next connection attempt and assign it to con.
    /**
     * @param tcon The connection to accept into.
     * @param callback The function to call when the operation is complete.
     */
    void async_accept(transport_con_ptr tcon, accept_handler callback) {
        lib::error_code ec;
        async_accept(tcon,callback,ec);
        if (ec) { throw exception(ec); }
    }
protected:
    /// Initialize logging
    /**
     * The loggers are located in the main endpoint class. As such, the
     * transport doesn't have direct access to them. This method is called
     * by the endpoint constructor to allow shared logging from the transport
     * component. These are raw pointers to member variables of the endpoint.
     * In particular, they cannot be used in the transport constructor as they
     * haven't been constructed yet, and cannot be used in the transport
     * destructor as they will have been destroyed by then.
     */
    void init_logging(alog_type* a, elog_type* e) {
        m_alog = a;
        m_elog = e;
    }

    void handle_accept(accept_handler callback, lib::asio::error_code const & 
        asio_ec)
    {
        lib::error_code ret_ec;

        m_alog->write(log::alevel::devel, "asio::handle_accept");

        if (asio_ec) {
            if (asio_ec == lib::asio::errc::operation_canceled) {
                ret_ec = make_error_code(websocketpp::error::operation_canceled);
            } else {
                log_err(log::elevel::info,"asio handle_accept",asio_ec);
                ret_ec = make_error_code(error::pass_through);
            }
        }

        callback(ret_ec);
    }

    /// Initiate a new connection
    // TODO: there have to be some more failure conditions here
    void async_connect(transport_con_ptr tcon, uri_ptr u, connect_handler cb) {
        using namespace lib::asio::ip;

        // Create a resolver
        if (!m_resolver) {
            m_resolver = lib::make_shared<lib::asio::ip::tcp::resolver>(
                lib::ref(*m_io_service));
        }

        tcon->set_uri(u);

        std::string proxy = tcon->get_proxy();
        std::string host;
        std::string port;

        if (proxy.empty()) {
            host = u->get_host();
            port = u->get_port_str();
        } else {
            lib::error_code ec;

            uri_ptr pu = lib::make_shared<uri>(proxy);

            if (!pu->get_valid()) {
                cb(make_error_code(error::proxy_invalid));
                return;
            }

            ec = tcon->proxy_init(u->get_authority());
            if (ec) {
                cb(ec);
                return;
            }

            host = pu->get_host();
            port = pu->get_port_str();
        }

        tcp::resolver::query query(host,port);

        if (m_alog->static_test(log::alevel::devel)) {
            m_alog->write(log::alevel::devel,
                "starting async DNS resolve for "+host+":"+port);
        }

        timer_ptr dns_timer;

        dns_timer = tcon->set_timer(
            config::timeout_dns_resolve,
            lib::bind(
                &type::handle_resolve_timeout,
                this,
                dns_timer,
                cb,
                lib::placeholders::_1
            )
        );

        if (config::enable_multithreading) {
            m_resolver->async_resolve(
                query,
                tcon->get_strand()->wrap(lib::bind(
                    &type::handle_resolve,
                    this,
                    tcon,
                    dns_timer,
                    cb,
                    lib::placeholders::_1,
                    lib::placeholders::_2
                ))
            );
        } else {
            m_resolver->async_resolve(
                query,
                lib::bind(
                    &type::handle_resolve,
                    this,
                    tcon,
                    dns_timer,
                    cb,
                    lib::placeholders::_1,
                    lib::placeholders::_2
                )
            );
        }
    }

    /// DNS resolution timeout handler
    /**
     * The timer pointer is included to ensure the timer isn't destroyed until
     * after it has expired.
     *
     * @param dns_timer Pointer to the timer in question
     * @param callback The function to call back
     * @param ec A status code indicating an error, if any.
     */
    void handle_resolve_timeout(timer_ptr, connect_handler callback,
        lib::error_code const & ec)
    {
        lib::error_code ret_ec;

        if (ec) {
            if (ec == transport::error::operation_aborted) {
                m_alog->write(log::alevel::devel,
                    "asio handle_resolve_timeout timer cancelled");
                return;
            }

            log_err(log::elevel::devel,"asio handle_resolve_timeout",ec);
            ret_ec = ec;
        } else {
            ret_ec = make_error_code(transport::error::timeout);
        }

        m_alog->write(log::alevel::devel,"DNS resolution timed out");
        m_resolver->cancel();
        callback(ret_ec);
    }

    void handle_resolve(transport_con_ptr tcon, timer_ptr dns_timer,
        connect_handler callback, lib::asio::error_code const & ec,
        lib::asio::ip::tcp::resolver::iterator iterator)
    {
        if (ec == lib::asio::error::operation_aborted ||
            lib::asio::is_neg(dns_timer->expires_from_now()))
        {
            m_alog->write(log::alevel::devel,"async_resolve cancelled");
            return;
        }

        dns_timer->cancel();

        if (ec) {
            log_err(log::elevel::info,"asio async_resolve",ec);
            callback(make_error_code(error::pass_through));
            return;
        }

        if (m_alog->static_test(log::alevel::devel)) {
            std::stringstream s;
            s << "Async DNS resolve successful. Results: ";

            lib::asio::ip::tcp::resolver::iterator it, end;
            for (it = iterator; it != end; ++it) {
                s << (*it).endpoint() << " ";
            }

            m_alog->write(log::alevel::devel,s.str());
        }

        m_alog->write(log::alevel::devel,"Starting async connect");

        timer_ptr con_timer;

        con_timer = tcon->set_timer(
            config::timeout_connect,
            lib::bind(
                &type::handle_connect_timeout,
                this,
                tcon,
                con_timer,
                callback,
                lib::placeholders::_1
            )
        );

        if (config::enable_multithreading) {
            lib::asio::async_connect(
                tcon->get_raw_socket(),
                iterator,
                tcon->get_strand()->wrap(lib::bind(
                    &type::handle_connect,
                    this,
                    tcon,
                    con_timer,
                    callback,
                    lib::placeholders::_1
                ))
            );
        } else {
            lib::asio::async_connect(
                tcon->get_raw_socket(),
                iterator,
                lib::bind(
                    &type::handle_connect,
                    this,
                    tcon,
                    con_timer,
                    callback,
                    lib::placeholders::_1
                )
            );
        }
    }

    /// Asio connect timeout handler
    /**
     * The timer pointer is included to ensure the timer isn't destroyed until
     * after it has expired.
     *
     * @param tcon Pointer to the transport connection that is being connected
     * @param con_timer Pointer to the timer in question
     * @param callback The function to call back
     * @param ec A status code indicating an error, if any.
     */
    void handle_connect_timeout(transport_con_ptr tcon, timer_ptr,
        connect_handler callback, lib::error_code const & ec)
    {
        lib::error_code ret_ec;

        if (ec) {
            if (ec == transport::error::operation_aborted) {
                m_alog->write(log::alevel::devel,
                    "asio handle_connect_timeout timer cancelled");
                return;
            }

            log_err(log::elevel::devel,"asio handle_connect_timeout",ec);
            ret_ec = ec;
        } else {
            ret_ec = make_error_code(transport::error::timeout);
        }

        m_alog->write(log::alevel::devel,"TCP connect timed out");
        tcon->cancel_socket_checked();
        callback(ret_ec);
    }

    void handle_connect(transport_con_ptr tcon, timer_ptr con_timer,
        connect_handler callback, lib::asio::error_code const & ec)
    {
        if (ec == lib::asio::error::operation_aborted ||
            lib::asio::is_neg(con_timer->expires_from_now()))
        {
            m_alog->write(log::alevel::devel,"async_connect cancelled");
            return;
        }

        con_timer->cancel();

        if (ec) {
            log_err(log::elevel::info,"asio async_connect",ec);
            callback(make_error_code(error::pass_through));
            return;
        }

        if (m_alog->static_test(log::alevel::devel)) {
            m_alog->write(log::alevel::devel,
                "Async connect to "+tcon->get_remote_endpoint()+" successful.");
        }

        callback(lib::error_code());
    }

    /// Initialize a connection
    /**
     * init is called by an endpoint once for each newly created connection.
     * It's purpose is to give the transport policy the chance to perform any
     * transport specific initialization that couldn't be done via the default
     * constructor.
     *
     * @param tcon A pointer to the transport portion of the connection.
     *
     * @return A status code indicating the success or failure of the operation
     */
    lib::error_code init(transport_con_ptr tcon) {
        m_alog->write(log::alevel::devel, "transport::asio::init");

        // Initialize the connection socket component
        socket_type::init(lib::static_pointer_cast<socket_con_type,
            transport_con_type>(tcon));

        lib::error_code ec;

        ec = tcon->init_asio(m_io_service);
        if (ec) {return ec;}

        tcon->set_tcp_pre_init_handler(m_tcp_pre_init_handler);
        tcon->set_tcp_post_init_handler(m_tcp_post_init_handler);

        return lib::error_code();
    }
private:
    /// Convenience method for logging the code and message for an error_code
    template <typename error_type>
    void log_err(log::level l, char const * msg, error_type const & ec) {
        std::stringstream s;
        s << msg << " error: " << ec << " (" << ec.message() << ")";
        m_elog->write(l,s.str());
    }

    enum state {
        UNINITIALIZED = 0,
        READY = 1,
        LISTENING = 2
    };

    // Handlers
    tcp_init_handler    m_tcp_pre_init_handler;
    tcp_init_handler    m_tcp_post_init_handler;

    // Network Resources
    io_service_ptr      m_io_service;
    bool                m_external_io_service;
    acceptor_ptr        m_acceptor;
    resolver_ptr        m_resolver;
    work_ptr            m_work;

    // Network constants
    int                 m_listen_backlog;
    bool                m_reuse_addr;

    elog_type* m_elog;
    alog_type* m_alog;

    // Transport state
    state               m_state;
};

} // namespace asio
} // namespace transport
} // namespace websocketpp

#endif // WEBSOCKETPP_TRANSPORT_ASIO_HPP
