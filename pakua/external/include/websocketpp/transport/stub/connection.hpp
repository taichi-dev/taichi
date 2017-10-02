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

#ifndef WEBSOCKETPP_TRANSPORT_STUB_CON_HPP
#define WEBSOCKETPP_TRANSPORT_STUB_CON_HPP

#include <websocketpp/transport/stub/base.hpp>

#include <websocketpp/transport/base/connection.hpp>

#include <websocketpp/logger/levels.hpp>

#include <websocketpp/common/connection_hdl.hpp>
#include <websocketpp/common/memory.hpp>
#include <websocketpp/common/platforms.hpp>

#include <string>
#include <vector>

namespace websocketpp {
namespace transport {
namespace stub {

/// Empty timer class to stub out for timer functionality that stub
/// transport doesn't support
struct timer {
    void cancel() {}
};

template <typename config>
class connection : public lib::enable_shared_from_this< connection<config> > {
public:
    /// Type of this connection transport component
    typedef connection<config> type;
    /// Type of a shared pointer to this connection transport component
    typedef lib::shared_ptr<type> ptr;

    /// transport concurrency policy
    typedef typename config::concurrency_type concurrency_type;
    /// Type of this transport's access logging policy
    typedef typename config::alog_type alog_type;
    /// Type of this transport's error logging policy
    typedef typename config::elog_type elog_type;

    // Concurrency policy types
    typedef typename concurrency_type::scoped_lock_type scoped_lock_type;
    typedef typename concurrency_type::mutex_type mutex_type;

    typedef lib::shared_ptr<timer> timer_ptr;

    explicit connection(bool is_server, alog_type & alog, elog_type & elog)
      : m_alog(alog), m_elog(elog)
    {
        m_alog.write(log::alevel::devel,"stub con transport constructor");
    }

    /// Get a shared pointer to this component
    ptr get_shared() {
        return type::shared_from_this();
    }

    /// Set whether or not this connection is secure
    /**
     * Todo: docs
     *
     * @since 0.3.0-alpha4
     *
     * @param value Whether or not this connection is secure.
     */
    void set_secure(bool value) {}

    /// Tests whether or not the underlying transport is secure
    /**
     * TODO: docs
     *
     * @return Whether or not the underlying transport is secure
     */
    bool is_secure() const {
        return false;
    }

    /// Set uri hook
    /**
     * Called by the endpoint as a connection is being established to provide
     * the uri being connected to to the transport layer.
     *
     * Implementation is optional and can be ignored if the transport has no
     * need for this information.
     *
     * @since 0.6.0
     *
     * @param u The uri to set
     */
    void set_uri(uri_ptr) {}

    /// Set human readable remote endpoint address
    /**
     * Sets the remote endpoint address returned by `get_remote_endpoint`. This
     * value should be a human readable string that describes the remote
     * endpoint. Typically an IP address or hostname, perhaps with a port. But
     * may be something else depending on the nature of the underlying
     * transport.
     *
     * If none is set a default is returned.
     *
     * @since 0.3.0-alpha4
     *
     * @param value The remote endpoint address to set.
     */
    void set_remote_endpoint(std::string value) {}

    /// Get human readable remote endpoint address
    /**
     * TODO: docs
     *
     * This value is used in access and error logs and is available to the end
     * application for including in user facing interfaces and messages.
     *
     * @return A string identifying the address of the remote endpoint
     */
    std::string get_remote_endpoint() const {
        return "unknown (stub transport)";
    }

    /// Get the connection handle
    /**
     * @return The handle for this connection.
     */
    connection_hdl get_handle() const {
        return connection_hdl();
    }

    /// Call back a function after a period of time.
    /**
     * Timers are not implemented in this transport. The timer pointer will
     * always be empty. The handler will never be called.
     *
     * @param duration Length of time to wait in milliseconds
     * @param callback The function to call back when the timer has expired
     * @return A handle that can be used to cancel the timer if it is no longer
     * needed.
     */
    timer_ptr set_timer(long duration, timer_handler handler) {
        return timer_ptr();
    }
protected:
    /// Initialize the connection transport
    /**
     * Initialize the connection's transport component.
     *
     * @param handler The `init_handler` to call when initialization is done
     */
    void init(init_handler handler) {
        m_alog.write(log::alevel::devel,"stub connection init");
        handler(make_error_code(error::not_implemented));
    }

    /// Initiate an async_read for at least num_bytes bytes into buf
    /**
     * Initiates an async_read request for at least num_bytes bytes. The input
     * will be read into buf. A maximum of len bytes will be input. When the
     * operation is complete, handler will be called with the status and number
     * of bytes read.
     *
     * This method may or may not call handler from within the initial call. The
     * application should be prepared to accept either.
     *
     * The application should never call this method a second time before it has
     * been called back for the first read. If this is done, the second read
     * will be called back immediately with a double_read error.
     *
     * If num_bytes or len are zero handler will be called back immediately
     * indicating success.
     *
     * @param num_bytes Don't call handler until at least this many bytes have
     * been read.
     * @param buf The buffer to read bytes into
     * @param len The size of buf. At maximum, this many bytes will be read.
     * @param handler The callback to invoke when the operation is complete or
     * ends in an error
     */
    void async_read_at_least(size_t num_bytes, char * buf, size_t len,
        read_handler handler)
    {
        m_alog.write(log::alevel::devel, "stub_con async_read_at_least");
        handler(make_error_code(error::not_implemented), 0);
    }

    /// Asyncronous Transport Write
    /**
     * Write len bytes in buf to the output stream. Call handler to report
     * success or failure. handler may or may not be called during async_write,
     * but it must be safe for this to happen.
     *
     * Will return 0 on success.
     *
     * @param buf buffer to read bytes from
     * @param len number of bytes to write
     * @param handler Callback to invoke with operation status.
     */
    void async_write(char const * buf, size_t len, write_handler handler) {
        m_alog.write(log::alevel::devel,"stub_con async_write");
        handler(make_error_code(error::not_implemented));
    }

    /// Asyncronous Transport Write (scatter-gather)
    /**
     * Write a sequence of buffers to the output stream. Call handler to report
     * success or failure. handler may or may not be called during async_write,
     * but it must be safe for this to happen.
     *
     * Will return 0 on success.
     *
     * @param bufs vector of buffers to write
     * @param handler Callback to invoke with operation status.
     */
    void async_write(std::vector<buffer> const & bufs, write_handler handler) {
        m_alog.write(log::alevel::devel,"stub_con async_write buffer list");
        handler(make_error_code(error::not_implemented));
    }

    /// Set Connection Handle
    /**
     * @param hdl The new handle
     */
    void set_handle(connection_hdl hdl) {}

    /// Call given handler back within the transport's event system (if present)
    /**
     * Invoke a callback within the transport's event system if it has one. If
     * it doesn't, the handler will be invoked immediately before this function
     * returns.
     *
     * @param handler The callback to invoke
     *
     * @return Whether or not the transport was able to register the handler for
     * callback.
     */
    lib::error_code dispatch(dispatch_handler handler) {
        handler();
        return lib::error_code();
    }

    /// Perform cleanup on socket shutdown_handler
    /**
     * @param h The `shutdown_handler` to call back when complete
     */
    void async_shutdown(shutdown_handler handler) {
        handler(lib::error_code());
    }
private:
    // member variables!
    alog_type & m_alog;
    elog_type & m_elog;
};


} // namespace stub
} // namespace transport
} // namespace websocketpp

#endif // WEBSOCKETPP_TRANSPORT_STUB_CON_HPP
