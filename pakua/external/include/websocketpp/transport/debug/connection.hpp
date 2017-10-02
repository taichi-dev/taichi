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

#ifndef WEBSOCKETPP_TRANSPORT_DEBUG_CON_HPP
#define WEBSOCKETPP_TRANSPORT_DEBUG_CON_HPP

#include <websocketpp/transport/debug/base.hpp>

#include <websocketpp/transport/base/connection.hpp>

#include <websocketpp/uri.hpp>
#include <websocketpp/logger/levels.hpp>

#include <websocketpp/common/connection_hdl.hpp>
#include <websocketpp/common/memory.hpp>
#include <websocketpp/common/platforms.hpp>

#include <string>
#include <vector>

namespace websocketpp {
namespace transport {
namespace debug {

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
      : m_reading(false), m_is_server(is_server), m_alog(alog), m_elog(elog)
    {
        m_alog.write(log::alevel::devel,"debug con transport constructor");
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
    void set_secure(bool) {}

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
    void set_remote_endpoint(std::string) {}

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
        return "unknown (debug transport)";
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
    timer_ptr set_timer(long, timer_handler handler) {
        m_alog.write(log::alevel::devel,"debug connection set timer");
        m_timer_handler = handler;
        return timer_ptr();
    }
    
    /// Manual input supply (read all)
    /**
     * Similar to read_some, but continues to read until all bytes in the
     * supplied buffer have been read or the connection runs out of read
     * requests.
     *
     * This method still may not read all of the bytes in the input buffer. if
     * it doesn't it indicates that the connection was most likely closed or
     * is in an error state where it is no longer accepting new input.
     *
     * @since 0.3.0
     *
     * @param buf Char buffer to read into the websocket
     * @param len Length of buf
     * @return The number of characters from buf actually read.
     */
    size_t read_all(char const * buf, size_t len) {        
        size_t total_read = 0;
        size_t temp_read = 0;

        do {
            temp_read = this->read_some_impl(buf+total_read,len-total_read);
            total_read += temp_read;
        } while (temp_read != 0 && total_read < len);

        return total_read;
    }
    
    // debug stuff to invoke the async handlers
    void expire_timer(lib::error_code const & ec) {
        m_timer_handler(ec);
    }
    
    void fullfil_write() {
        m_write_handler(lib::error_code());
    }
protected:
    /// Initialize the connection transport
    /**
     * Initialize the connection's transport component.
     *
     * @param handler The `init_handler` to call when initialization is done
     */
    void init(init_handler handler) {
        m_alog.write(log::alevel::devel,"debug connection init");
        handler(lib::error_code());
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
        std::stringstream s;
        s << "debug_con async_read_at_least: " << num_bytes;
        m_alog.write(log::alevel::devel,s.str());

        if (num_bytes > len) {
            handler(make_error_code(error::invalid_num_bytes),size_t(0));
            return;
        }

        if (m_reading == true) {
            handler(make_error_code(error::double_read),size_t(0));
            return;
        }

        if (num_bytes == 0 || len == 0) {
            handler(lib::error_code(),size_t(0));
            return;
        }

        m_buf = buf;
        m_len = len;
        m_bytes_needed = num_bytes;
        m_read_handler = handler;
        m_cursor = 0;
        m_reading = true;
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
    void async_write(char const *, size_t, write_handler handler) {
        m_alog.write(log::alevel::devel,"debug_con async_write");
        m_write_handler = handler;
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
    void async_write(std::vector<buffer> const &, write_handler handler) {
        m_alog.write(log::alevel::devel,"debug_con async_write buffer list");
        m_write_handler = handler;
    }

    /// Set Connection Handle
    /**
     * @param hdl The new handle
     */
    void set_handle(connection_hdl) {}

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
    
    size_t read_some_impl(char const * buf, size_t len) {
        m_alog.write(log::alevel::devel,"debug_con read_some");

        if (!m_reading) {
            m_elog.write(log::elevel::devel,"write while not reading");
            return 0;
        }

        size_t bytes_to_copy = (std::min)(len,m_len-m_cursor);

        std::copy(buf,buf+bytes_to_copy,m_buf+m_cursor);

        m_cursor += bytes_to_copy;

        if (m_cursor >= m_bytes_needed) {
            complete_read(lib::error_code());
        }

        return bytes_to_copy;
    }

    /// Signal that a requested read is complete
    /**
     * Sets the reading flag to false and returns the handler that should be
     * called back with the result of the read. The cursor position that is sent
     * is whatever the value of m_cursor is.
     *
     * It MUST NOT be called when m_reading is false.
     * it MUST be called while holding the read lock
     *
     * It is important to use this method rather than directly setting/calling
     * m_read_handler back because this function makes sure to delete the
     * locally stored handler which contains shared pointers that will otherwise
     * cause circular reference based memory leaks.
     *
     * @param ec The error code to forward to the read handler
     */
    void complete_read(lib::error_code const & ec) {
        m_reading = false;

        read_handler handler = m_read_handler;
        m_read_handler = read_handler();

        handler(ec,m_cursor);
    }
private:
    timer_handler m_timer_handler;
    
    // Read space (Protected by m_read_mutex)
    char *          m_buf;
    size_t          m_len;
    size_t          m_bytes_needed;
    read_handler    m_read_handler;
    size_t          m_cursor;

    // transport resources
    connection_hdl  m_connection_hdl;
    write_handler   m_write_handler;
    shutdown_handler    m_shutdown_handler;

    bool            m_reading;
    bool const      m_is_server;
    bool            m_is_secure;
    alog_type &     m_alog;
    elog_type &     m_elog;
    std::string     m_remote_endpoint;
};


} // namespace debug
} // namespace transport
} // namespace websocketpp

#endif // WEBSOCKETPP_TRANSPORT_DEBUG_CON_HPP
