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

#ifndef WEBSOCKETPP_TRANSPORT_IOSTREAM_CON_HPP
#define WEBSOCKETPP_TRANSPORT_IOSTREAM_CON_HPP

#include <websocketpp/transport/iostream/base.hpp>

#include <websocketpp/transport/base/connection.hpp>

#include <websocketpp/uri.hpp>

#include <websocketpp/logger/levels.hpp>

#include <websocketpp/common/connection_hdl.hpp>
#include <websocketpp/common/memory.hpp>
#include <websocketpp/common/platforms.hpp>

#include <algorithm>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace websocketpp {
namespace transport {
namespace iostream {

/// Empty timer class to stub out for timer functionality that iostream
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
      : m_output_stream(NULL)
      , m_reading(false)
      , m_is_server(is_server)
      , m_is_secure(false)
      , m_alog(alog)
      , m_elog(elog)
      , m_remote_endpoint("iostream transport")
    {
        m_alog.write(log::alevel::devel,"iostream con transport constructor");
    }

    /// Get a shared pointer to this component
    ptr get_shared() {
        return type::shared_from_this();
    }

    /// Register a std::ostream with the transport for writing output
    /**
     * Register a std::ostream with the transport. All future writes will be
     * done to this output stream.
     *
     * @param o A pointer to the ostream to use for output.
     */
    void register_ostream(std::ostream * o) {
        // TODO: lock transport state?
        scoped_lock_type lock(m_read_mutex);
        m_output_stream = o;
    }

    /// Set uri hook
    /**
     * Called by the endpoint as a connection is being established to provide
     * the uri being connected to to the transport layer.
     *
     * This transport policy doesn't use the uri so it is ignored.
     *
     * @since 0.6.0
     *
     * @param u The uri to set
     */
    void set_uri(uri_ptr) {}

    /// Overloaded stream input operator
    /**
     * Attempts to read input from the given stream into the transport. Bytes
     * will be extracted from the input stream to fulfill any pending reads.
     * Input in this manner will only read until the current read buffer has
     * been filled. Then it will signal the library to process the input. If the
     * library's input handler adds a new async_read, additional bytes will be
     * read, otherwise the input operation will end.
     *
     * When this function returns one of the following conditions is true:
     * - There is no outstanding read operation
     * - There are no more bytes available in the input stream
     *
     * You can use tellg() on the input stream to determine if all of the input
     * bytes were read or not.
     *
     * If there is no pending read operation when the input method is called, it
     * will return immediately and tellg() will not have changed.
     */
    friend std::istream & operator>> (std::istream & in, type & t) {
        // this serializes calls to external read.
        scoped_lock_type lock(t.m_read_mutex);

        t.read(in);

        return in;
    }

    /// Manual input supply (read some)
    /**
     * Copies bytes from buf into WebSocket++'s input buffers. Bytes will be
     * copied from the supplied buffer to fulfill any pending library reads. It
     * will return the number of bytes successfully processed. If there are no
     * pending reads read_some will return immediately. Not all of the bytes may
     * be able to be read in one call.
     *
     * @since 0.3.0-alpha4
     *
     * @param buf Char buffer to read into the websocket
     * @param len Length of buf
     * @return The number of characters from buf actually read.
     */
    size_t read_some(char const * buf, size_t len) {
        // this serializes calls to external read.
        scoped_lock_type lock(m_read_mutex);

        return this->read_some_impl(buf,len);
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
        // this serializes calls to external read.
        scoped_lock_type lock(m_read_mutex);

        size_t total_read = 0;
        size_t temp_read = 0;

        do {
            temp_read = this->read_some_impl(buf+total_read,len-total_read);
            total_read += temp_read;
        } while (temp_read != 0 && total_read < len);

        return total_read;
    }

    /// Manual input supply (DEPRECATED)
    /**
     * @deprecated DEPRECATED in favor of read_some()
     * @see read_some()
     */
    size_t readsome(char const * buf, size_t len) {
        return this->read_some(buf,len);
    }

    /// Signal EOF
    /**
     * Signals to the transport that data stream being read has reached EOF and
     * that no more bytes may be read or written to/from the transport.
     *
     * @since 0.3.0-alpha4
     */
    void eof() {
        // this serializes calls to external read.
        scoped_lock_type lock(m_read_mutex);

        if (m_reading) {
            complete_read(make_error_code(transport::error::eof));
        }
    }

    /// Signal transport error
    /**
     * Signals to the transport that a fatal data stream error has occurred and
     * that no more bytes may be read or written to/from the transport.
     *
     * @since 0.3.0-alpha4
     */
    void fatal_error() {
        // this serializes calls to external read.
        scoped_lock_type lock(m_read_mutex);

        if (m_reading) {
            complete_read(make_error_code(transport::error::pass_through));
        }
    }

    /// Set whether or not this connection is secure
    /**
     * The iostream transport does not provide any security features. As such
     * it defaults to returning false when `is_secure` is called. However, the
     * iostream transport may be used to wrap an external socket API that may
     * provide secure transport. This method allows that external API to flag
     * whether or not this connection is secure so that users of the WebSocket++
     * API will get more accurate information.
     *
     * @since 0.3.0-alpha4
     *
     * @param value Whether or not this connection is secure.
     */
    void set_secure(bool value) {
        m_is_secure = value;
    }

    /// Tests whether or not the underlying transport is secure
    /**
     * iostream transport will return false always because it has no information
     * about the ultimate remote endpoint. This may or may not be accurate
     * depending on the real source of bytes being input. The `set_secure`
     * method may be used to flag connections that are secured by an external
     * API
     *
     * @return Whether or not the underlying transport is secure
     */
    bool is_secure() const {
        return m_is_secure;
    }

    /// Set human readable remote endpoint address
    /**
     * Sets the remote endpoint address returned by `get_remote_endpoint`. This
     * value should be a human readable string that describes the remote
     * endpoint. Typically an IP address or hostname, perhaps with a port. But
     * may be something else depending on the nature of the underlying
     * transport.
     *
     * If none is set the default is "iostream transport".
     *
     * @since 0.3.0-alpha4
     *
     * @param value The remote endpoint address to set.
     */
    void set_remote_endpoint(std::string value) {
        m_remote_endpoint = value;
    }

    /// Get human readable remote endpoint address
    /**
     * The iostream transport has no information about the ultimate remote
     * endpoint. It will return the string "iostream transport". The
     * `set_remote_endpoint` method may be used by external network code to set
     * a more accurate value.
     *
     * This value is used in access and error logs and is available to the end
     * application for including in user facing interfaces and messages.
     *
     * @return A string identifying the address of the remote endpoint
     */
    std::string get_remote_endpoint() const {
        return m_remote_endpoint;
    }

    /// Get the connection handle
    /**
     * @return The handle for this connection.
     */
    connection_hdl get_handle() const {
        return m_connection_hdl;
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
    timer_ptr set_timer(long, timer_handler) {
        return timer_ptr();
    }

    /// Sets the write handler
    /**
     * The write handler is called when the iostream transport receives data
     * that needs to be written to the appropriate output location. This handler
     * can be used in place of registering an ostream for output.
     *
     * The signature of the handler is
     * `lib::error_code (connection_hdl, char const *, size_t)` The
     * code returned will be reported and logged by the core library.
     *
     * See also, set_vector_write_handler, for an optional write handler that
     * allows more efficient handling of multiple writes at once.
     *
     * @see set_vector_write_handler
     *
     * @since 0.5.0
     *
     * @param h The handler to call when data is to be written.
     */
    void set_write_handler(write_handler h) {
        m_write_handler = h;
    }

    /// Sets the vectored write handler
    /**
     * The vectored write handler is called when the iostream transport receives
     * multiple chunks of data that need to be written to the appropriate output
     * location. This handler can be used in conjunction with the write_handler
     * in place of registering an ostream for output.
     *
     * The sequence of buffers represents bytes that should be written
     * consecutively and it is suggested to group the buffers into as few next
     * layer packets as possible. Vector write is used to allow implementations
     * that support it to coalesce writes into a single TCP packet or TLS
     * segment for improved efficiency.
     *
     * This is an optional handler. If it is not defined then multiple calls
     * will be made to the standard write handler.
     *
     * The signature of the handler is
     * `lib::error_code (connection_hdl, std::vector<websocketpp::transport::buffer>
     * const & bufs)`. The code returned will be reported and logged by the core
     * library. The `websocketpp::transport::buffer` type is a struct with two
     * data members. buf (char const *) and len (size_t).
     *
     * @since 0.6.0
     *
     * @param h The handler to call when vectored data is to be written.
     */
    void set_vector_write_handler(vector_write_handler h) {
        m_vector_write_handler = h;
    }

    /// Sets the shutdown handler
    /**
     * The shutdown handler is called when the iostream transport receives a
     * notification from the core library that it is finished with all read and
     * write operations and that the underlying transport can be cleaned up.
     *
     * If you are using iostream transport with another socket library, this is
     * a good time to close/shutdown the socket for this connection.
     *
     * The signature of the handler is `lib::error_code (connection_hdl)`. The
     * code returned will be reported and logged by the core library.
     *
     * @since 0.5.0
     *
     * @param h The handler to call on connection shutdown.
     */
    void set_shutdown_handler(shutdown_handler h) {
        m_shutdown_handler = h;
    }
protected:
    /// Initialize the connection transport
    /**
     * Initialize the connection's transport component.
     *
     * @param handler The `init_handler` to call when initialization is done
     */
    void init(init_handler handler) {
        m_alog.write(log::alevel::devel,"iostream connection init");
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
    void async_read_at_least(size_t num_bytes, char *buf, size_t len,
        read_handler handler)
    {
        std::stringstream s;
        s << "iostream_con async_read_at_least: " << num_bytes;
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
     * Write len bytes in buf to the output method. Call handler to report
     * success or failure. handler may or may not be called during async_write,
     * but it must be safe for this to happen.
     *
     * Will return 0 on success. Other possible errors (not exhaustive)
     * output_stream_required: No output stream was registered to write to
     * bad_stream: a ostream pass through error
     *
     * This method will attempt to write to the registered ostream first. If an
     * ostream is not registered it will use the write handler. If neither are
     * registered then an error is passed up to the connection.
     *
     * @param buf buffer to read bytes from
     * @param len number of bytes to write
     * @param handler Callback to invoke with operation status.
     */
    void async_write(char const * buf, size_t len, transport::write_handler
        handler)
    {
        m_alog.write(log::alevel::devel,"iostream_con async_write");
        // TODO: lock transport state?

        lib::error_code ec;

        if (m_output_stream) {
            m_output_stream->write(buf,len);

            if (m_output_stream->bad()) {
                ec = make_error_code(error::bad_stream);
            }
        } else if (m_write_handler) {
            ec = m_write_handler(m_connection_hdl, buf, len);
        } else {
            ec = make_error_code(error::output_stream_required);
        }

        handler(ec);
    }

    /// Asyncronous Transport Write (scatter-gather)
    /**
     * Write a sequence of buffers to the output method. Call handler to report
     * success or failure. handler may or may not be called during async_write,
     * but it must be safe for this to happen.
     *
     * Will return 0 on success. Other possible errors (not exhaustive)
     * output_stream_required: No output stream was registered to write to
     * bad_stream: a ostream pass through error
     *
     * This method will attempt to write to the registered ostream first. If an
     * ostream is not registered it will use the write handler. If neither are
     * registered then an error is passed up to the connection.
     *
     * @param bufs vector of buffers to write
     * @param handler Callback to invoke with operation status.
     */
    void async_write(std::vector<buffer> const & bufs, transport::write_handler
        handler)
    {
        m_alog.write(log::alevel::devel,"iostream_con async_write buffer list");
        // TODO: lock transport state?

        lib::error_code ec;

        if (m_output_stream) {
            std::vector<buffer>::const_iterator it;
            for (it = bufs.begin(); it != bufs.end(); it++) {
                m_output_stream->write((*it).buf,(*it).len);

                if (m_output_stream->bad()) {
                    ec = make_error_code(error::bad_stream);
                    break;
                }
            }
        } else if (m_vector_write_handler) {
            ec = m_vector_write_handler(m_connection_hdl, bufs);
        } else if (m_write_handler) {
            std::vector<buffer>::const_iterator it;
            for (it = bufs.begin(); it != bufs.end(); it++) {
                ec = m_write_handler(m_connection_hdl, (*it).buf, (*it).len);
                if (ec) {break;}
            }

        } else {
            ec = make_error_code(error::output_stream_required);
        }

        handler(ec);
    }

    /// Set Connection Handle
    /**
     * @param hdl The new handle
     */
    void set_handle(connection_hdl hdl) {
        m_connection_hdl = hdl;
    }

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
     * If a shutdown handler is set, call it and pass through its return error
     * code. Otherwise assume there is nothing to do and pass through a success
     * code.
     *
     * @param handler The `shutdown_handler` to call back when complete
     */
    void async_shutdown(transport::shutdown_handler handler) {
        lib::error_code ec;

        if (m_shutdown_handler) {
            ec = m_shutdown_handler(m_connection_hdl);
        }

        handler(ec);
    }
private:
    void read(std::istream &in) {
        m_alog.write(log::alevel::devel,"iostream_con read");

        while (in.good()) {
            if (!m_reading) {
                m_elog.write(log::elevel::devel,"write while not reading");
                break;
            }

            in.read(m_buf+m_cursor,static_cast<std::streamsize>(m_len-m_cursor));

            if (in.gcount() == 0) {
                m_elog.write(log::elevel::devel,"read zero bytes");
                break;
            }

            m_cursor += static_cast<size_t>(in.gcount());

            // TODO: error handling
            if (in.bad()) {
                m_reading = false;
                complete_read(make_error_code(error::bad_stream));
            }

            if (m_cursor >= m_bytes_needed) {
                m_reading = false;
                complete_read(lib::error_code());
            }
        }
    }

    size_t read_some_impl(char const * buf, size_t len) {
        m_alog.write(log::alevel::devel,"iostream_con read_some");

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

    // Read space (Protected by m_read_mutex)
    char *          m_buf;
    size_t          m_len;
    size_t          m_bytes_needed;
    read_handler    m_read_handler;
    size_t          m_cursor;

    // transport resources
    std::ostream *  m_output_stream;
    connection_hdl  m_connection_hdl;
    write_handler   m_write_handler;
    vector_write_handler m_vector_write_handler;
    shutdown_handler    m_shutdown_handler;

    bool            m_reading;
    bool const      m_is_server;
    bool            m_is_secure;
    alog_type &     m_alog;
    elog_type &     m_elog;
    std::string     m_remote_endpoint;

    // This lock ensures that only one thread can edit read data for this
    // connection. This is a very coarse lock that is basically locked all the
    // time. The nature of the connection is such that it cannot be
    // parallelized, the locking is here to prevent intra-connection concurrency
    // in order to allow inter-connection concurrency.
    mutex_type      m_read_mutex;
};


} // namespace iostream
} // namespace transport
} // namespace websocketpp

#endif // WEBSOCKETPP_TRANSPORT_IOSTREAM_CON_HPP
