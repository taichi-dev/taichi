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

#ifndef WEBSOCKETPP_TRANSPORT_BASE_CON_HPP
#define WEBSOCKETPP_TRANSPORT_BASE_CON_HPP

#include <websocketpp/common/cpp11.hpp>
#include <websocketpp/common/connection_hdl.hpp>
#include <websocketpp/common/functional.hpp>
#include <websocketpp/common/system_error.hpp>

#include <string>

namespace websocketpp {
/// Transport policies provide network connectivity and timers
/**
 * ### Connection Interface
 *
 * Transport connection components needs to provide:
 *
 * **init**\n
 * `void init(init_handler handler)`\n
 * Called once shortly after construction to give the policy the chance to
 * perform one time initialization. When complete, the policy must call the
 * supplied `init_handler` to continue setup. The handler takes one argument
 * with the error code if any. If an error is returned here setup will fail and
 * the connection will be aborted or terminated.
 *
 * WebSocket++ will call init only once. The transport must call `handler`
 * exactly once.
 *
 * **async_read_at_least**\n
 * `void async_read_at_least(size_t num_bytes, char *buf, size_t len,
 * read_handler handler)`\n
 * start an async read for at least num_bytes and at most len
 * bytes into buf. Call handler when done with number of bytes read.
 *
 * WebSocket++ promises to have only one async_read_at_least in flight at a
 * time. The transport must promise to only call read_handler once per async
 * read.
 *
 * **async_write**\n
 * `void async_write(const char* buf, size_t len, write_handler handler)`\n
 * `void async_write(std::vector<buffer> & bufs, write_handler handler)`\n
 * Start a write of all of the data in buf or bufs. In second case data is
 * written sequentially and in place without copying anything to a temporary
 * location.
 *
 * Websocket++ promises to have only one async_write in flight at a time.
 * The transport must promise to only call the write_handler once per async
 * write
 *
 * **set_handle**\n
 * `void set_handle(connection_hdl hdl)`\n
 * Called by WebSocket++ to let this policy know the hdl to the connection. It
 * may be stored for later use or ignored/discarded. This handle should be used
 * if the policy adds any connection handlers. Connection handlers must be
 * called with the handle as the first argument so that the handler code knows
 * which connection generated the callback.
 *
 * **set_timer**\n
 * `timer_ptr set_timer(long duration, timer_handler handler)`\n
 * WebSocket++ uses the timers provided by the transport policy as the
 * implementation of timers is often highly coupled with the implementation of
 * the networking event loops.
 *
 * Transport timer support is an optional feature. A transport method may elect
 * to implement a dummy timer object and have this method return an empty
 * pointer. If so, all timer related features of WebSocket++ core will be
 * disabled. This includes many security features designed to prevent denial of
 * service attacks. Use timer-free transport policies with caution.
 *
 * **get_remote_endpoint**\n
 * `std::string get_remote_endpoint()`\n
 * retrieve address of remote endpoint
 *
 * **is_secure**\n
 * `void is_secure()`\n
 * whether or not the connection to the remote endpoint is secure
 *
 * **dispatch**\n
 * `lib::error_code dispatch(dispatch_handler handler)`: invoke handler within
 * the transport's event system if it uses one. Otherwise, this method should
 * simply call `handler` immediately.
 *
 * **async_shutdown**\n
 * `void async_shutdown(shutdown_handler handler)`\n
 * Perform any cleanup necessary (if any). Call `handler` when complete.
 */
namespace transport {

/// The type and signature of the callback passed to the init hook
typedef lib::function<void(lib::error_code const &)> init_handler;

/// The type and signature of the callback passed to the read method
typedef lib::function<void(lib::error_code const &,size_t)> read_handler;

/// The type and signature of the callback passed to the write method
typedef lib::function<void(lib::error_code const &)> write_handler;

/// The type and signature of the callback passed to the read method
typedef lib::function<void(lib::error_code const &)> timer_handler;

/// The type and signature of the callback passed to the shutdown method
typedef lib::function<void(lib::error_code const &)> shutdown_handler;

/// The type and signature of the callback passed to the interrupt method
typedef lib::function<void()> interrupt_handler;

/// The type and signature of the callback passed to the dispatch method
typedef lib::function<void()> dispatch_handler;

/// A simple utility buffer class
struct buffer {
    buffer(char const * b, size_t l) : buf(b),len(l) {}

    char const * buf;
    size_t len;
};

/// Generic transport related errors
namespace error {
enum value {
    /// Catch-all error for transport policy errors that don't fit in other
    /// categories
    general = 1,

    /// underlying transport pass through
    pass_through,

    /// async_read_at_least call requested more bytes than buffer can store
    invalid_num_bytes,

    /// async_read called while another async_read was in progress
    double_read,

    /// Operation aborted
    operation_aborted,

    /// Operation not supported
    operation_not_supported,

    /// End of file
    eof,

    /// TLS short read
    tls_short_read,

    /// Timer expired
    timeout,

    /// read or write after shutdown
    action_after_shutdown,

    /// Other TLS error
    tls_error
};

class category : public lib::error_category {
    public:
    category() {}

    char const * name() const _WEBSOCKETPP_NOEXCEPT_TOKEN_ {
        return "websocketpp.transport";
    }

    std::string message(int value) const {
        switch(value) {
            case general:
                return "Generic transport policy error";
            case pass_through:
                return "Underlying Transport Error";
            case invalid_num_bytes:
                return "async_read_at_least call requested more bytes than buffer can store";
            case operation_aborted:
                return "The operation was aborted";
            case operation_not_supported:
                return "The operation is not supported by this transport";
            case eof:
                return "End of File";
            case tls_short_read:
                return "TLS Short Read";
            case timeout:
                return "Timer Expired";
            case action_after_shutdown:
                return "A transport action was requested after shutdown";
            case tls_error:
                return "Generic TLS related error";
            default:
                return "Unknown";
        }
    }
};

inline lib::error_category const & get_category() {
    static category instance;
    return instance;
}

inline lib::error_code make_error_code(error::value e) {
    return lib::error_code(static_cast<int>(e), get_category());
}

} // namespace error
} // namespace transport
} // namespace websocketpp
_WEBSOCKETPP_ERROR_CODE_ENUM_NS_START_
template<> struct is_error_code_enum<websocketpp::transport::error::value>
{
    static bool const value = true;
};
_WEBSOCKETPP_ERROR_CODE_ENUM_NS_END_

#endif // WEBSOCKETPP_TRANSPORT_BASE_CON_HPP
