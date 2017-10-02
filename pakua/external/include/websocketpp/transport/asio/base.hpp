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

#ifndef WEBSOCKETPP_TRANSPORT_ASIO_BASE_HPP
#define WEBSOCKETPP_TRANSPORT_ASIO_BASE_HPP

#include <websocketpp/common/asio.hpp>
#include <websocketpp/common/cpp11.hpp>
#include <websocketpp/common/functional.hpp>
#include <websocketpp/common/system_error.hpp>
#include <websocketpp/common/type_traits.hpp>

#include <string>

namespace websocketpp {
namespace transport {
/// Transport policy that uses asio
/**
 * This policy uses a single asio io_service to provide transport
 * services to a WebSocket++ endpoint.
 */
namespace asio {

// Class to manage the memory to be used for handler-based custom allocation.
// It contains a single block of memory which may be returned for allocation
// requests. If the memory is in use when an allocation request is made, the
// allocator delegates allocation to the global heap.
class handler_allocator {
public:
    static const size_t size = 1024;
    
    handler_allocator() : m_in_use(false) {}

#ifdef _WEBSOCKETPP_DEFAULT_DELETE_FUNCTIONS_
	handler_allocator(handler_allocator const & cpy) = delete;
	handler_allocator & operator =(handler_allocator const &) = delete;
#endif

    void * allocate(std::size_t memsize) {
        if (!m_in_use && memsize < size) {
            m_in_use = true;
            return static_cast<void*>(&m_storage);
        } else {
            return ::operator new(memsize);
        }
    }

    void deallocate(void * pointer) {
        if (pointer == &m_storage) {
            m_in_use = false;
        } else {
            ::operator delete(pointer);
        }
    }

private:
    // Storage space used for handler-based custom memory allocation.
    lib::aligned_storage<size>::type m_storage;

    // Whether the handler-based custom allocation storage has been used.
    bool m_in_use;
};

// Wrapper class template for handler objects to allow handler memory
// allocation to be customised. Calls to operator() are forwarded to the
// encapsulated handler.
template <typename Handler>
class custom_alloc_handler {
public:
    custom_alloc_handler(handler_allocator& a, Handler h)
      : allocator_(a),
        handler_(h)
    {}

    template <typename Arg1>
    void operator()(Arg1 arg1) {
        handler_(arg1);
    }

    template <typename Arg1, typename Arg2>
    void operator()(Arg1 arg1, Arg2 arg2) {
        handler_(arg1, arg2);
    }

    friend void* asio_handler_allocate(std::size_t size,
        custom_alloc_handler<Handler> * this_handler)
    {
        return this_handler->allocator_.allocate(size);
    }

    friend void asio_handler_deallocate(void* pointer, std::size_t /*size*/,
        custom_alloc_handler<Handler> * this_handler)
    {
        this_handler->allocator_.deallocate(pointer);
    }

private:
    handler_allocator & allocator_;
    Handler handler_;
};

// Helper function to wrap a handler object to add custom allocation.
template <typename Handler>
inline custom_alloc_handler<Handler> make_custom_alloc_handler(
    handler_allocator & a, Handler h)
{
    return custom_alloc_handler<Handler>(a, h);
}







// Forward declaration of class endpoint so that it can be friended/referenced
// before being included.
template <typename config>
class endpoint;

typedef lib::function<void (lib::asio::error_code const & ec,
    size_t bytes_transferred)> async_read_handler;

typedef lib::function<void (lib::asio::error_code const & ec,
    size_t bytes_transferred)> async_write_handler;

typedef lib::function<void (lib::error_code const & ec)> pre_init_handler;

// handle_timer: dynamic parameters, multiple copies
// handle_proxy_write
// handle_proxy_read
// handle_async_write
// handle_pre_init


/// Asio transport errors
namespace error {
enum value {
    /// Catch-all error for transport policy errors that don't fit in other
    /// categories
    general = 1,

    /// async_read_at_least call requested more bytes than buffer can store
    invalid_num_bytes,

    /// there was an error in the underlying transport library
    pass_through,

    /// The connection to the requested proxy server failed
    proxy_failed,

    /// Invalid Proxy URI
    proxy_invalid,

    /// Invalid host or service
    invalid_host_service
};

/// Asio transport error category
class category : public lib::error_category {
public:
    char const * name() const _WEBSOCKETPP_NOEXCEPT_TOKEN_ {
        return "websocketpp.transport.asio";
    }

    std::string message(int value) const {
        switch(value) {
            case error::general:
                return "Generic asio transport policy error";
            case error::invalid_num_bytes:
                return "async_read_at_least call requested more bytes than buffer can store";
            case error::pass_through:
                return "Underlying Transport Error";
            case error::proxy_failed:
                return "Proxy connection failed";
            case error::proxy_invalid:
                return "Invalid proxy URI";
            case error::invalid_host_service:
                return "Invalid host or service";
            default:
                return "Unknown";
        }
    }
};

/// Get a reference to a static copy of the asio transport error category
inline lib::error_category const & get_category() {
    static category instance;
    return instance;
}

/// Create an error code with the given value and the asio transport category
inline lib::error_code make_error_code(error::value e) {
    return lib::error_code(static_cast<int>(e), get_category());
}

} // namespace error
} // namespace asio
} // namespace transport
} // namespace websocketpp

_WEBSOCKETPP_ERROR_CODE_ENUM_NS_START_
template<> struct is_error_code_enum<websocketpp::transport::asio::error::value>
{
    static bool const value = true;
};
_WEBSOCKETPP_ERROR_CODE_ENUM_NS_END_
#endif // WEBSOCKETPP_TRANSPORT_ASIO_HPP
