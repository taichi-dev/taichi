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

#ifndef WEBSOCKETPP_TRANSPORT_DEBUG_HPP
#define WEBSOCKETPP_TRANSPORT_DEBUG_HPP

#include <websocketpp/common/memory.hpp>
#include <websocketpp/logger/levels.hpp>

#include <websocketpp/transport/base/endpoint.hpp>
#include <websocketpp/transport/debug/connection.hpp>

namespace websocketpp {
namespace transport {
namespace debug {

template <typename config>
class endpoint {
public:
    /// Type of this endpoint transport component
    typedef endpoint type;
    /// Type of a pointer to this endpoint transport component
    typedef lib::shared_ptr<type> ptr;

    /// Type of this endpoint's concurrency policy
    typedef typename config::concurrency_type concurrency_type;
    /// Type of this endpoint's error logging policy
    typedef typename config::elog_type elog_type;
    /// Type of this endpoint's access logging policy
    typedef typename config::alog_type alog_type;

    /// Type of this endpoint transport component's associated connection
    /// transport component.
    typedef debug::connection<config> transport_con_type;
    /// Type of a shared pointer to this endpoint transport component's
    /// associated connection transport component
    typedef typename transport_con_type::ptr transport_con_ptr;

    // generate and manage our own io_service
    explicit endpoint()
    {
        //std::cout << "transport::iostream::endpoint constructor" << std::endl;
    }

    /// Set whether or not endpoint can create secure connections
    /**
     * TODO: docs
     *
     * Setting this value only indicates whether or not the endpoint is capable
     * of producing and managing secure connections. Connections produced by
     * this endpoint must also be individually flagged as secure if they are.
     *
     * @since 0.3.0-alpha4
     *
     * @param value Whether or not the endpoint can create secure connections.
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
     *
     * @param a A pointer to the access logger to use.
     * @param e A pointer to the error logger to use.
     */
    void init_logging(alog_type *, elog_type *) {}

    /// Initiate a new connection
    /**
     * @param tcon A pointer to the transport connection component of the
     * connection to connect.
     * @param u A URI pointer to the URI to connect to.
     * @param cb The function to call back with the results when complete.
     */
    void async_connect(transport_con_ptr, uri_ptr, connect_handler cb) {
        cb(lib::error_code());
    }

    /// Initialize a connection
    /**
     * Init is called by an endpoint once for each newly created connection.
     * It's purpose is to give the transport policy the chance to perform any
     * transport specific initialization that couldn't be done via the default
     * constructor.
     *
     * @param tcon A pointer to the transport portion of the connection.
     * @return A status code indicating the success or failure of the operation
     */
    lib::error_code init(transport_con_ptr) {
        return lib::error_code();
    }
private:

};

} // namespace debug
} // namespace transport
} // namespace websocketpp

#endif // WEBSOCKETPP_TRANSPORT_DEBUG_HPP
