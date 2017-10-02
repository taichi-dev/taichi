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

#ifndef WEBSOCKETPP_MESSAGE_BUFFER_ALLOC_HPP
#define WEBSOCKETPP_MESSAGE_BUFFER_ALLOC_HPP

#include <websocketpp/common/memory.hpp>

#include <string>

namespace websocketpp {
namespace message_buffer {

/* # message:
 * object that stores a message while it is being sent or received. Contains
 * the message payload itself, the message header, the extension data, and the
 * opcode.
 *
 * # connection_message_manager:
 * An object that manages all of the message_buffers associated with a given
 * connection. Implements the get_message_buffer(size) method that returns
 * a message buffer at least size bytes long.
 *
 * Message buffers are reference counted with shared ownership semantics. Once
 * requested from the manager the requester and it's associated downstream code
 * may keep a pointer to the message indefinitely at a cost of extra resource
 * usage. Once the reference count drops to the point where the manager is the
 * only reference the messages is recycled using whatever method is implemented
 * in the manager.
 *
 * # endpoint_message_manager:
 * An object that manages connection_message_managers. Implements the
 * get_message_manager() method. This is used once by each connection to
 * request the message manager that they are supposed to use to manage message
 * buffers for their own use.
 *
 * TYPES OF CONNECTION_MESSAGE_MANAGERS
 * - allocate a message with the exact size every time one is requested
 * - maintain a pool of pre-allocated messages and return one when needed.
 *   Recycle previously used messages back into the pool
 *
 * TYPES OF ENDPOINT_MESSAGE_MANAGERS
 *  - allocate a new connection manager for each connection. Message pools
 *    become connection specific. This increases memory usage but improves
 *    concurrency.
 *  - allocate a single connection manager and share a pointer to it with all
 *    connections created by this endpoint. The message pool will be shared
 *    among all connections, improving memory usage and performance at the cost
 *    of reduced concurrency
 */

/// Custom deleter for use in shared_ptrs to message.
/**
 * This is used to catch messages about to be deleted and offer the manager the
 * ability to recycle them instead. Message::recycle will return true if it was
 * successfully recycled and false otherwise. In the case of exceptions or error
 * this deleter frees the memory.
 */
template <typename T>
void message_deleter(T* msg) {
    try {
        if (!msg->recycle()) {
            delete msg;
        }
    } catch (...) {
        // TODO: is there a better way to ensure this function doesn't throw?
        delete msg;
    }
}

/// Represents a buffer for a single WebSocket message.
/**
 *
 *
 */
template <typename con_msg_manager>
class message {
public:
    typedef lib::shared_ptr<message> ptr;

    typedef typename con_msg_manager::weak_ptr con_msg_man_ptr;

    message(con_msg_man_ptr manager, size_t size = 128)
      : m_manager(manager)
      , m_payload(size) {}

    frame::opcode::value get_opcode() const {
        return m_opcode;
    }
    const std::string& get_header() const {
        return m_header;
    }
    const std::string& get_extension_data() const {
        return m_extension_data;
    }
    const std::string& get_payload() const {
        return m_payload;
    }

    /// Recycle the message
    /**
     * A request to recycle this message was received. Forward that request to
     * the connection message manager for processing. Errors and exceptions
     * from the manager's recycle member function should be passed back up the
     * call chain. The caller to message::recycle will deal with them.
     *
     * Recycle must *only* be called by the message shared_ptr's destructor.
     * Once recycled successfully, ownership of the memory has been passed to
     * another system and must not be accessed again.
     *
     * @return true if the message was successfully recycled, false otherwise.
     */
    bool recycle() {
        typename con_msg_manager::ptr shared = m_manager.lock();

        if (shared) {
            return shared->(recycle(this));
        } else {
            return false;
        }
    }
private:
    con_msg_man_ptr             m_manager;

    frame::opcode::value        m_opcode;
    std::string                 m_header;
    std::string                 m_extension_data;
    std::string                 m_payload;
};

namespace alloc {

/// A connection message manager that allocates a new message for each
/// request.
template <typename message>
class con_msg_manager {
public:
    typedef lib::shared_ptr<con_msg_manager> ptr;
    typedef lib::weak_ptr<con_msg_manager> weak_ptr;

    typedef typename message::ptr message_ptr;

    /// Get a message buffer with specified size
    /**
     * @param size Minimum size in bytes to request for the message payload.
     *
     * @return A shared pointer to a new message with specified size.
     */
    message_ptr get_message(size_t size) const {
        return lib::make_shared<message>(size);
    }

    /// Recycle a message
    /**
     * This method shouldn't be called. If it is, return false to indicate an
     * error. The rest of the method recycle chain should notice this and free
     * the memory.
     *
     * @param msg The message to be recycled.
     *
     * @return true if the message was successfully recycled, false otherwse.
     */
    bool recycle(message * msg) {
        return false;
    }
};

/// An endpoint message manager that allocates a new manager for each
/// connection.
template <typename con_msg_manager>
class endpoint_msg_manager {
public:
    typedef typename con_msg_manager::ptr con_msg_man_ptr;

    /// Get a pointer to a connection message manager
    /**
     * @return A pointer to the requested connection message manager.
     */
    con_msg_man_ptr get_manager() const {
        return lib::make_shared<con_msg_manager>();
    }
};

} // namespace alloc

namespace pool {

/// A connection messages manager that maintains a pool of messages that is
/// used to fulfill get_message requests.
class con_msg_manager {

};

/// An endpoint manager that maintains a shared pool of connection managers
/// and returns an appropriate one for the requesting connection.
class endpoint_msg_manager {

};

} // namespace pool

} // namespace message_buffer
} // namespace websocketpp

#endif // WEBSOCKETPP_MESSAGE_BUFFER_ALLOC_HPP
