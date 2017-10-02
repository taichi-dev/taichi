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

#ifndef WEBSOCKETPP_MESSAGE_BUFFER_MESSAGE_HPP
#define WEBSOCKETPP_MESSAGE_BUFFER_MESSAGE_HPP

#include <websocketpp/common/memory.hpp>
#include <websocketpp/frame.hpp>

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


/// Represents a buffer for a single WebSocket message.
/**
 *
 *
 */
template <template<class> class con_msg_manager>
class message {
public:
    typedef lib::shared_ptr<message> ptr;

    typedef con_msg_manager<message> con_msg_man_type;
    typedef typename con_msg_man_type::ptr con_msg_man_ptr;
    typedef typename con_msg_man_type::weak_ptr con_msg_man_weak_ptr;

    /// Construct an empty message
    /**
     * Construct an empty message
     */
    message(const con_msg_man_ptr manager)
      : m_manager(manager)
      , m_prepared(false)
      , m_fin(true)
      , m_terminal(false)
      , m_compressed(false) {}

    /// Construct a message and fill in some values
    /**
     *
     */
    message(const con_msg_man_ptr manager, frame::opcode::value op, size_t size = 128)
      : m_manager(manager)
      , m_opcode(op)
      , m_prepared(false)
      , m_fin(true)
      , m_terminal(false)
      , m_compressed(false)
    {
        m_payload.reserve(size);
    }

    /// Return whether or not the message has been prepared for sending
    /**
     * The prepared flag indicates that the message has been prepared by a
     * websocket protocol processor and is ready to be written to the wire.
     *
     * @return whether or not the message has been prepared for sending
     */
    bool get_prepared() const {
        return m_prepared;
    }

    /// Set or clear the flag that indicates that the message has been prepared
    /**
     * This flag should not be set by end user code without a very good reason.
     *
     * @param value The value to set the prepared flag to
     */
    void set_prepared(bool value) {
        m_prepared = value;
    }

    /// Return whether or not the message is flagged as compressed
    /**
     * @return whether or not the message is/should be compressed
     */
    bool get_compressed() const {
        return m_compressed;
    }

    /// Set or clear the compression flag
    /**
     * Setting the compression flag indicates that the data in this message
     * would benefit from compression. If both endpoints negotiate a compression
     * extension WebSocket++ will attempt to compress messages with this flag.
     * Setting this flag does not guarantee that the message will be compressed.
     *
     * @param value The value to set the compressed flag to
     */
    void set_compressed(bool value) {
        m_compressed = value;
    }

    /// Get whether or not the message is terminal
    /**
     * Messages can be flagged as terminal, which results in the connection
     * being close after they are written rather than the implementation going
     * on to the next message in the queue. This is typically used internally
     * for close messages only.
     *
     * @return Whether or not this message is marked terminal
     */
    bool get_terminal() const {
        return m_terminal;
    }

    /// Set the terminal flag
    /**
     * This flag should not be set by end user code without a very good reason.
     *
     * @see get_terminal()
     *
     * @param value The value to set the terminal flag to.
     */
    void set_terminal(bool value) {
        m_terminal = value;
    }
    /// Read the fin bit
    /**
     * A message with the fin bit set will be sent as the last message of its
     * sequence. A message with the fin bit cleared will require subsequent
     * frames of opcode continuation until one of them has the fin bit set.
     *
     * The remote end likely will not deliver any bytes until the frame with the fin
     * bit set has been received.
     *
     * @return Whether or not the fin bit is set
     */
    bool get_fin() const {
        return m_fin;
    }

    /// Set the fin bit
    /**
     * @see get_fin for a more detailed explaination of the fin bit
     *
     * @param value The value to set the fin bit to.
     */
    void set_fin(bool value) {
        m_fin = value;
    }

    /// Return the message opcode
    frame::opcode::value get_opcode() const {
        return m_opcode;
    }

    /// Set the opcode
    void set_opcode(frame::opcode::value op) {
        m_opcode = op;
    }

    /// Return the prepared frame header
    /**
     * This value is typically set by a websocket protocol processor
     * and shouldn't be tampered with.
     */
    std::string const & get_header() const {
        return m_header;
    }

    /// Set prepared frame header
    /**
     * Under normal circumstances this should not be called by end users
     *
     * @param header A string to set the header to.
     */
    void set_header(std::string const & header) {
        m_header = header;
    }

    std::string const & get_extension_data() const {
        return m_extension_data;
    }

    /// Get a reference to the payload string
    /**
     * @return A const reference to the message's payload string
     */
    std::string const & get_payload() const {
        return m_payload;
    }

    /// Get a non-const reference to the payload string
    /**
     * @return A reference to the message's payload string
     */
    std::string & get_raw_payload() {
        return m_payload;
    }

    /// Set payload data
    /**
     * Set the message buffer's payload to the given value.
     *
     * @param payload A string to set the payload to.
     */
    void set_payload(std::string const & payload) {
        m_payload = payload;
    }

    /// Set payload data
    /**
     * Set the message buffer's payload to the given value.
     *
     * @param payload A pointer to a data array to set to.
     * @param len The length of new payload in bytes.
     */
    void set_payload(void const * payload, size_t len) {
        m_payload.reserve(len);
        char const * pl = static_cast<char const *>(payload);
        m_payload.assign(pl, pl + len);
    }

    /// Append payload data
    /**
     * Append data to the message buffer's payload.
     *
     * @param payload A string containing the data array to append.
     */
    void append_payload(std::string const & payload) {
        m_payload.append(payload);
    }

    /// Append payload data
    /**
     * Append data to the message buffer's payload.
     *
     * @param payload A pointer to a data array to append
     * @param len The length of payload in bytes
     */
    void append_payload(void const * payload, size_t len) {
        m_payload.reserve(m_payload.size()+len);
        m_payload.append(static_cast<char const *>(payload),len);
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
        con_msg_man_ptr shared = m_manager.lock();

        if (shared) {
            return shared->recycle(this);
        } else {
            return false;
        }
    }
private:
    con_msg_man_weak_ptr        m_manager;
    std::string                 m_header;
    std::string                 m_extension_data;
    std::string                 m_payload;
    frame::opcode::value        m_opcode;
    bool                        m_prepared;
    bool                        m_fin;
    bool                        m_terminal;
    bool                        m_compressed;
};

} // namespace message_buffer
} // namespace websocketpp

#endif // WEBSOCKETPP_MESSAGE_BUFFER_MESSAGE_HPP
