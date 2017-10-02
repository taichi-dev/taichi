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

#ifndef WEBSOCKETPP_PROCESSOR_HPP
#define WEBSOCKETPP_PROCESSOR_HPP

#include <websocketpp/processors/base.hpp>
#include <websocketpp/common/system_error.hpp>

#include <websocketpp/close.hpp>
#include <websocketpp/utilities.hpp>
#include <websocketpp/uri.hpp>

#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace websocketpp {
/// Processors encapsulate the protocol rules specific to each WebSocket version
/**
 * The processors namespace includes a number of free functions that operate on
 * various WebSocket related data structures and perform processing that is not
 * related to specific versions of the protocol.
 *
 * It also includes the abstract interface for the protocol specific processing
 * engines. These engines wrap all of the logic necessary for parsing and
 * validating WebSocket handshakes and messages of specific protocol version
 * and set of allowed extensions.
 *
 * An instance of a processor represents the state of a single WebSocket
 * connection of the associated version. One processor instance is needed per
 * logical WebSocket connection.
 */
namespace processor {

/// Determine whether or not a generic HTTP request is a WebSocket handshake
/**
 * @param r The HTTP request to read.
 *
 * @return True if the request is a WebSocket handshake, false otherwise
 */
template <typename request_type>
bool is_websocket_handshake(request_type& r) {
    using utility::ci_find_substr;

    std::string const & upgrade_header = r.get_header("Upgrade");

    if (ci_find_substr(upgrade_header, constants::upgrade_token,
        sizeof(constants::upgrade_token)-1) == upgrade_header.end())
    {
        return false;
    }

    std::string const & con_header = r.get_header("Connection");

    if (ci_find_substr(con_header, constants::connection_token,
        sizeof(constants::connection_token)-1) == con_header.end())
    {
        return false;
    }

    return true;
}

/// Extract the version from a WebSocket handshake request
/**
 * A blank version header indicates a spec before versions were introduced.
 * The only such versions in shipping products are Hixie Draft 75 and Hixie
 * Draft 76. Draft 75 is present in Chrome 4-5 and Safari 5.0.0, Draft 76 (also
 * known as hybi 00 is present in Chrome 6-13 and Safari 5.0.1+. As
 * differentiating between these two sets of browsers is very difficult and
 * Safari 5.0.1+ accounts for the vast majority of cases in the wild this
 * function assumes that all handshakes without a valid version header are
 * Hybi 00.
 *
 * @param r The WebSocket handshake request to read.
 *
 * @return The WebSocket handshake version or -1 if there was an extraction
 * error.
 */
template <typename request_type>
int get_websocket_version(request_type& r) {
    if (!r.ready()) {
        return -2;
    }
    
    if (r.get_header("Sec-WebSocket-Version").empty()) {
        return 0;
    }

    int version;
    std::istringstream ss(r.get_header("Sec-WebSocket-Version"));

    if ((ss >> version).fail()) {
        return -1;
    }

    return version;
}

/// Extract a URI ptr from the host header of the request
/**
 * @param request The request to extract the Host header from.
 *
 * @param scheme The scheme under which this request was received (ws, wss,
 * http, https, etc)
 *
 * @return A uri_pointer that encodes the value of the host header.
 */
template <typename request_type>
uri_ptr get_uri_from_host(request_type & request, std::string scheme) {
    std::string h = request.get_header("Host");

    size_t last_colon = h.rfind(":");
    size_t last_sbrace = h.rfind("]");

    // no : = hostname with no port
    // last : before ] = ipv6 literal with no port
    // : with no ] = hostname with port
    // : after ] = ipv6 literal with port
    if (last_colon == std::string::npos ||
        (last_sbrace != std::string::npos && last_sbrace > last_colon))
    {
        return lib::make_shared<uri>(scheme, h, request.get_uri());
    } else {
        return lib::make_shared<uri>(scheme,
                               h.substr(0,last_colon),
                               h.substr(last_colon+1),
                               request.get_uri());
    }
}

/// WebSocket protocol processor abstract base class
template <typename config>
class processor {
public:
    typedef processor<config> type;
    typedef typename config::request_type request_type;
    typedef typename config::response_type response_type;
    typedef typename config::message_type::ptr message_ptr;
    typedef std::pair<lib::error_code,std::string> err_str_pair;

    explicit processor(bool secure, bool p_is_server)
      : m_secure(secure)
      , m_server(p_is_server)
      , m_max_message_size(config::max_message_size)
    {}

    virtual ~processor() {}

    /// Get the protocol version of this processor
    virtual int get_version() const = 0;

    /// Get maximum message size
    /**
     * Get maximum message size. Maximum message size determines the point at which the
     * processor will fail a connection with the message_too_big protocol error.
     *
     * The default is retrieved from the max_message_size value from the template config
     *
     * @since 0.3.0
     */
    size_t get_max_message_size() const {
        return m_max_message_size;
    }
    
    /// Set maximum message size
    /**
     * Set maximum message size. Maximum message size determines the point at which the
     * processor will fail a connection with the message_too_big protocol error.
     *
     * The default is retrieved from the max_message_size value from the template config
     *
     * @since 0.3.0
     *
     * @param new_value The value to set as the maximum message size.
     */
    void set_max_message_size(size_t new_value) {
        m_max_message_size = new_value;
    }

    /// Returns whether or not the permessage_compress extension is implemented
    /**
     * Compile time flag that indicates whether this processor has implemented
     * the permessage_compress extension. By default this is false.
     */
    virtual bool has_permessage_compress() const {
        return false;
    }

    /// Initializes extensions based on the Sec-WebSocket-Extensions header
    /**
     * Reads the Sec-WebSocket-Extensions header and determines if any of the
     * requested extensions are supported by this processor. If they are their
     * settings data is initialized and an extension string to send to the
     * is returned.
     *
     * @param request The request or response headers to look at.
     */
    virtual err_str_pair negotiate_extensions(request_type const &) {
        return err_str_pair();
    }
    
    /// Initializes extensions based on the Sec-WebSocket-Extensions header
    /**
     * Reads the Sec-WebSocket-Extensions header and determines if any of the
     * requested extensions were accepted by the server. If they are their
     * settings data is initialized. If they are not a list of required
     * extensions (if any) is returned. This list may be sent back to the server
     * as a part of the 1010/Extension required close code.
     *
     * @param response The request or response headers to look at.
     */
    virtual err_str_pair negotiate_extensions(response_type const &) {
        return err_str_pair();
    }

    /// validate a WebSocket handshake request for this version
    /**
     * @param request The WebSocket handshake request to validate.
     * is_websocket_handshake(request) must be true and
     * get_websocket_version(request) must equal this->get_version().
     *
     * @return A status code, 0 on success, non-zero for specific sorts of
     * failure
     */
    virtual lib::error_code validate_handshake(request_type const & request) const = 0;

    /// Calculate the appropriate response for this websocket request
    /**
     * @param req The request to process
     *
     * @param subprotocol The subprotocol in use
     *
     * @param res The response to store the processed response in
     *
     * @return An error code, 0 on success, non-zero for other errors
     */
    virtual lib::error_code process_handshake(request_type const & req,
        std::string const & subprotocol, response_type& res) const = 0;

    /// Fill in an HTTP request for an outgoing connection handshake
    /**
     * @param req The request to process.
     *
     * @return An error code, 0 on success, non-zero for other errors
     */
    virtual lib::error_code client_handshake_request(request_type & req,
        uri_ptr uri, std::vector<std::string> const & subprotocols) const = 0;

    /// Validate the server's response to an outgoing handshake request
    /**
     * @param req The original request sent
     * @param res The reponse to generate
     * @return An error code, 0 on success, non-zero for other errors
     */
    virtual lib::error_code validate_server_handshake_response(request_type
        const & req, response_type & res) const = 0;

    /// Given a completed response, get the raw bytes to put on the wire
    virtual std::string get_raw(response_type const & request) const = 0;

    /// Return the value of the header containing the CORS origin.
    virtual std::string const & get_origin(request_type const & request) const = 0;

    /// Extracts requested subprotocols from a handshake request
    /**
     * Extracts a list of all subprotocols that the client has requested in the
     * given opening handshake request.
     *
     * @param [in] req The request to extract from
     * @param [out] subprotocol_list A reference to a vector of strings to store
     * the results in.
     */
    virtual lib::error_code extract_subprotocols(const request_type & req,
        std::vector<std::string> & subprotocol_list) = 0;

    /// Extracts client uri from a handshake request
    virtual uri_ptr get_uri(request_type const & request) const = 0;

    /// process new websocket connection bytes
    /**
     * WebSocket connections are a continous stream of bytes that must be
     * interpreted by a protocol processor into discrete frames.
     *
     * @param buf Buffer from which bytes should be read.
     * @param len Length of buffer
     * @param ec Reference to an error code to return any errors in
     * @return Number of bytes processed
     */
    virtual size_t consume(uint8_t *buf, size_t len, lib::error_code & ec) = 0;

    /// Checks if there is a message ready
    /**
     * Checks if the most recent consume operation processed enough bytes to
     * complete a new WebSocket message. The message can be retrieved by calling
     * get_message() which will reset the internal state to not-ready and allow
     * consume to read more bytes.
     *
     * @return Whether or not a message is ready.
     */
    virtual bool ready() const = 0;

    /// Retrieves the most recently processed message
    /**
     * Retrieves a shared pointer to the recently completed message if there is
     * one. If ready() returns true then there is a message available.
     * Retrieving the message with get_message will reset the state of ready.
     * As such, each new message may be retrieved only once. Calling get_message
     * when there is no message available will result in a null pointer being
     * returned.
     *
     * @return A pointer to the most recently processed message or a null shared
     *         pointer.
     */
    virtual message_ptr get_message() = 0;

    /// Tests whether the processor is in a fatal error state
    virtual bool get_error() const = 0;

    /// Retrieves the number of bytes presently needed by the processor
    /// This value may be used as a hint to the transport layer as to how many
    /// bytes to wait for before running consume again.
    virtual size_t get_bytes_needed() const {
        return 1;
    }

    /// Prepare a data message for writing
    /**
     * Performs validation, masking, compression, etc. will return an error if
     * there was an error, otherwise msg will be ready to be written
     */
    virtual lib::error_code prepare_data_frame(message_ptr in, message_ptr out) = 0;

    /// Prepare a ping frame
    /**
     * Ping preparation is entirely state free. There is no payload validation
     * other than length. Payload need not be UTF-8.
     *
     * @param in The string to use for the ping payload
     * @param out The message buffer to prepare the ping in.
     * @return Status code, zero on success, non-zero on failure
     */
    virtual lib::error_code prepare_ping(std::string const & in, message_ptr out) const 
        = 0;

    /// Prepare a pong frame
    /**
     * Pong preparation is entirely state free. There is no payload validation
     * other than length. Payload need not be UTF-8.
     *
     * @param in The string to use for the pong payload
     * @param out The message buffer to prepare the pong in.
     * @return Status code, zero on success, non-zero on failure
     */
    virtual lib::error_code prepare_pong(std::string const & in, message_ptr out) const 
        = 0;

    /// Prepare a close frame
    /**
     * Close preparation is entirely state free. The code and reason are both
     * subject to validation. Reason must be valid UTF-8. Code must be a valid
     * un-reserved WebSocket close code. Use close::status::no_status to
     * indicate no code. If no code is supplied a reason may not be specified.
     *
     * @param code The close code to send
     * @param reason The reason string to send
     * @param out The message buffer to prepare the fame in
     * @return Status code, zero on success, non-zero on failure
     */
    virtual lib::error_code prepare_close(close::status::value code,
        std::string const & reason, message_ptr out) const = 0;
protected:
    bool const m_secure;
    bool const m_server;
    size_t m_max_message_size;
};

} // namespace processor
} // namespace websocketpp

#endif //WEBSOCKETPP_PROCESSOR_HPP
