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

#ifndef WEBSOCKETPP_PROCESSOR_BASE_HPP
#define WEBSOCKETPP_PROCESSOR_BASE_HPP

#include <websocketpp/close.hpp>
#include <websocketpp/utilities.hpp>
#include <websocketpp/uri.hpp>

#include <websocketpp/common/cpp11.hpp>
#include <websocketpp/common/system_error.hpp>

#include <string>

namespace websocketpp {
namespace processor {

/// Constants related to processing WebSocket connections
namespace constants {

static char const upgrade_token[] = "websocket";
static char const connection_token[] = "upgrade";
static char const handshake_guid[] = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";

} // namespace constants


/// Processor class related error codes
namespace error_cat {
enum value {
    BAD_REQUEST = 0, // Error was the result of improperly formatted user input
    INTERNAL_ERROR = 1, // Error was a logic error internal to WebSocket++
    PROTOCOL_VIOLATION = 2,
    MESSAGE_TOO_BIG = 3,
    PAYLOAD_VIOLATION = 4 // Error was due to receiving invalid payload data
};
} // namespace error_cat

/// Error code category and codes used by all processor types
namespace error {
enum processor_errors {
    /// Catch-all error for processor policy errors that don't fit in other
    /// categories
    general = 1,

    /// Error was the result of improperly formatted user input
    bad_request,

    /// Processor encountered a protocol violation in an incoming message
    protocol_violation,

    /// Processor encountered a message that was too large
    message_too_big,

    /// Processor encountered invalid payload data.
    invalid_payload,

    /// The processor method was called with invalid arguments
    invalid_arguments,

    /// Opcode was invalid for requested operation
    invalid_opcode,

    /// Control frame too large
    control_too_big,

    /// Illegal use of reserved bit
    invalid_rsv_bit,

    /// Fragmented control message
    fragmented_control,

    /// Continuation without message
    invalid_continuation,

    /// Clients may not send unmasked frames
    masking_required,

    /// Servers may not send masked frames
    masking_forbidden,

    /// Payload length not minimally encoded
    non_minimal_encoding,

    /// Not supported on 32 bit systems
    requires_64bit,

    /// Invalid UTF-8 encoding
    invalid_utf8,

    /// Operation required not implemented functionality
    not_implemented,

    /// Invalid HTTP method
    invalid_http_method,

    /// Invalid HTTP version
    invalid_http_version,

    /// Invalid HTTP status
    invalid_http_status,

    /// Missing Required Header
    missing_required_header,

    /// Embedded SHA-1 library error
    sha1_library,

    /// No support for this feature in this protocol version.
    no_protocol_support,

    /// Reserved close code used
    reserved_close_code,

    /// Invalid close code used
    invalid_close_code,

    /// Using a reason requires a close code
    reason_requires_code,

    /// Error parsing subprotocols
    subprotocol_parse_error,

    /// Error parsing extensions
    extension_parse_error,

    /// Extension related operation was ignored because extensions are disabled
    extensions_disabled,
    
    /// Short Ke3 read. Hybi00 requires a third key to be read from the 8 bytes
    /// after the handshake. Less than 8 bytes were read.
    short_key3
};

/// Category for processor errors
class processor_category : public lib::error_category {
public:
    processor_category() {}

    char const * name() const _WEBSOCKETPP_NOEXCEPT_TOKEN_ {
        return "websocketpp.processor";
    }

    std::string message(int value) const {
        switch(value) {
            case error::general:
                return "Generic processor error";
            case error::bad_request:
                return "invalid user input";
            case error::protocol_violation:
                return "Generic protocol violation";
            case error::message_too_big:
                return "A message was too large";
            case error::invalid_payload:
                return "A payload contained invalid data";
            case error::invalid_arguments:
                return "invalid function arguments";
            case error::invalid_opcode:
                return "invalid opcode";
            case error::control_too_big:
                return "Control messages are limited to fewer than 125 characters";
            case error::invalid_rsv_bit:
                return "Invalid use of reserved bits";
            case error::fragmented_control:
                return "Control messages cannot be fragmented";
            case error::invalid_continuation:
                return "Invalid message continuation";
            case error::masking_required:
                return "Clients may not send unmasked frames";
            case error::masking_forbidden:
                return "Servers may not send masked frames";
            case error::non_minimal_encoding:
                return "Payload length was not minimally encoded";
            case error::requires_64bit:
                return "64 bit frames are not supported on 32 bit systems";
            case error::invalid_utf8:
                return "Invalid UTF8 encoding";
            case error::not_implemented:
                return "Operation required not implemented functionality";
            case error::invalid_http_method:
                return "Invalid HTTP method.";
            case error::invalid_http_version:
                return "Invalid HTTP version.";
            case error::invalid_http_status:
                return "Invalid HTTP status.";
            case error::missing_required_header:
                return "A required HTTP header is missing";
            case error::sha1_library:
                return "SHA-1 library error";
            case error::no_protocol_support:
                return "The WebSocket protocol version in use does not support this feature";
            case error::reserved_close_code:
                return "Reserved close code used";
            case error::invalid_close_code:
                return "Invalid close code used";
            case error::reason_requires_code:
                return "Using a close reason requires a valid close code";
            case error::subprotocol_parse_error:
                return "Error parsing subprotocol header";
            case error::extension_parse_error:
                return "Error parsing extension header";
            case error::extensions_disabled:
                return "Extensions are disabled";
            case error::short_key3:
                return "Short Hybi00 Key 3 read";
            default:
                return "Unknown";
        }
    }
};

/// Get a reference to a static copy of the processor error category
inline lib::error_category const & get_processor_category() {
    static processor_category instance;
    return instance;
}

/// Create an error code with the given value and the processor category
inline lib::error_code make_error_code(error::processor_errors e) {
    return lib::error_code(static_cast<int>(e), get_processor_category());
}

/// Converts a processor error_code into a websocket close code
/**
 * Looks up the appropriate WebSocket close code that should be sent after an
 * error of this sort occurred.
 *
 * If the error is not in the processor category close::status::blank is
 * returned.
 *
 * If the error isn't normally associated with reasons to close a connection
 * (such as errors intended to be used internally or delivered to client
 * applications, ex: invalid arguments) then
 * close::status::internal_endpoint_error is returned.
 */
inline close::status::value to_ws(lib::error_code ec) {
    if (ec.category() != get_processor_category()) {
        return close::status::blank;
    }

    switch (ec.value()) {
        case error::protocol_violation:
        case error::control_too_big:
        case error::invalid_opcode:
        case error::invalid_rsv_bit:
        case error::fragmented_control:
        case error::invalid_continuation:
        case error::masking_required:
        case error::masking_forbidden:
        case error::reserved_close_code:
        case error::invalid_close_code:
            return close::status::protocol_error;
        case error::invalid_payload:
        case error::invalid_utf8:
            return close::status::invalid_payload;
        case error::message_too_big:
            return close::status::message_too_big;
        default:
            return close::status::internal_endpoint_error;
    }
}

} // namespace error
} // namespace processor
} // namespace websocketpp

_WEBSOCKETPP_ERROR_CODE_ENUM_NS_START_
template<> struct is_error_code_enum<websocketpp::processor::error::processor_errors>
{
    static bool const value = true;
};
_WEBSOCKETPP_ERROR_CODE_ENUM_NS_END_

#endif //WEBSOCKETPP_PROCESSOR_BASE_HPP
