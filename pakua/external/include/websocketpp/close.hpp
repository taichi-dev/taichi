
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

#ifndef WEBSOCKETPP_CLOSE_HPP
#define WEBSOCKETPP_CLOSE_HPP

/** \file
 * A package of types and methods for manipulating WebSocket close codes.
 */

#include <websocketpp/error.hpp>
#include <websocketpp/common/network.hpp>
#include <websocketpp/common/stdint.hpp>
#include <websocketpp/utf8_validator.hpp>

#include <string>

namespace websocketpp {
/// A package of types and methods for manipulating WebSocket close codes.
namespace close {
/// A package of types and methods for manipulating WebSocket close status'
namespace status {
    /// The type of a close code value.
    typedef uint16_t value;

    /// A blank value for internal use.
    static value const blank = 0;

    /// Close the connection without a WebSocket close handshake.
    /**
     * This special value requests that the WebSocket connection be closed
     * without performing the WebSocket closing handshake. This does not comply
     * with RFC6455, but should be safe to do if necessary. This could be useful
     * for clients that need to disconnect quickly and cannot afford the
     * complete handshake.
     */
    static value const omit_handshake = 1;

    /// Close the connection with a forced TCP drop.
    /**
     * This special value requests that the WebSocket connection be closed by
     * forcibly dropping the TCP connection. This will leave the other side of
     * the connection with a broken connection and some expensive timeouts. this
     * should not be done except in extreme cases or in cases of malicious
     * remote endpoints.
     */
    static value const force_tcp_drop = 2;

    /// Normal closure, meaning that the purpose for which the connection was
    /// established has been fulfilled.
    static value const normal = 1000;

    /// The endpoint was "going away", such as a server going down or a browser
    /// navigating away from a page.
    static value const going_away = 1001;

    /// A protocol error occurred.
    static value const protocol_error = 1002;

    /// The connection was terminated because an endpoint received a type of
    /// data it cannot accept.
    /**
     * (e.g., an endpoint that understands only text data MAY send this if it
     * receives a binary message).
     */
    static value const unsupported_data = 1003;

    /// A dummy value to indicate that no status code was received.
    /**
     * This value is illegal on the wire.
     */
    static value const no_status = 1005;

    /// A dummy value to indicate that the connection was closed abnormally.
    /**
     * In such a case there was no close frame to extract a value from. This
     * value is illegal on the wire.
     */
    static value const abnormal_close = 1006;

    /// An endpoint received message data inconsistent with its type.
    /**
     * For example: Invalid UTF8 bytes in a text message.
     */
    static value const invalid_payload = 1007;

    /// An endpoint received a message that violated its policy.
    /**
     * This is a generic status code that can be returned when there is no other
     * more suitable status code (e.g., 1003 or 1009) or if there is a need to
     * hide specific details about the policy.
     */
    static value const policy_violation = 1008;

    /// An endpoint received a message too large to process.
    static value const message_too_big = 1009;

    /// A client expected the server to accept a required extension request
    /**
     * The list of extensions that are needed SHOULD appear in the /reason/ part
     * of the Close frame. Note that this status code is not used by the server,
     * because it can fail the WebSocket handshake instead.
     */
    static value const extension_required = 1010;

    /// An endpoint encountered an unexpected condition that prevented it from
    /// fulfilling the request.
    static value const internal_endpoint_error = 1011;

    /// Indicates that the service is restarted. A client may reconnect and if
    /// if it chooses to do so, should reconnect using a randomized delay of
    /// 5-30s
    static value const service_restart = 1012;

    /// Indicates that the service is experiencing overload. A client should
    /// only connect to a different IP (when there are multiple for the target)
    /// or reconnect to the same IP upon user action.
    static value const try_again_later = 1013;

    /// An endpoint failed to perform a TLS handshake
    /**
     * Designated for use in applications expecting a status code to indicate
     * that the connection was closed due to a failure to perform a TLS
     * handshake (e.g., the server certificate can't be verified). This value is
     * illegal on the wire.
     */
    static value const tls_handshake = 1015;
    
    /// A generic subprotocol error
    /**
     * Indicates that a subprotocol error occurred. Typically this involves
     * receiving a message that is not formatted as a valid message for the
     * subprotocol in use.
     */
    static value const subprotocol_error = 3000;
    
    /// A invalid subprotocol data
    /**
     * Indicates that data was received that violated the specification of the
     * subprotocol in use.
     */
    static value const invalid_subprotocol_data = 3001;

    /// First value in range reserved for future protocol use
    static value const rsv_start = 1016;
    /// Last value in range reserved for future protocol use
    static value const rsv_end = 2999;

    /// Test whether a close code is in a reserved range
    /**
     * @param [in] code The code to test
     * @return Whether or not code is reserved
     */
    inline bool reserved(value code) {
        return ((code >= rsv_start && code <= rsv_end) ||
                code == 1004 || code == 1014);
    }

    /// First value in range that is always invalid on the wire
    static value const invalid_low = 999;
    /// Last value in range that is always invalid on the wire
    static value const invalid_high = 5000;

    /// Test whether a close code is invalid on the wire
    /**
     * @param [in] code The code to test
     * @return Whether or not code is invalid on the wire
     */
    inline bool invalid(value code) {
        return (code <= invalid_low || code >= invalid_high ||
                code == no_status || code == abnormal_close ||
                code == tls_handshake);
    }

    /// Determine if the code represents an unrecoverable error
    /**
     * There is a class of errors for which once they are discovered normal
     * WebSocket functionality can no longer occur. This function determines
     * if a given code is one of these values. This information is used to
     * determine if the system has the capability of waiting for a close
     * acknowledgement or if it should drop the TCP connection immediately
     * after sending its close frame.
     *
     * @param [in] code The value to test.
     * @return True if the code represents an unrecoverable error
     */
    inline bool terminal(value code) {
        return (code == protocol_error || code == invalid_payload ||
                code == policy_violation || code == message_too_big ||
                 code == internal_endpoint_error);
    }
    
    /// Return a human readable interpretation of a WebSocket close code
    /**
     * See https://tools.ietf.org/html/rfc6455#section-7.4 for more details.
     *
     * @since 0.3.0
     *
     * @param [in] code The code to look up.
     * @return A human readable interpretation of the code.
     */
    inline std::string get_string(value code) {
        switch (code) {
            case normal:
                return "Normal close";
            case going_away:
                return "Going away";
            case protocol_error:
                return "Protocol error";
            case unsupported_data:
                return "Unsupported data";
            case no_status:
                return "No status set";
            case abnormal_close:
                return "Abnormal close";
            case invalid_payload:
                return "Invalid payload";
            case policy_violation:
                return "Policy violoation";
            case message_too_big:
                return "Message too big";
            case extension_required:
                return "Extension required";
            case internal_endpoint_error:
                return "Internal endpoint error";
            case tls_handshake:
                return "TLS handshake failure";
            case subprotocol_error:
                return "Generic subprotocol error";
            case invalid_subprotocol_data:
                return "Invalid subprotocol data";
            default:
                return "Unknown";
        }
    }
} // namespace status

/// Type used to convert close statuses between integer and wire representations
union code_converter {
    uint16_t i;
    char c[2];
};

/// Extract a close code value from a close payload
/**
 * If there is no close value (ie string is empty) status::no_status is
 * returned. If a code couldn't be extracted (usually do to a short or
 * otherwise mangled payload) status::protocol_error is returned and the ec
 * value is flagged as an error. Note that this case is different than the case
 * where protocol error is received over the wire.
 *
 * If the value is in an invalid or reserved range ec is set accordingly.
 *
 * @param [in] payload Close frame payload value received over the wire.
 * @param [out] ec Set to indicate what error occurred, if any.
 * @return The extracted value
 */
inline status::value extract_code(std::string const & payload, lib::error_code
    & ec)
{
    ec = lib::error_code();

    if (payload.size() == 0) {
        return status::no_status;
    } else if (payload.size() == 1) {
        ec = make_error_code(error::bad_close_code);
        return status::protocol_error;
    }

    code_converter val;

    val.c[0] = payload[0];
    val.c[1] = payload[1];

    status::value code(ntohs(val.i));

    if (status::invalid(code)) {
        ec = make_error_code(error::invalid_close_code);
    }

    if (status::reserved(code)) {
        ec = make_error_code(error::reserved_close_code);
    }

    return code;
}

/// Extract the reason string from a close payload
/**
 * The string should be a valid UTF8 message. error::invalid_utf8 will be set if
 * the function extracts a reason that is not valid UTF8.
 *
 * @param [in] payload The payload string to extract a reason from.
 * @param [out] ec Set to indicate what error occurred, if any.
 * @return The reason string.
 */
inline std::string extract_reason(std::string const & payload, lib::error_code
    & ec)
{
    std::string reason;
    ec = lib::error_code();

    if (payload.size() > 2) {
        reason.append(payload.begin()+2,payload.end());
    }

    if (!websocketpp::utf8_validator::validate(reason)) {
        ec = make_error_code(error::invalid_utf8);
    }

    return reason;
}

} // namespace close
} // namespace websocketpp

#endif // WEBSOCKETPP_CLOSE_HPP
