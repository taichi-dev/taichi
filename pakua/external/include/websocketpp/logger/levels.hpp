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

#ifndef WEBSOCKETPP_LOGGER_LEVELS_HPP
#define WEBSOCKETPP_LOGGER_LEVELS_HPP

#include <websocketpp/common/stdint.hpp>

namespace websocketpp {
namespace log {

/// Type of a channel package
typedef uint32_t level;

/// Package of values for hinting at the nature of a given logger.
/**
 * Used by the library to signal to the logging class a hint that it can use to
 * set itself up. For example, the `access` hint indicates that it is an access
 * log that might be suitable for being printed to an access log file or to cout
 * whereas `error` might be suitable for an error log file or cerr. 
 */
struct channel_type_hint {
    /// Type of a channel type hint value
    typedef uint32_t value;
    
    /// No information
    static value const none = 0;
    /// Access log
    static value const access = 1;
    /// Error log
    static value const error = 2;
};

/// Package of log levels for logging errors
struct elevel {
    /// Special aggregate value representing "no levels"
    static level const none = 0x0;
    /// Low level debugging information (warning: very chatty)
    static level const devel = 0x1;
    /// Information about unusual system states or other minor internal library
    /// problems, less chatty than devel.
    static level const library = 0x2;
    /// Information about minor configuration problems or additional information
    /// about other warnings.
    static level const info = 0x4;
    /// Information about important problems not severe enough to terminate
    /// connections.
    static level const warn = 0x8;
    /// Recoverable error. Recovery may mean cleanly closing the connection with
    /// an appropriate error code to the remote endpoint.
    static level const rerror = 0x10;
    /// Unrecoverable error. This error will trigger immediate unclean
    /// termination of the connection or endpoint.
    static level const fatal = 0x20;
    /// Special aggregate value representing "all levels"
    static level const all = 0xffffffff;

    /// Get the textual name of a channel given a channel id
    /**
     * The id must be that of a single channel. Passing an aggregate channel
     * package results in undefined behavior.
     *
     * @param channel The channel id to look up.
     *
     * @return The name of the specified channel.
     */
    static char const * channel_name(level channel) {
        switch(channel) {
            case devel:
                return "devel";
            case library:
                return "library";
            case info:
                return "info";
            case warn:
                return "warning";
            case rerror:
                return "error";
            case fatal:
                return "fatal";
            default:
                return "unknown";
        }
    }
};

/// Package of log levels for logging access events
struct alevel {
    /// Special aggregate value representing "no levels"
    static level const none = 0x0;
    /// Information about new connections
    /**
     * One line for each new connection that includes a host of information
     * including: the remote address, websocket version, requested resource,
     * http code, remote user agent
     */
    static level const connect = 0x1;
    /// One line for each closed connection. Includes closing codes and reasons.
    static level const disconnect = 0x2;
    /// One line per control frame
    static level const control = 0x4;
    /// One line per frame, includes the full frame header
    static level const frame_header = 0x8;
    /// One line per frame, includes the full message payload (warning: chatty)
    static level const frame_payload = 0x10;
    /// Reserved
    static level const message_header = 0x20;
    /// Reserved
    static level const message_payload = 0x40;
    /// Reserved
    static level const endpoint = 0x80;
    /// Extra information about opening handshakes
    static level const debug_handshake = 0x100;
    /// Extra information about closing handshakes
    static level const debug_close = 0x200;
    /// Development messages (warning: very chatty)
    static level const devel = 0x400;
    /// Special channel for application specific logs. Not used by the library.
    static level const app = 0x800;
    /// Access related to HTTP requests
    static level const http = 0x1000;
    /// One line for each failed WebSocket connection with details
    static level const fail = 0x2000;
    /// Aggregate package representing the commonly used core access channels
    /// Connect, Disconnect, Fail, and HTTP
    static level const access_core = 0x00003003;
    /// Special aggregate value representing "all levels"
    static level const all = 0xffffffff;

    /// Get the textual name of a channel given a channel id
    /**
     * Get the textual name of a channel given a channel id. The id must be that
     * of a single channel. Passing an aggregate channel package results in
     * undefined behavior.
     *
     * @param channel The channelid to look up.
     *
     * @return The name of the specified channel.
     */
    static char const * channel_name(level channel) {
        switch(channel) {
            case connect:
                return "connect";
            case disconnect:
                return "disconnect";
            case control:
                return "control";
            case frame_header:
                return "frame_header";
            case frame_payload:
                return "frame_payload";
            case message_header:
                return "message_header";
            case message_payload:
                return "message_payload";
            case endpoint:
                return "endpoint";
            case debug_handshake:
                return "debug_handshake";
            case debug_close:
                return "debug_close";
            case devel:
                return "devel";
            case app:
                return "application";
            case http:
                return "http";
            case fail:
                return "fail";
            default:
                return "unknown";
        }
    }
};

} // logger
} // websocketpp

#endif //WEBSOCKETPP_LOGGER_LEVELS_HPP
