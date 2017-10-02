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

#ifndef HTTP_CONSTANTS_HPP
#define HTTP_CONSTANTS_HPP

#include <exception>
#include <map>
#include <string>
#include <vector>
#include <utility>

namespace websocketpp {
/// HTTP handling support
namespace http {
    /// The type of an HTTP attribute list
    /**
     * The attribute list is an unordered key/value map. Encoded attribute
     * values are delimited by semicolons.
     */
    typedef std::map<std::string,std::string> attribute_list;

    /// The type of an HTTP parameter list
    /**
     * The parameter list is an ordered pairing of a parameter and its
     * associated attribute list. Encoded parameter values are delimited by
     * commas.
     */
    typedef std::vector< std::pair<std::string,attribute_list> > parameter_list;

    /// Literal value of the HTTP header delimiter
    static char const header_delimiter[] = "\r\n";

    /// Literal value of the HTTP header separator
    static char const header_separator[] = ":";

    /// Literal value of an empty header
    static std::string const empty_header;

    /// Maximum size in bytes before rejecting an HTTP header as too big.
    size_t const max_header_size = 16000;
    
    /// Default Maximum size in bytes for HTTP message bodies.
    size_t const max_body_size = 32000000;

    /// Number of bytes to use for temporary istream read buffers
    size_t const istream_buffer = 512;

    /// invalid HTTP token characters
    /**
     * 0x00 - 0x32, 0x7f-0xff
     * ( ) < > @ , ; : \ " / [ ] ? = { }
     */
    static char const header_token[] = {
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, // 00..0f
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, // 10..1f
        0,1,0,1,1,1,1,1,0,0,1,1,0,1,1,0, // 20..2f
        1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0, // 30..3f
        0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, // 40..4f
        1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1, // 50..5f
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, // 60..6f
        1,1,1,1,1,1,1,1,1,1,1,0,1,0,1,0, // 70..7f
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, // 80..8f
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, // 90..9f
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, // a0..af
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, // b0..bf
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, // c0..cf
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, // d0..df
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, // e0..ef
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, // f0..ff
    };

    /// Is the character a token
    inline bool is_token_char(unsigned char c) {
        return (header_token[c] == 1);
    }

    /// Is the character a non-token
    inline bool is_not_token_char(unsigned char c) {
        return !header_token[c];
    }

    /// Is the character whitespace
    /**
     * whitespace is space (32) or horizontal tab (9)
     */
    inline bool is_whitespace_char(unsigned char c) {
        return (c == 9 || c == 32);
    }

    /// Is the character non-whitespace
    inline bool is_not_whitespace_char(unsigned char c) {
        return (c != 9 && c != 32);
    }

    /// HTTP Status codes
    namespace status_code {
        enum value {
            uninitialized = 0,

            continue_code = 100,
            switching_protocols = 101,

            ok = 200,
            created = 201,
            accepted = 202,
            non_authoritative_information = 203,
            no_content = 204,
            reset_content = 205,
            partial_content = 206,

            multiple_choices = 300,
            moved_permanently = 301,
            found = 302,
            see_other = 303,
            not_modified = 304,
            use_proxy = 305,
            temporary_redirect = 307,

            bad_request = 400,
            unauthorized = 401,
            payment_required = 402,
            forbidden = 403,
            not_found = 404,
            method_not_allowed = 405,
            not_acceptable = 406,
            proxy_authentication_required = 407,
            request_timeout = 408,
            conflict = 409,
            gone = 410,
            length_required = 411,
            precondition_failed = 412,
            request_entity_too_large = 413,
            request_uri_too_long = 414,
            unsupported_media_type = 415,
            request_range_not_satisfiable = 416,
            expectation_failed = 417,
            im_a_teapot = 418,
            upgrade_required = 426,
            precondition_required = 428,
            too_many_requests = 429,
            request_header_fields_too_large = 431,

            internal_server_error = 500,
            not_implemented = 501,
            bad_gateway = 502,
            service_unavailable = 503,
            gateway_timeout = 504,
            http_version_not_supported = 505,
            not_extended = 510,
            network_authentication_required = 511
        };

        // TODO: should this be inline?
        inline std::string get_string(value c) {
            switch (c) {
                case uninitialized:
                    return "Uninitialized";
                case continue_code:
                    return "Continue";
                case switching_protocols:
                    return "Switching Protocols";
                case ok:
                    return "OK";
                case created:
                    return "Created";
                case accepted:
                    return "Accepted";
                case non_authoritative_information:
                    return "Non Authoritative Information";
                case no_content:
                    return "No Content";
                case reset_content:
                    return "Reset Content";
                case partial_content:
                    return "Partial Content";
                case multiple_choices:
                    return "Multiple Choices";
                case moved_permanently:
                    return "Moved Permanently";
                case found:
                    return "Found";
                case see_other:
                    return "See Other";
                case not_modified:
                    return "Not Modified";
                case use_proxy:
                    return "Use Proxy";
                case temporary_redirect:
                    return "Temporary Redirect";
                case bad_request:
                    return "Bad Request";
                case unauthorized:
                    return "Unauthorized";
                case payment_required:
                    return "Payment Required";
                case forbidden:
                    return "Forbidden";
                case not_found:
                    return "Not Found";
                case method_not_allowed:
                    return "Method Not Allowed";
                case not_acceptable:
                    return "Not Acceptable";
                case proxy_authentication_required:
                    return "Proxy Authentication Required";
                case request_timeout:
                    return "Request Timeout";
                case conflict:
                    return "Conflict";
                case gone:
                    return "Gone";
                case length_required:
                    return "Length Required";
                case precondition_failed:
                    return "Precondition Failed";
                case request_entity_too_large:
                    return "Request Entity Too Large";
                case request_uri_too_long:
                    return "Request-URI Too Long";
                case unsupported_media_type:
                    return "Unsupported Media Type";
                case request_range_not_satisfiable:
                    return "Requested Range Not Satisfiable";
                case expectation_failed:
                    return "Expectation Failed";
                case im_a_teapot:
                    return "I'm a teapot";
                case upgrade_required:
                    return "Upgrade Required";
                case precondition_required:
                    return "Precondition Required";
                case too_many_requests:
                    return "Too Many Requests";
                case request_header_fields_too_large:
                    return "Request Header Fields Too Large";
                case internal_server_error:
                    return "Internal Server Error";
                case not_implemented:
                    return "Not Implemented";
                case bad_gateway:
                    return "Bad Gateway";
                case service_unavailable:
                    return "Service Unavailable";
                case gateway_timeout:
                    return "Gateway Timeout";
                case http_version_not_supported:
                    return "HTTP Version Not Supported";
                case not_extended:
                    return "Not Extended";
                case network_authentication_required:
                    return "Network Authentication Required";
                default:
                    return "Unknown";
            }
        }
    }

    class exception : public std::exception {
    public:
        exception(const std::string& log_msg,
                  status_code::value error_code,
                  const std::string& error_msg = std::string(),
                  const std::string& body = std::string())
          : m_msg(log_msg)
          , m_error_msg(error_msg)
          , m_body(body)
          , m_error_code(error_code) {}

        ~exception() throw() {}

        virtual const char* what() const throw() {
            return m_msg.c_str();
        }

        std::string         m_msg;
        std::string         m_error_msg;
        std::string         m_body;
        status_code::value  m_error_code;
    };
}
}

#endif // HTTP_CONSTANTS_HPP
