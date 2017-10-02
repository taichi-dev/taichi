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

#ifndef WEBSOCKETPP_URI_HPP
#define WEBSOCKETPP_URI_HPP

#include <websocketpp/error.hpp>

#include <websocketpp/common/memory.hpp>

#include <algorithm>
#include <sstream>
#include <string>

namespace websocketpp {

// TODO: figure out why this fixes horrible linking errors.

/// Default port for ws://
static uint16_t const uri_default_port = 80;
/// Default port for wss://
static uint16_t const uri_default_secure_port = 443;

class uri {
public:
    explicit uri(std::string const & uri_string) : m_valid(false) {
        std::string::const_iterator it;
        std::string::const_iterator temp;

        int state = 0;

        it = uri_string.begin();
        size_t uri_len = uri_string.length();

        if (uri_len >= 7 && std::equal(it,it+6,"wss://")) {
            m_secure = true;
            m_scheme = "wss";
            it += 6;
        } else if (uri_len >= 6 && std::equal(it,it+5,"ws://")) {
            m_secure = false;
            m_scheme = "ws";
            it += 5;
        } else if (uri_len >= 8 && std::equal(it,it+7,"http://")) {
            m_secure = false;
            m_scheme = "http";
            it += 7;
        } else if (uri_len >= 9 && std::equal(it,it+8,"https://")) {
            m_secure = true;
            m_scheme = "https";
            it += 8;
        } else {
            return;
        }

        // extract host.
        // either a host string
        // an IPv4 address
        // or an IPv6 address
        if (*it == '[') {
            ++it;
            // IPv6 literal
            // extract IPv6 digits until ]

            // TODO: this doesn't work on g++... not sure why
            //temp = std::find(it,it2,']');

            temp = it;
            while (temp != uri_string.end()) {
                if (*temp == ']') {
                    break;
                }
                ++temp;
            }

            if (temp == uri_string.end()) {
                return;
            } else {
                // validate IPv6 literal parts
                // can contain numbers, a-f and A-F
                m_host.append(it,temp);
            }
            it = temp+1;
            if (it == uri_string.end()) {
                state = 2;
            } else if (*it == '/') {
                state = 2;
                ++it;
            } else if (*it == ':') {
                state = 1;
                ++it;
            } else {
                // problem
                return;
            }
        } else {
            // IPv4 or hostname
            // extract until : or /
            while (state == 0) {
                if (it == uri_string.end()) {
                    state = 2;
                    break;
                } else if (*it == '/') {
                    state = 2;
                } else if (*it == ':') {
                    // end hostname start port
                    state = 1;
                } else {
                    m_host += *it;
                }
                ++it;
            }
        }

        // parse port
        std::string port;
        while (state == 1) {
            if (it == uri_string.end()) {
                // state is not used after this point presently.
                // this should be re-enabled if it ever is needed in a future
                // refactoring
                //state = 3;
                break;
            } else if (*it == '/') {
                state = 3;
            } else {
                port += *it;
            }
            ++it;
        }

        lib::error_code ec;
        m_port = get_port_from_string(port, ec);

        if (ec) {
            return;
        }

        m_resource = "/";
        m_resource.append(it,uri_string.end());


        m_valid = true;
    }

    uri(bool secure, std::string const & host, uint16_t port,
        std::string const & resource)
      : m_scheme(secure ? "wss" : "ws")
      , m_host(host)
      , m_resource(resource.empty() ? "/" : resource)
      , m_port(port)
      , m_secure(secure)
      , m_valid(true) {}

    uri(bool secure, std::string const & host, std::string const & resource)
      : m_scheme(secure ? "wss" : "ws")
      , m_host(host)
      , m_resource(resource.empty() ? "/" : resource)
      , m_port(secure ? uri_default_secure_port : uri_default_port)
      , m_secure(secure)
      , m_valid(true) {}

    uri(bool secure, std::string const & host, std::string const & port,
        std::string const & resource)
      : m_scheme(secure ? "wss" : "ws")
      , m_host(host)
      , m_resource(resource.empty() ? "/" : resource)
      , m_secure(secure)
    {
        lib::error_code ec;
        m_port = get_port_from_string(port,ec);
        m_valid = !ec;
    }

    uri(std::string const & scheme, std::string const & host, uint16_t port,
        std::string const & resource)
      : m_scheme(scheme)
      , m_host(host)
      , m_resource(resource.empty() ? "/" : resource)
      , m_port(port)
      , m_secure(scheme == "wss" || scheme == "https")
      , m_valid(true) {}

    uri(std::string scheme, std::string const & host, std::string const & resource)
      : m_scheme(scheme)
      , m_host(host)
      , m_resource(resource.empty() ? "/" : resource)
      , m_port((scheme == "wss" || scheme == "https") ? uri_default_secure_port : uri_default_port)
      , m_secure(scheme == "wss" || scheme == "https")
      , m_valid(true) {}

    uri(std::string const & scheme, std::string const & host,
        std::string const & port, std::string const & resource)
      : m_scheme(scheme)
      , m_host(host)
      , m_resource(resource.empty() ? "/" : resource)
      , m_secure(scheme == "wss" || scheme == "https")
    {
        lib::error_code ec;
        m_port = get_port_from_string(port,ec);
        m_valid = !ec;
    }

    bool get_valid() const {
        return m_valid;
    }

    bool get_secure() const {
        return m_secure;
    }

    std::string const & get_scheme() const {
        return m_scheme;
    }

    std::string const & get_host() const {
        return m_host;
    }

    std::string get_host_port() const {
        if (m_port == (m_secure ? uri_default_secure_port : uri_default_port)) {
            return m_host;
        } else {
            std::stringstream p;
            p << m_host << ":" << m_port;
            return p.str();
        }
    }

    std::string get_authority() const {
        std::stringstream p;
        p << m_host << ":" << m_port;
        return p.str();
    }

    uint16_t get_port() const {
        return m_port;
    }

    std::string get_port_str() const {
        std::stringstream p;
        p << m_port;
        return p.str();
    }

    std::string const & get_resource() const {
        return m_resource;
    }

    std::string str() const {
        std::stringstream s;

        s << m_scheme << "://" << m_host;

        if (m_port != (m_secure ? uri_default_secure_port : uri_default_port)) {
            s << ":" << m_port;
        }

        s << m_resource;
        return s.str();
    }

    /// Return the query portion
    /**
     * Returns the query portion (after the ?) of the URI or an empty string if
     * there is none.
     *
     * @return query portion of the URI.
     */
    std::string get_query() const {
        std::size_t found = m_resource.find('?');
        if (found != std::string::npos) {
            return m_resource.substr(found + 1);
        } else {
            return "";
        }
    }

    // get fragment

    // hi <3

    // get the string representation of this URI

    //std::string base() const; // is this still needed?

    // setter methods set some or all (in the case of parse) based on the input.
    // These functions throw a uri_exception on failure.
    /*void set_uri(const std::string& uri);

    void set_secure(bool secure);
    void set_host(const std::string& host);
    void set_port(uint16_t port);
    void set_port(const std::string& port);
    void set_resource(const std::string& resource);*/
private:
    uint16_t get_port_from_string(std::string const & port, lib::error_code &
        ec) const
    {
        ec = lib::error_code();

        if (port.empty()) {
            return (m_secure ? uri_default_secure_port : uri_default_port);
        }

        unsigned int t_port = static_cast<unsigned int>(atoi(port.c_str()));

        if (t_port > 65535) {
            ec = error::make_error_code(error::invalid_port);
        }

        if (t_port == 0) {
            ec = error::make_error_code(error::invalid_port);
        }

        return static_cast<uint16_t>(t_port);
    }

    std::string m_scheme;
    std::string m_host;
    std::string m_resource;
    uint16_t    m_port;
    bool        m_secure;
    bool        m_valid;
};

/// Pointer to a URI
typedef lib::shared_ptr<uri> uri_ptr;

} // namespace websocketpp

#endif // WEBSOCKETPP_URI_HPP
