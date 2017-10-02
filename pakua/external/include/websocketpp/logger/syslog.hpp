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
 *
 * The initial version of this logging policy was contributed to the WebSocket++
 * project by Tom Hughes.
 */

#ifndef WEBSOCKETPP_LOGGER_SYSLOG_HPP
#define WEBSOCKETPP_LOGGER_SYSLOG_HPP

#include <syslog.h>

#include <websocketpp/logger/basic.hpp>

#include <websocketpp/common/cpp11.hpp>
#include <websocketpp/logger/levels.hpp>

namespace websocketpp {
namespace log {

/// Basic logger that outputs to syslog
template <typename concurrency, typename names>
class syslog : public basic<concurrency, names> {
public:
    typedef basic<concurrency, names> base;

    /// Construct the logger
    /**
     * @param hint A channel type specific hint for how to construct the logger
     */
    syslog<concurrency,names>(channel_type_hint::value hint =
        channel_type_hint::access)
      : basic<concurrency,names>(hint), m_channel_type_hint(hint) {}

    /// Construct the logger
    /**
     * @param channels A set of channels to statically enable
     * @param hint A channel type specific hint for how to construct the logger
     */
    syslog<concurrency,names>(level channels, channel_type_hint::value hint =
        channel_type_hint::access)
      : basic<concurrency,names>(channels, hint), m_channel_type_hint(hint) {}

    /// Write a string message to the given channel
    /**
     * @param channel The channel to write to
     * @param msg The message to write
     */
    void write(level channel, std::string const & msg) {
        write(channel, msg.c_str());
    }

    /// Write a cstring message to the given channel
    /**
     * @param channel The channel to write to
     * @param msg The message to write
     */
    void write(level channel, char const * msg) {
        scoped_lock_type lock(base::m_lock);
        if (!this->dynamic_test(channel)) { return; }
        ::syslog(syslog_priority(channel), "[%s] %s",
            names::channel_name(channel), msg);
    }
private:
    typedef typename base::scoped_lock_type scoped_lock_type;

    /// The default level is used for all access logs and any error logs that
    /// don't trivially map to one of the standard syslog levels.
    static int const default_level = LOG_INFO;

    /// retrieve the syslog priority code given a WebSocket++ channel
    /**
     * @param channel The level to look up
     * @return The syslog level associated with `channel`
     */
    int syslog_priority(level channel) const {
        if (m_channel_type_hint == channel_type_hint::access) {
            return syslog_priority_access(channel);
        } else {
            return syslog_priority_error(channel);
        }
    }

    /// retrieve the syslog priority code given a WebSocket++ error channel
    /**
     * @param channel The level to look up
     * @return The syslog level associated with `channel`
     */
    int syslog_priority_error(level channel) const {
        switch (channel) {
            case elevel::devel:
                return LOG_DEBUG;
            case elevel::library:
                return LOG_DEBUG;
            case elevel::info:
                return LOG_INFO;
            case elevel::warn:
                return LOG_WARNING;
            case elevel::rerror:
                return LOG_ERR;
            case elevel::fatal:
                return LOG_CRIT;
            default:
                return default_level;
        }
    }

    /// retrieve the syslog priority code given a WebSocket++ access channel
    /**
     * @param channel The level to look up
     * @return The syslog level associated with `channel`
     */
    _WEBSOCKETPP_CONSTEXPR_TOKEN_ int syslog_priority_access(level) const {
        return default_level;
    }

    channel_type_hint::value m_channel_type_hint;
};

} // log
} // websocketpp

#endif // WEBSOCKETPP_LOGGER_SYSLOG_HPP
