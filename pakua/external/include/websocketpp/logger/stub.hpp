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

#ifndef WEBSOCKETPP_LOGGER_STUB_HPP
#define WEBSOCKETPP_LOGGER_STUB_HPP

#include <websocketpp/logger/levels.hpp>

#include <websocketpp/common/cpp11.hpp>

#include <string>

namespace websocketpp {
namespace log {

/// Stub logger that ignores all input
class stub {
public:
    /// Construct the logger
    /**
     * @param hint A channel type specific hint for how to construct the logger
     */
    explicit stub(channel_type_hint::value) {}

    /// Construct the logger
    /**
     * @param default_channels A set of channels to statically enable
     * @param hint A channel type specific hint for how to construct the logger
     */
    stub(level, channel_type_hint::value) {}
    _WEBSOCKETPP_CONSTEXPR_TOKEN_ stub() {}

    /// Dynamically enable the given list of channels
    /**
     * All operations on the stub logger are no-ops and all arguments are
     * ignored
     *
     * @param channels The package of channels to enable
     */
    void set_channels(level) {}

    /// Dynamically disable the given list of channels
    /**
     * All operations on the stub logger are no-ops and all arguments are
     * ignored
     *
     * @param channels The package of channels to disable
     */
    void clear_channels(level) {}

    /// Write a string message to the given channel
    /**
     * Writing on the stub logger is a no-op and all arguments are ignored
     *
     * @param channel The channel to write to
     * @param msg The message to write
     */
    void write(level, std::string const &) {}

    /// Write a cstring message to the given channel
    /**
     * Writing on the stub logger is a no-op and all arguments are ignored
     *
     * @param channel The channel to write to
     * @param msg The message to write
     */
    void write(level, char const *) {}

    /// Test whether a channel is statically enabled
    /**
     * The stub logger has no channels so all arguments are ignored and
     * `static_test` always returns false.
     *
     * @param channel The package of channels to test
     */
    _WEBSOCKETPP_CONSTEXPR_TOKEN_ bool static_test(level) const {
        return false;
    }

    /// Test whether a channel is dynamically enabled
    /**
     * The stub logger has no channels so all arguments are ignored and
     * `dynamic_test` always returns false.
     *
     * @param channel The package of channels to test
     */
    bool dynamic_test(level) {
        return false;
    }
};

} // log
} // websocketpp

#endif // WEBSOCKETPP_LOGGER_STUB_HPP
