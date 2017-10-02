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

#ifndef WEBSOCKETPP_CONFIG_MINIMAL_CLIENT_HPP
#define WEBSOCKETPP_CONFIG_MINIMAL_CLIENT_HPP

#include <websocketpp/config/minimal_server.hpp>

namespace websocketpp {
namespace config {

/// Client config with minimal dependencies
/**
 * This config strips out as many dependencies as possible. It is suitable for
 * use as a base class for custom configs that want to implement or choose their
 * own policies for components that even the core config includes.
 *
 * NOTE: this config stubs out enough that it cannot be used directly. You must
 * supply at least a transport policy and a cryptographically secure random 
 * number generation policy for a config based on `minimal_client` to do 
 * anything useful.
 *
 * Present dependency list for minimal_server config:
 *
 * C++98 STL:
 * <algorithm>
 * <map>
 * <sstream>
 * <string>
 * <vector>
 *
 * C++11 STL or Boost
 * <memory>
 * <functional>
 * <system_error>
 *
 * Operating System:
 * <stdint.h> or <boost/cstdint.hpp>
 * <netinet/in.h> or <winsock2.h> (for ntohl.. could potentially bundle this)
 *
 * @since 0.4.0-dev
 */
typedef minimal_server minimal_client;

} // namespace config
} // namespace websocketpp

#endif // WEBSOCKETPP_CONFIG_MINIMAL_CLIENT_HPP
