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

 // This header defines WebSocket++ macros for C++11 compatibility based on the 
 // Boost.Config library. This will correctly configure most target platforms
 // simply by including this header before any other WebSocket++ header.

#ifndef WEBSOCKETPP_CONFIG_BOOST_CONFIG_HPP
#define WEBSOCKETPP_CONFIG_BOOST_CONFIG_HPP

#include <boost/config.hpp>

//  _WEBSOCKETPP_CPP11_MEMORY_ and _WEBSOCKETPP_CPP11_FUNCTIONAL_ presently
//  only work if either both or neither is defined.
#if !defined BOOST_NO_CXX11_SMART_PTR && !defined BOOST_NO_CXX11_HDR_FUNCTIONAL
    #define _WEBSOCKETPP_CPP11_MEMORY_
    #define _WEBSOCKETPP_CPP11_FUNCTIONAL_
#endif

#ifdef BOOST_ASIO_HAS_STD_CHRONO
    #define _WEBSOCKETPP_CPP11_CHRONO_
#endif

#ifndef BOOST_NO_CXX11_HDR_RANDOM
    #define _WEBSOCKETPP_CPP11_RANDOM_DEVICE_
#endif

#ifndef BOOST_NO_CXX11_HDR_REGEX
    #define _WEBSOCKETPP_CPP11_REGEX_
#endif

#ifndef BOOST_NO_CXX11_HDR_SYSTEM_ERROR
    #define _WEBSOCKETPP_CPP11_SYSTEM_ERROR_
#endif

#ifndef BOOST_NO_CXX11_HDR_THREAD
    #define _WEBSOCKETPP_CPP11_THREAD_
#endif

#ifndef BOOST_NO_CXX11_HDR_INITIALIZER_LIST
    #define _WEBSOCKETPP_INITIALIZER_LISTS_
#endif

#define _WEBSOCKETPP_NOEXCEPT_TOKEN_  BOOST_NOEXCEPT
#define _WEBSOCKETPP_CONSTEXPR_TOKEN_  BOOST_CONSTEXPR
// TODO: nullptr support

#endif // WEBSOCKETPP_CONFIG_BOOST_CONFIG_HPP
