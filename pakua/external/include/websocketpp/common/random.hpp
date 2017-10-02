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

#ifndef WEBSOCKETPP_COMMON_RANDOM_DEVICE_HPP
#define WEBSOCKETPP_COMMON_RANDOM_DEVICE_HPP

#include <websocketpp/common/cpp11.hpp>

// If we've determined that we're in full C++11 mode and the user hasn't
// explicitly disabled the use of C++11 random header, then prefer it to
// boost.
#if defined _WEBSOCKETPP_CPP11_INTERNAL_ && !defined _WEBSOCKETPP_NO_CPP11_RANDOM_DEVICE_
    #ifndef _WEBSOCKETPP_CPP11_RANDOM_DEVICE_
        #define _WEBSOCKETPP_CPP11_RANDOM_DEVICE_
    #endif
#endif


// If we're on Visual Studio 2010 or higher and haven't explicitly disabled
// the use of C++11 random header then prefer it to boost.
#if defined(_MSC_VER) && _MSC_VER >= 1600 && !defined _WEBSOCKETPP_NO_CPP11_MEMORY_
    #ifndef _WEBSOCKETPP_CPP11_MEMORY_
        #define _WEBSOCKETPP_CPP11_MEMORY_
    #endif
#endif



#ifdef _WEBSOCKETPP_CPP11_RANDOM_DEVICE_
    #include <random>
#else
    #include <boost/version.hpp>

    #if (BOOST_VERSION/100000) == 1 && ((BOOST_VERSION/100)%1000) > 46
        #include <boost/random/uniform_int_distribution.hpp>
        #include <boost/random/random_device.hpp>
    #elif (BOOST_VERSION/100000) == 1 && ((BOOST_VERSION/100)%1000) >= 43
        #include <boost/nondet_random.hpp>
    #else
        // TODO: static_assert(false, "Could not find a suitable random_device")
    #endif
#endif

namespace websocketpp {
namespace lib {

#ifdef _WEBSOCKETPP_CPP11_RANDOM_DEVICE_
    using std::random_device;
    using std::uniform_int_distribution;
#else
    using boost::random::random_device;
    using boost::random::uniform_int_distribution;
#endif

} // namespace lib
} // namespace websocketpp

#endif // WEBSOCKETPP_COMMON_RANDOM_DEVICE_HPP
