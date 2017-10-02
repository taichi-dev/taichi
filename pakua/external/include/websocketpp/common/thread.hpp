/*
 * Copyright (c) 2015, Peter Thorson. All rights reserved.
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

#ifndef WEBSOCKETPP_COMMON_THREAD_HPP
#define WEBSOCKETPP_COMMON_THREAD_HPP

#include <websocketpp/common/cpp11.hpp>

// If we autodetect C++11 and haven't been explicitly instructed to not use
// C++11 threads, then set the defines that instructs the rest of this header
// to use C++11 <thread> and <mutex>
#if defined _WEBSOCKETPP_CPP11_INTERNAL_ && !defined _WEBSOCKETPP_NO_CPP11_THREAD_
    // MinGW by default does not support C++11 thread/mutex so even if the
    // internal check for C++11 passes, ignore it if we are on MinGW
    #if (!defined(__MINGW32__) && !defined(__MINGW64__))
        #ifndef _WEBSOCKETPP_CPP11_THREAD_
            #define _WEBSOCKETPP_CPP11_THREAD_
        #endif
    #endif
#endif

// If we're on Visual Studio 2013 or higher and haven't explicitly disabled
// the use of C++11 thread header then prefer it to boost.
#if defined(_MSC_VER) && _MSC_VER >= 1800 && !defined _WEBSOCKETPP_NO_CPP11_THREAD_
    #ifndef _WEBSOCKETPP_CPP11_THREAD_
        #define _WEBSOCKETPP_CPP11_THREAD_
    #endif
#endif

#ifdef _WEBSOCKETPP_CPP11_THREAD_
    #include <thread>
    #include <mutex>
    #include <condition_variable>
#else
    #include <boost/thread.hpp>
    #include <boost/thread/mutex.hpp>
    #include <boost/thread/condition_variable.hpp>
#endif

namespace websocketpp {
namespace lib {

#ifdef _WEBSOCKETPP_CPP11_THREAD_
    using std::mutex;
    using std::lock_guard;
    using std::thread;
    using std::unique_lock;
    using std::condition_variable;
#else
    using boost::mutex;
    using boost::lock_guard;
    using boost::thread;
    using boost::unique_lock;
    using boost::condition_variable;
#endif

} // namespace lib
} // namespace websocketpp

#endif // WEBSOCKETPP_COMMON_THREAD_HPP
