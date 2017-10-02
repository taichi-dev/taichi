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

#ifndef WEBSOCKETPP_COMMON_FUNCTIONAL_HPP
#define WEBSOCKETPP_COMMON_FUNCTIONAL_HPP

#include <websocketpp/common/cpp11.hpp>

// If we've determined that we're in full C++11 mode and the user hasn't
// explicitly disabled the use of C++11 functional header, then prefer it to
// boost.
#if defined _WEBSOCKETPP_CPP11_INTERNAL_ && !defined _WEBSOCKETPP_NO_CPP11_FUNCTIONAL_
    #ifndef _WEBSOCKETPP_CPP11_FUNCTIONAL_
        #define _WEBSOCKETPP_CPP11_FUNCTIONAL_
    #endif
#endif

// If we're on Visual Studio 2010 or higher and haven't explicitly disabled
// the use of C++11 functional header then prefer it to boost.
#if defined(_MSC_VER) && _MSC_VER >= 1600 && !defined _WEBSOCKETPP_NO_CPP11_FUNCTIONAL_
    #ifndef _WEBSOCKETPP_CPP11_FUNCTIONAL_
        #define _WEBSOCKETPP_CPP11_FUNCTIONAL_
    #endif
#endif



#ifdef _WEBSOCKETPP_CPP11_FUNCTIONAL_
    #include <functional>
#else
    #include <boost/bind.hpp>
    #include <boost/function.hpp>
    #include <boost/ref.hpp>
#endif



namespace websocketpp {
namespace lib {

#ifdef _WEBSOCKETPP_CPP11_FUNCTIONAL_
    using std::function;
    using std::bind;
    using std::ref;
    namespace placeholders = std::placeholders;

    // There are some cases where a C++11 compiler balks at using std::ref
    // but a C++03 compiler using boost function requires boost::ref. As such
    // lib::ref is not useful in these cases. Instead this macro allows the use
    // of boost::ref in the case of a boost compile or no reference wrapper at
    // all in the case of a C++11 compile
    #define _WEBSOCKETPP_REF(x) x

    template <typename T>
    void clear_function(T & x) {
        x = nullptr;
    }
#else
    using boost::function;
    using boost::bind;
    using boost::ref;
    namespace placeholders {
        /// \todo this feels hacky, is there a better way?
        using ::_1;
        using ::_2;
        using ::_3;
    }

    // See above definition for more details on what this is and why it exists
    #define _WEBSOCKETPP_REF(x) boost::ref(x)

    template <typename T>
    void clear_function(T & x) {
        x.clear();
    }
#endif

} // namespace lib
} // namespace websocketpp

#endif // WEBSOCKETPP_COMMON_FUNCTIONAL_HPP
