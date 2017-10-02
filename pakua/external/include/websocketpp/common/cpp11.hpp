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

#ifndef WEBSOCKETPP_COMMON_CPP11_HPP
#define WEBSOCKETPP_COMMON_CPP11_HPP

/**
 * This header sets up some constants based on the state of C++11 support
 */

// Hide clang feature detection from other compilers
#ifndef __has_feature         // Optional of course.
  #define __has_feature(x) 0  // Compatibility with non-clang compilers.
#endif
#ifndef __has_extension
  #define __has_extension __has_feature // Compatibility with pre-3.0 compilers.
#endif

// The code below attempts to use information provided by the build system or
// user supplied defines to selectively enable C++11 language and library
// features. In most cases features that are targeted individually may also be
// selectively disabled via an associated _WEBSOCKETPP_NOXXX_ define.

#if defined(_WEBSOCKETPP_CPP11_STL_) || __cplusplus >= 201103L || defined(_WEBSOCKETPP_CPP11_STRICT_)
    // This check tests for blanket c++11 coverage. It can be activated in one
    // of three ways. Either the compiler itself reports that it is a full 
    // C++11 compiler via the __cplusplus macro or the user/build system
    // supplies one of the two preprocessor defines below:
    
    // This is defined to allow other WebSocket++ common headers to enable
    // C++11 features when they are detected by this file rather than
    // duplicating the above logic in every common header.
    #define _WEBSOCKETPP_CPP11_INTERNAL_
    
    // _WEBSOCKETPP_CPP11_STRICT_
    //
    // This define reports to WebSocket++ that 100% of the language and library
    // features of C++11 are available. Using this define on a non-C++11
    // compiler will result in problems.
    
    // _WEBSOCKETPP_CPP11_STL_ 
    //
    // This define enables *most* C++11 options that were implemented early on
    // by compilers. It is typically used for compilers that have many, but not
    // all C++11 features. It should be safe to use on GCC 4.7-4.8 and perhaps
    // earlier. 
    #ifndef _WEBSOCKETPP_NOEXCEPT_TOKEN_
        #define _WEBSOCKETPP_NOEXCEPT_TOKEN_ noexcept
    #endif
    #ifndef _WEBSOCKETPP_CONSTEXPR_TOKEN_
        #define _WEBSOCKETPP_CONSTEXPR_TOKEN_ constexpr
    #endif
    #ifndef _WEBSOCKETPP_INITIALIZER_LISTS_
        #define _WEBSOCKETPP_INITIALIZER_LISTS_
    #endif
    #ifndef _WEBSOCKETPP_NULLPTR_TOKEN_
        #define _WEBSOCKETPP_NULLPTR_TOKEN_ nullptr
    #endif
    #ifndef _WEBSOCKETPP_MOVE_SEMANTICS_
        #define _WEBSOCKETPP_MOVE_SEMANTICS_
    #endif
    #ifndef _WEBSOCKETPP_DEFAULT_DELETE_FUNCTIONS_
        #define _WEBSOCKETPP_DEFAULT_DELETE_FUNCTIONS_
    #endif
    
    #ifndef __GNUC__
        // GCC as of version 4.9 (latest) does not support std::put_time yet.
        // so ignore it
        #define _WEBSOCKETPP_PUTTIME_
    #endif
#else
    // In the absence of a blanket define, try to use compiler versions or
    // feature testing macros to selectively enable what we can.

    // Test for noexcept
    #ifndef _WEBSOCKETPP_NOEXCEPT_TOKEN_
        #ifdef _WEBSOCKETPP_NOEXCEPT_
            // build system says we have noexcept
            #define _WEBSOCKETPP_NOEXCEPT_TOKEN_ noexcept
        #else
            #if __has_feature(cxx_noexcept)
                // clang feature detect says we have noexcept
                #define _WEBSOCKETPP_NOEXCEPT_TOKEN_ noexcept
            #elif defined(_MSC_VER) && _MSC_VER >= 1900
            	// Visual Studio 2015+ has noexcept
                #define _WEBSOCKETPP_NOEXCEPT_TOKEN_ noexcept
            #else
                // assume we don't have noexcept
                #define _WEBSOCKETPP_NOEXCEPT_TOKEN_
            #endif
        #endif
    #endif

    // Test for constexpr
    #ifndef _WEBSOCKETPP_CONSTEXPR_TOKEN_
        #ifdef _WEBSOCKETPP_CONSTEXPR_
            // build system says we have constexpr
            #define _WEBSOCKETPP_CONSTEXPR_TOKEN_ constexpr
        #else
            #if __has_feature(cxx_constexpr)
                // clang feature detect says we have constexpr
                #define _WEBSOCKETPP_CONSTEXPR_TOKEN_ constexpr
            #elif defined(_MSC_VER) && _MSC_VER >= 1900
            	// Visual Studio 2015+ has constexpr
                #define _WEBSOCKETPP_CONSTEXPR_TOKEN_ constexpr
            #else
                // assume we don't have constexpr
                #define _WEBSOCKETPP_CONSTEXPR_TOKEN_
            #endif
        #endif
    #endif

    // Enable initializer lists on clang when available.
    #if __has_feature(cxx_generalized_initializers) && !defined(_WEBSOCKETPP_INITIALIZER_LISTS_)
        #define _WEBSOCKETPP_INITIALIZER_LISTS_
    #endif
    
    // Test for nullptr
    #ifndef _WEBSOCKETPP_NULLPTR_TOKEN_
        #ifdef _WEBSOCKETPP_NULLPTR_
            // build system says we have nullptr
            #define _WEBSOCKETPP_NULLPTR_TOKEN_ nullptr
        #else
            #if __has_feature(cxx_nullptr)
                // clang feature detect says we have nullptr
                #define _WEBSOCKETPP_NULLPTR_TOKEN_ nullptr
            #elif defined(_MSC_VER) &&_MSC_VER >= 1600
                // Visual Studio version that has nullptr
                #define _WEBSOCKETPP_NULLPTR_TOKEN_ nullptr
            #else
                // assume we don't have nullptr
                #define _WEBSOCKETPP_NULLPTR_TOKEN_ 0
            #endif
        #endif
    #endif
#endif

#endif // WEBSOCKETPP_COMMON_CPP11_HPP
