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

#ifndef WEBSOCKETPP_COMMON_STDINT_HPP
#define WEBSOCKETPP_COMMON_STDINT_HPP

#ifndef __STDC_LIMIT_MACROS
    #define __STDC_LIMIT_MACROS 1
#endif

#if defined (_WIN32) && defined (_MSC_VER) && (_MSC_VER < 1600)
    #include <boost/cstdint.hpp>

    using boost::int8_t;
    using boost::int_least8_t;
    using boost::int_fast8_t;
    using boost::uint8_t;
    using boost::uint_least8_t;
    using boost::uint_fast8_t;

    using boost::int16_t;
    using boost::int_least16_t;
    using boost::int_fast16_t;
    using boost::uint16_t;
    using boost::uint_least16_t;
    using boost::uint_fast16_t;

    using boost::int32_t;
    using boost::int_least32_t;
    using boost::int_fast32_t;
    using boost::uint32_t;
    using boost::uint_least32_t;
    using boost::uint_fast32_t;

    #ifndef BOOST_NO_INT64_T
    using boost::int64_t;
    using boost::int_least64_t;
    using boost::int_fast64_t;
    using boost::uint64_t;
    using boost::uint_least64_t;
    using boost::uint_fast64_t;
    #endif
    using boost::intmax_t;
    using boost::uintmax_t;
#else
    #include <stdint.h>
#endif

#endif // WEBSOCKETPP_COMMON_STDINT_HPP
