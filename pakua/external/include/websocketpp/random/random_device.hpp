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

#ifndef WEBSOCKETPP_RANDOM_RANDOM_DEVICE_HPP
#define WEBSOCKETPP_RANDOM_RANDOM_DEVICE_HPP

#include <websocketpp/common/random.hpp>

namespace websocketpp {
namespace random {
/// RNG policy based on std::random_device or boost::random_device
namespace random_device {

/// Thread safe non-deterministic random integer generator.
/**
 * This template class provides thread safe non-deterministic random integer
 * generation. Numbers are produced in a uniformly distributed range from the
 * smallest to largest value that int_type can store.
 *
 * Thread-safety is provided via locking based on the concurrency template
 * parameter.
 *
 * Non-deterministic RNG is provided via websocketpp::lib which uses either
 * C++11 or Boost 1.47+'s random_device class.
 *
 * Call operator() to generate the next number
 */
template <typename int_type, typename concurrency>
class int_generator {
    public:
        typedef typename concurrency::scoped_lock_type scoped_lock_type;
        typedef typename concurrency::mutex_type mutex_type;

        /// constructor
        //mac TODO: figure out if signed types present a range problem
        int_generator() {}

        /// advances the engine's state and returns the generated value
        int_type operator()() {
            scoped_lock_type guard(m_lock);
            return m_dis(m_rng);
        }
    private:


        lib::random_device m_rng;
        lib::uniform_int_distribution<int_type> m_dis;

        mutex_type m_lock;
};

} // namespace random_device
} // namespace random
} // namespace websocketpp

#endif //WEBSOCKETPP_RANDOM_RANDOM_DEVICE_HPP
