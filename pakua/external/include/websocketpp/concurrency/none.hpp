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

#ifndef WEBSOCKETPP_CONCURRENCY_NONE_HPP
#define WEBSOCKETPP_CONCURRENCY_NONE_HPP

namespace websocketpp {

/// Concurrency handling support
namespace concurrency {

/// Implementation for no-op locking primitives
namespace none_impl {
/// A fake mutex implementation that does nothing
class fake_mutex {
public:
    fake_mutex() {}
    ~fake_mutex() {}
};

/// A fake lock guard implementation that does nothing
class fake_lock_guard {
public:
    explicit fake_lock_guard(fake_mutex) {}
    ~fake_lock_guard() {}
};
} // namespace none_impl

/// Stub concurrency policy that implements the interface using no-ops.
/**
 * This policy documents the concurrency policy interface using no-ops. It can
 * be used as a reference or base for building a new concurrency policy. It can
 * also be used as is to disable all locking for endpoints used in purely single
 * threaded programs.
 */
class none {
public:
    /// The type of a mutex primitive
    /**
     * std::mutex is an example.
     */
    typedef none_impl::fake_mutex mutex_type;

    /// The type of a scoped/RAII lock primitive.
    /**
     * The scoped lock constructor should take a mutex_type as a parameter,
     * acquire that lock, and release it in its destructor. std::lock_guard is
     * an example.
     */
    typedef none_impl::fake_lock_guard scoped_lock_type;
};

} // namespace concurrency
} // namespace websocketpp

#endif // WEBSOCKETPP_CONCURRENCY_ASYNC_HPP
