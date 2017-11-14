/*
    Copyright 2005-2015 Intel Corporation.  All Rights Reserved.

    This file is part of Threading Building Blocks. Threading Building Blocks is free software;
    you can redistribute it and/or modify it under the terms of the GNU General Public License
    version 2  as  published  by  the  Free Software Foundation.  Threading Building Blocks is
    distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
    implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
    See  the GNU General Public License for more details.   You should have received a copy of
    the  GNU General Public License along with Threading Building Blocks; if not, write to the
    Free Software Foundation, Inc.,  51 Franklin St,  Fifth Floor,  Boston,  MA 02110-1301 USA

    As a special exception,  you may use this file  as part of a free software library without
    restriction.  Specifically,  if other files instantiate templates  or use macros or inline
    functions from this file, or you compile this file and link it with other files to produce
    an executable,  this file does not by itself cause the resulting executable to be covered
    by the GNU General Public License. This exception does not however invalidate any other
    reasons why the executable file might be covered by the GNU General Public License.
*/

#ifndef __TBB_mutex_padding_H
#define __TBB_mutex_padding_H

// wrapper for padding mutexes to be alone on a cache line, without requiring they be allocated
// from a pool.  Because we allow them to be defined anywhere they must be two cache lines in size.


namespace tbb {
namespace interface7 {
namespace internal {

static const size_t cache_line_size = 64;

// Pad a mutex to occupy a number of full cache lines sufficient to avoid false sharing
// with other data; space overhead is up to 2*cache_line_size-1.
template<typename Mutex, bool is_rw> class padded_mutex;

template<typename Mutex>
class padded_mutex<Mutex,false> : tbb::internal::mutex_copy_deprecated_and_disabled {
    typedef long pad_type;
    pad_type my_pad[((sizeof(Mutex)+cache_line_size-1)/cache_line_size+1)*cache_line_size/sizeof(pad_type)];

    Mutex *impl() { return (Mutex *)((uintptr_t(this)|(cache_line_size-1))+1);}

public:
    static const bool is_rw_mutex = Mutex::is_rw_mutex;
    static const bool is_recursive_mutex = Mutex::is_recursive_mutex;
    static const bool is_fair_mutex = Mutex::is_fair_mutex;

    padded_mutex() { new(impl()) Mutex(); }
    ~padded_mutex() { impl()->~Mutex(); }

    //! Represents acquisition of a mutex.
    class scoped_lock :  tbb::internal::no_copy {
        typename Mutex::scoped_lock my_scoped_lock;
    public:
        scoped_lock() : my_scoped_lock() {}
        scoped_lock( padded_mutex& m ) : my_scoped_lock(*m.impl()) { }
        ~scoped_lock() {  }

        void acquire( padded_mutex& m ) { my_scoped_lock.acquire(*m.impl()); }
        bool try_acquire( padded_mutex& m ) { return my_scoped_lock.try_acquire(*m.impl()); }
        void release() { my_scoped_lock.release(); }
    };
};

template<typename Mutex>
class padded_mutex<Mutex,true> : tbb::internal::mutex_copy_deprecated_and_disabled {
    typedef long pad_type;
    pad_type my_pad[((sizeof(Mutex)+cache_line_size-1)/cache_line_size+1)*cache_line_size/sizeof(pad_type)];

    Mutex *impl() { return (Mutex *)((uintptr_t(this)|(cache_line_size-1))+1);}

public:
    static const bool is_rw_mutex = Mutex::is_rw_mutex;
    static const bool is_recursive_mutex = Mutex::is_recursive_mutex;
    static const bool is_fair_mutex = Mutex::is_fair_mutex;

    padded_mutex() { new(impl()) Mutex(); }
    ~padded_mutex() { impl()->~Mutex(); }

    //! Represents acquisition of a mutex.
    class scoped_lock :  tbb::internal::no_copy {
        typename Mutex::scoped_lock my_scoped_lock;
    public:
        scoped_lock() : my_scoped_lock() {}
        scoped_lock( padded_mutex& m, bool write = true ) : my_scoped_lock(*m.impl(),write) { }
        ~scoped_lock() {  }

        void acquire( padded_mutex& m, bool write = true ) { my_scoped_lock.acquire(*m.impl(),write); }
        bool try_acquire( padded_mutex& m, bool write = true ) { return my_scoped_lock.try_acquire(*m.impl(),write); }
        bool upgrade_to_writer() { return my_scoped_lock.upgrade_to_writer(); }
        bool downgrade_to_reader() { return my_scoped_lock.downgrade_to_reader(); }
        void release() { my_scoped_lock.release(); }
    };
};

} // namespace internal
} // namespace interface7
} // namespace tbb

#endif /* __TBB_mutex_padding_H */
