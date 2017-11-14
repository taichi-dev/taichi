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

#ifndef _TBB_CRITICAL_SECTION_H_
#define _TBB_CRITICAL_SECTION_H_

#if _WIN32||_WIN64
#include "machine/windows_api.h"
#else
#include <pthread.h>
#include <errno.h>
#endif  // _WIN32||WIN64

#include "tbb_stddef.h"
#include "tbb_thread.h"
#include "tbb_exception.h"

#include "tbb_profiling.h"

namespace tbb {

    namespace internal {
class critical_section_v4 : internal::no_copy {
#if _WIN32||_WIN64
    CRITICAL_SECTION my_impl;
#else
    pthread_mutex_t my_impl;
#endif
    tbb_thread::id my_tid;
public:

    void __TBB_EXPORTED_METHOD internal_construct();

    critical_section_v4() { 
#if _WIN32||_WIN64
        InitializeCriticalSectionEx( &my_impl, 4000, 0 );
#else
        pthread_mutex_init(&my_impl, NULL);
#endif
        internal_construct();
    }

    ~critical_section_v4() {
        __TBB_ASSERT(my_tid == tbb_thread::id(), "Destroying a still-held critical section");
#if _WIN32||_WIN64
        DeleteCriticalSection(&my_impl); 
#else
        pthread_mutex_destroy(&my_impl);
#endif
    }

    class scoped_lock : internal::no_copy {
    private:
        critical_section_v4 &my_crit;
    public:
        scoped_lock( critical_section_v4& lock_me) :my_crit(lock_me) {
            my_crit.lock();
        }

        ~scoped_lock() {
            my_crit.unlock();
        }
    };

    void lock() { 
        tbb_thread::id local_tid = this_tbb_thread::get_id();
        if(local_tid == my_tid) throw_exception( eid_improper_lock );
#if _WIN32||_WIN64
        EnterCriticalSection( &my_impl );
#else
        int rval = pthread_mutex_lock(&my_impl);
        __TBB_ASSERT_EX(!rval, "critical_section::lock: pthread_mutex_lock failed");
#endif
        __TBB_ASSERT(my_tid == tbb_thread::id(), NULL);
        my_tid = local_tid;
    }

    bool try_lock() {
        bool gotlock;
        tbb_thread::id local_tid = this_tbb_thread::get_id();
        if(local_tid == my_tid) return false;
#if _WIN32||_WIN64
        gotlock = TryEnterCriticalSection( &my_impl ) != 0;
#else
        int rval = pthread_mutex_trylock(&my_impl);
        // valid returns are 0 (locked) and [EBUSY]
        __TBB_ASSERT(rval == 0 || rval == EBUSY, "critical_section::trylock: pthread_mutex_trylock failed");
        gotlock = rval == 0;
#endif
        if(gotlock)  {
            my_tid = local_tid;
        }
        return gotlock;
    }

    void unlock() {
        __TBB_ASSERT(this_tbb_thread::get_id() == my_tid, "thread unlocking critical_section is not thread that locked it");
        my_tid = tbb_thread::id();
#if _WIN32||_WIN64
        LeaveCriticalSection( &my_impl );
#else
        int rval = pthread_mutex_unlock(&my_impl);
        __TBB_ASSERT_EX(!rval, "critical_section::unlock: pthread_mutex_unlock failed");
#endif
    }

    static const bool is_rw_mutex = false;
    static const bool is_recursive_mutex = false;
    static const bool is_fair_mutex = true;
}; // critical_section_v4
} // namespace internal
typedef internal::critical_section_v4 critical_section;

__TBB_DEFINE_PROFILING_SET_NAME(critical_section)
} // namespace tbb
#endif  // _TBB_CRITICAL_SECTION_H_
