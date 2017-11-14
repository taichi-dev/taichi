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

#ifndef __TBB_null_rw_mutex_H
#define __TBB_null_rw_mutex_H

#include "tbb_stddef.h"

namespace tbb {
    
//! A rw mutex which does nothing
/** A null_rw_mutex is a rw mutex that does nothing and simulates successful operation.
    @ingroup synchronization */
class null_rw_mutex : internal::mutex_copy_deprecated_and_disabled {
public:   
    //! Represents acquisition of a mutex.
    class scoped_lock : internal::no_copy {   
    public:   
        scoped_lock() {}
        scoped_lock( null_rw_mutex& , bool = true ) {}
        ~scoped_lock() {}
        void acquire( null_rw_mutex& , bool = true ) {}
        bool upgrade_to_writer() { return true; }
        bool downgrade_to_reader() { return true; }
        bool try_acquire( null_rw_mutex& , bool = true ) { return true; }
        void release() {}
    };
  
    null_rw_mutex() {}
    
    // Mutex traits   
    static const bool is_rw_mutex = true;   
    static const bool is_recursive_mutex = true;
    static const bool is_fair_mutex = true;
};  

}

#endif /* __TBB_null_rw_mutex_H */
