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

#ifndef __TBB_global_control_H
#define __TBB_global_control_H

#if !TBB_PREVIEW_GLOBAL_CONTROL && !__TBB_BUILD
#error Set TBB_PREVIEW_GLOBAL_CONTROL before including global_control.h
#endif

#include "tbb_stddef.h"

namespace tbb {
namespace interface9 {

class global_control {
public:
    enum parameter {
        max_allowed_parallelism,
        thread_stack_size,
        parameter_max // insert new parameters above this point
    };

    global_control(parameter p, size_t value) :
        my_value(value), my_next(NULL), my_param(p) {
        __TBB_ASSERT(my_param < parameter_max, "Invalid parameter");
#if __TBB_WIN8UI_SUPPORT
        // For Windows Store* apps it's impossible to set stack size
        if (p==thread_stack_size)
            return;
#elif __TBB_x86_64 && (_WIN32 || _WIN64)
        if (p==thread_stack_size)
            __TBB_ASSERT_RELEASE((unsigned)value == value, "Stack size is limited to unsigned int range");
#endif
        if (my_param==max_allowed_parallelism)
            // TODO: support for serialization via max_allowed_parallelism==1
            __TBB_ASSERT_RELEASE(my_value>1, "Values of 1 and 0 are not supported for max_allowed_parallelism.");
        internal_create();
    }

    ~global_control() {
        __TBB_ASSERT(my_param < parameter_max, "Invalid parameter. Probably the object was corrupted.");
#if __TBB_WIN8UI_SUPPORT
        // For Windows Store* apps it's impossible to set stack size
        if (my_param==thread_stack_size)
            return;
#endif
        internal_destroy();
    }

    static size_t active_value(parameter p) {
        __TBB_ASSERT(p < parameter_max, "Invalid parameter");
        return active_value((int)p);
    }
private:
    size_t    my_value;
    global_control *my_next;
    parameter my_param;

    void __TBB_EXPORTED_METHOD internal_create();
    void __TBB_EXPORTED_METHOD internal_destroy();
    static size_t __TBB_EXPORTED_FUNC active_value(int param);
};
} // namespace interface9

using interface9::global_control;

} // tbb

#endif // __TBB_global_control_H
