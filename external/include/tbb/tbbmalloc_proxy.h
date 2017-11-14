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

/*
Replacing the standard memory allocation routines in Microsoft* C/C++ RTL 
(malloc/free, global new/delete, etc.) with the TBB memory allocator. 

Include the following header to a source of any binary which is loaded during 
application startup

#include "tbb/tbbmalloc_proxy.h"

or add following parameters to the linker options for the binary which is 
loaded during application startup. It can be either exe-file or dll.

For win32
tbbmalloc_proxy.lib /INCLUDE:"___TBB_malloc_proxy"
win64
tbbmalloc_proxy.lib /INCLUDE:"__TBB_malloc_proxy"
*/

#ifndef __TBB_tbbmalloc_proxy_H
#define __TBB_tbbmalloc_proxy_H

#if _MSC_VER

#ifdef _DEBUG
    #pragma comment(lib, "tbbmalloc_proxy_debug.lib")
#else
    #pragma comment(lib, "tbbmalloc_proxy.lib")
#endif

#if defined(_WIN64)
    #pragma comment(linker, "/include:__TBB_malloc_proxy")
#else
    #pragma comment(linker, "/include:___TBB_malloc_proxy")
#endif

#else
/* Primarily to support MinGW */

extern "C" void __TBB_malloc_proxy();
struct __TBB_malloc_proxy_caller {
    __TBB_malloc_proxy_caller() { __TBB_malloc_proxy(); }
} volatile __TBB_malloc_proxy_helper_object;

#endif // _MSC_VER

#endif //__TBB_tbbmalloc_proxy_H
