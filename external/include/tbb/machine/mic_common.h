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

#ifndef __TBB_mic_common_H
#define __TBB_mic_common_H

#ifndef __TBB_machine_H
#error Do not #include this internal file directly; use public TBB headers instead.
#endif

#if ! __TBB_DEFINE_MIC
    #error mic_common.h should be included only when building for Intel(R) Many Integrated Core Architecture
#endif

#ifndef __TBB_PREFETCHING
#define __TBB_PREFETCHING 1
#endif
#if __TBB_PREFETCHING
#include <immintrin.h>
#define __TBB_cl_prefetch(p) _mm_prefetch((const char*)p, _MM_HINT_T1)
#define __TBB_cl_evict(p) _mm_clevict(p, _MM_HINT_T1)
#endif

/** Intel(R) Many Integrated Core Architecture does not support mfence and pause instructions **/
#define __TBB_full_memory_fence() __asm__ __volatile__("lock; addl $0,(%%rsp)":::"memory")
#define __TBB_Pause(x) _mm_delay_32(16*(x))
#define __TBB_STEALING_PAUSE 1500/16
#include <sched.h>
#define __TBB_Yield() sched_yield()

/** Specifics **/
#define __TBB_STEALING_ABORT_ON_CONTENTION 1
#define __TBB_YIELD2P 1
#define __TBB_HOARD_NONLOCAL_TASKS 1

#if ! ( __FreeBSD__ || __linux__ )
    #error Intel(R) Many Integrated Core Compiler does not define __FreeBSD__ or __linux__ anymore. Check for the __TBB_XXX_BROKEN defined under __FreeBSD__ or __linux__.
#endif /* ! ( __FreeBSD__ || __linux__ ) */

#endif /* __TBB_mic_common_H */
