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

#ifndef __TBB_tbb_H
#define __TBB_tbb_H

/** 
    This header bulk-includes declarations or definitions of all the functionality 
    provided by TBB (save for malloc dependent headers). 

    If you use only a few TBB constructs, consider including specific headers only.
    Any header listed below can be included independently of others.
**/

#if TBB_PREVIEW_AGGREGATOR
#include "aggregator.h"
#endif
#include "aligned_space.h"
#include "atomic.h"
#include "blocked_range.h"
#include "blocked_range2d.h"
#include "blocked_range3d.h"
#include "cache_aligned_allocator.h"
#include "combinable.h"
#include "concurrent_hash_map.h"
#if TBB_PREVIEW_CONCURRENT_LRU_CACHE
#include "concurrent_lru_cache.h"
#endif
#include "concurrent_priority_queue.h"
#include "concurrent_queue.h"
#include "concurrent_unordered_map.h"
#include "concurrent_unordered_set.h"
#include "concurrent_vector.h"
#include "critical_section.h"
#include "enumerable_thread_specific.h"
#include "flow_graph.h"
#if TBB_PREVIEW_GLOBAL_CONTROL
#include "global_control.h"
#endif
#include "mutex.h"
#include "null_mutex.h"
#include "null_rw_mutex.h"
#include "parallel_do.h"
#include "parallel_for.h"
#include "parallel_for_each.h"
#include "parallel_invoke.h"
#include "parallel_reduce.h"
#include "parallel_scan.h"
#include "parallel_sort.h"
#include "partitioner.h"
#include "pipeline.h"
#include "queuing_mutex.h"
#include "queuing_rw_mutex.h"
#include "reader_writer_lock.h"
#include "recursive_mutex.h"
#include "spin_mutex.h"
#include "spin_rw_mutex.h"
#include "task.h"
#include "task_arena.h"
#include "task_group.h"
#include "task_scheduler_init.h"
#include "task_scheduler_observer.h"
#include "tbb_allocator.h"
#include "tbb_exception.h"
#include "tbb_thread.h"
#include "tick_count.h"

#endif /* __TBB_tbb_H */
