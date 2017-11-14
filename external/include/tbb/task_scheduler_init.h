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

#ifndef __TBB_task_scheduler_init_H
#define __TBB_task_scheduler_init_H

#include "tbb_stddef.h"
#include "limits.h"

namespace tbb {

typedef std::size_t stack_size_type;

//! @cond INTERNAL
namespace internal {
    //! Internal to library. Should not be used by clients.
    /** @ingroup task_scheduling */
    class scheduler;
} // namespace internal
//! @endcond

//! Class delimiting the scope of task scheduler activity.
/** A thread can construct a task_scheduler_init object and keep it alive
    while it uses TBB's tasking subsystem (including parallel algorithms).

    This class allows to customize properties of the TBB task pool to some extent.
    For example it can limit concurrency level of parallel work initiated by the
    given thread. It also can be used to specify stack size of the TBB worker threads,
    though this setting is not effective if the thread pool has already been created.

    If a parallel construct is used without task_scheduler_init object previously
    created, the scheduler will be initialized automatically with default settings,
    and will persist until this thread exits. Default concurrency level is defined
    as described in task_scheduler_init::initialize().
    @ingroup task_scheduling */
class task_scheduler_init: internal::no_copy {
    enum ExceptionPropagationMode {
        propagation_mode_exact = 1u,
        propagation_mode_captured = 2u,
        propagation_mode_mask = propagation_mode_exact | propagation_mode_captured
    };
#if __TBB_SUPPORTS_WORKERS_WAITING_IN_TERMINATE
    enum {
        wait_workers_in_terminate_flag = 128u
    };
#endif

    /** NULL if not currently initialized. */
    internal::scheduler* my_scheduler;
public:

    //! Typedef for number of threads that is automatic.
    static const int automatic = -1;

    //! Argument to initialize() or constructor that causes initialization to be deferred.
    static const int deferred = -2;

    //! Ensure that scheduler exists for this thread
    /** A value of -1 lets TBB decide on the number of threads, which is usually
        maximal hardware concurrency for this process, that is the number of logical
        CPUs on the machine (possibly limited by the processor affinity mask of this
        process (Windows) or of this thread (Linux, FreeBSD). It is preferable option
        for production code because it helps to avoid nasty surprises when several
        TBB based components run side-by-side or in a nested fashion inside the same
        process.

        The number_of_threads is ignored if any other task_scheduler_inits 
        currently exist.  A thread may construct multiple task_scheduler_inits.  
        Doing so does no harm because the underlying scheduler is reference counted. */
    void __TBB_EXPORTED_METHOD initialize( int number_of_threads=automatic );

    //! The overloaded method with stack size parameter
    /** Overloading is necessary to preserve ABI compatibility */
    void __TBB_EXPORTED_METHOD initialize( int number_of_threads, stack_size_type thread_stack_size );

    //! Inverse of method initialize.
    void __TBB_EXPORTED_METHOD terminate();

    //! Shorthand for default constructor followed by call to initialize(number_of_threads).
#if __TBB_SUPPORTS_WORKERS_WAITING_IN_TERMINATE
    task_scheduler_init( int number_of_threads=automatic, stack_size_type thread_stack_size=0, bool wait_workers_in_terminate = false ) : my_scheduler(NULL)
#else
    task_scheduler_init( int number_of_threads=automatic, stack_size_type thread_stack_size=0 ) : my_scheduler(NULL)
#endif
    {
        // Two lowest order bits of the stack size argument may be taken to communicate
        // default exception propagation mode of the client to be used when the
        // client manually creates tasks in the master thread and does not use
        // explicit task group context object. This is necessary because newer 
        // TBB binaries with exact propagation enabled by default may be used 
        // by older clients that expect tbb::captured_exception wrapper.
        // All zeros mean old client - no preference. 
        __TBB_ASSERT( !(thread_stack_size & propagation_mode_mask), "Requested stack size is not aligned" );
#if TBB_USE_EXCEPTIONS
        thread_stack_size |= TBB_USE_CAPTURED_EXCEPTION ? propagation_mode_captured : propagation_mode_exact;
#endif /* TBB_USE_EXCEPTIONS */
#if __TBB_SUPPORTS_WORKERS_WAITING_IN_TERMINATE
        if (wait_workers_in_terminate)
            my_scheduler = (internal::scheduler*)wait_workers_in_terminate_flag;
#endif
        initialize( number_of_threads, thread_stack_size );
    }

    //! Destroy scheduler for this thread if thread has no other live task_scheduler_inits.
    ~task_scheduler_init() {
        if( my_scheduler ) 
            terminate();
        internal::poison_pointer( my_scheduler );
    }
    //! Returns the number of threads TBB scheduler would create if initialized by default.
    /** Result returned by this method does not depend on whether the scheduler 
        has already been initialized.
        
        Because tbb 2.0 does not support blocking tasks yet, you may use this method
        to boost the number of threads in the tbb's internal pool, if your tasks are 
        doing I/O operations. The optimal number of additional threads depends on how
        much time your tasks spend in the blocked state.
        
        Before TBB 3.0 U4 this method returned the number of logical CPU in the
        system. Currently on Windows, Linux and FreeBSD it returns the number of
        logical CPUs available to the current process in accordance with its affinity
        mask.
        
        NOTE: The return value of this method never changes after its first invocation. 
        This means that changes in the process affinity mask that took place after
        this method was first invoked will not affect the number of worker threads
        in the TBB worker threads pool. */
    static int __TBB_EXPORTED_FUNC default_num_threads ();

    //! Returns true if scheduler is active (initialized); false otherwise
    bool is_active() const { return my_scheduler != NULL; }
};

} // namespace tbb

#endif /* __TBB_task_scheduler_init_H */
