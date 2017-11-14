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

#ifndef __TBB_task_scheduler_observer_H
#define __TBB_task_scheduler_observer_H

#include "atomic.h"
#if __TBB_TASK_ARENA
#include "task_arena.h"
#endif //__TBB_TASK_ARENA

#if __TBB_SCHEDULER_OBSERVER

namespace tbb {
namespace interface6 {
class task_scheduler_observer;
}
namespace internal {

class observer_proxy;
class observer_list;

class task_scheduler_observer_v3 {
    friend class observer_proxy;
    friend class observer_list;
    friend class interface6::task_scheduler_observer;

    //! Pointer to the proxy holding this observer.
    /** Observers are proxied by the scheduler to maintain persistent lists of them. **/
    observer_proxy* my_proxy;

    //! Counter preventing the observer from being destroyed while in use by the scheduler.
    /** Valid only when observation is on. **/
    atomic<intptr_t> my_busy_count;

public:
    //! Enable or disable observation
    /** For local observers the method can be used only when the current thread
        has the task scheduler initialized or is attached to an arena.

        Repeated calls with the same state are no-ops. **/
    void __TBB_EXPORTED_METHOD observe( bool state=true );

    //! Returns true if observation is enabled, false otherwise.
    bool is_observing() const {return my_proxy!=NULL;}

    //! Construct observer with observation disabled.
    task_scheduler_observer_v3() : my_proxy(NULL) { my_busy_count.store<relaxed>(0); }

    //! Entry notification
    /** Invoked from inside observe(true) call and whenever a worker enters the arena 
        this observer is associated with. If a thread is already in the arena when
        the observer is activated, the entry notification is called before it
        executes the first stolen task.

        Obsolete semantics. For global observers it is called by a thread before
        the first steal since observation became enabled. **/
    virtual void on_scheduler_entry( bool /*is_worker*/ ) {} 

    //! Exit notification
    /** Invoked from inside observe(false) call and whenever a worker leaves the
        arena this observer is associated with.

        Obsolete semantics. For global observers it is called by a thread before
        the first steal since observation became enabled. **/
    virtual void on_scheduler_exit( bool /*is_worker*/ ) {}

    //! Destructor automatically switches observation off if it is enabled.
    virtual ~task_scheduler_observer_v3() { if(my_proxy) observe(false);}
};

} // namespace internal

#if __TBB_ARENA_OBSERVER
namespace interface6 {
class task_scheduler_observer : public internal::task_scheduler_observer_v3 {
    friend class internal::task_scheduler_observer_v3;
    friend class internal::observer_proxy;
    friend class internal::observer_list;

    /** Negative numbers with the largest absolute value to minimize probability
        of coincidence in case of a bug in busy count usage. **/
    // TODO: take more high bits for version number
    static const intptr_t v6_trait = (intptr_t)((~(uintptr_t)0 >> 1) + 1);

    //! contains task_arena pointer or tag indicating local or global semantics of the observer
    intptr_t my_context_tag;
    enum { global_tag = 0, implicit_tag = 1 };

public:
    //! Construct local or global observer in inactive state (observation disabled).
    /** For a local observer entry/exit notifications are invoked whenever a worker
        thread joins/leaves the arena of the observer's owner thread. If a thread is
        already in the arena when the observer is activated, the entry notification is
        called before it executes the first stolen task. **/
    /** TODO: Obsolete.
        Global observer semantics is obsolete as it violates master thread isolation
        guarantees and is not composable. Thus the current default behavior of the
        constructor is obsolete too and will be changed in one of the future versions
        of the library. **/
    task_scheduler_observer( bool local = false ) {
        my_context_tag = local? implicit_tag : global_tag;
    }

#if __TBB_TASK_ARENA
    //! Construct local observer for a given arena in inactive state (observation disabled).
    /** entry/exit notifications are invoked whenever a thread joins/leaves arena.
        If a thread is already in the arena when the observer is activated, the entry notification
        is called before it executes the first stolen task. **/
    task_scheduler_observer( task_arena & a) {
        my_context_tag = (intptr_t)&a;
    }
#endif //__TBB_TASK_ARENA

    /** Destructor protects instance of the observer from concurrent notification.
       It is recommended to disable observation before destructor of a derived class starts,
       otherwise it can lead to concurrent notification callback on partly destroyed object **/
    virtual ~task_scheduler_observer() { if(my_proxy) observe(false); }

    //! Enable or disable observation
    /** Warning: concurrent invocations of this method are not safe.
        Repeated calls with the same state are no-ops. **/
    void observe( bool state=true ) {
        if( state && !my_proxy ) {
            __TBB_ASSERT( !my_busy_count, "Inconsistent state of task_scheduler_observer instance");
            my_busy_count.store<relaxed>(v6_trait);
        }
        internal::task_scheduler_observer_v3::observe(state);
    }

    //! Return commands for may_sleep()
    enum { keep_awake = false, allow_sleep = true };

    //! The callback can be invoked by a worker thread before it goes to sleep.
    /** If it returns false ('keep_awake'), the thread will keep spinning and looking for work.
        It will not be called for master threads. **/
    virtual bool may_sleep() { return allow_sleep; }
};

} //namespace interface6
using interface6::task_scheduler_observer;
#else /*__TBB_ARENA_OBSERVER*/
typedef tbb::internal::task_scheduler_observer_v3 task_scheduler_observer;
#endif /*__TBB_ARENA_OBSERVER*/

} // namespace tbb

#endif /* __TBB_SCHEDULER_OBSERVER */

#endif /* __TBB_task_scheduler_observer_H */
