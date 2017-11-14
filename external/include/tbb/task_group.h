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

#ifndef __TBB_task_group_H
#define __TBB_task_group_H

#include "task.h"
#include "tbb_exception.h"

#if __TBB_TASK_GROUP_CONTEXT

namespace tbb {

namespace internal {
    template<typename F> class task_handle_task;
}

class task_group;
class structured_task_group;

template<typename F>
class task_handle : internal::no_assign {
    template<typename _F> friend class internal::task_handle_task;
    friend class task_group;
    friend class structured_task_group;

    static const intptr_t scheduled = 0x1;

    F my_func;
    intptr_t my_state;

    void mark_scheduled () {
        // The check here is intentionally lax to avoid the impact of interlocked operation
        if ( my_state & scheduled )
            internal::throw_exception( internal::eid_invalid_multiple_scheduling );
        my_state |= scheduled;
    }
public:
    task_handle( const F& f ) : my_func(f), my_state(0) {}

    void operator() () const { my_func(); }
};

enum task_group_status {
    not_complete,
    complete,
    canceled
};

namespace internal {

template<typename F>
class task_handle_task : public task {
    task_handle<F>& my_handle;
    /*override*/ task* execute() {
        my_handle();
        return NULL;
    }
public:
    task_handle_task( task_handle<F>& h ) : my_handle(h) { h.mark_scheduled(); }
};

class task_group_base : internal::no_copy {
protected:
    empty_task* my_root;
    task_group_context my_context;

    task& owner () { return *my_root; }

    template<typename F>
    task_group_status internal_run_and_wait( F& f ) {
        __TBB_TRY {
            if ( !my_context.is_group_execution_cancelled() )
                f();
        } __TBB_CATCH( ... ) {
            my_context.register_pending_exception();
        }
        return wait();
    }

    template<typename F, typename Task>
    void internal_run( F& f ) {
        owner().spawn( *new( owner().allocate_additional_child_of(*my_root) ) Task(f) );
    }

public:
    task_group_base( uintptr_t traits = 0 )
        : my_context(task_group_context::bound, task_group_context::default_traits | traits)
    {
        my_root = new( task::allocate_root(my_context) ) empty_task;
        my_root->set_ref_count(1);
    }

    ~task_group_base() __TBB_NOEXCEPT(false) {
        if( my_root->ref_count() > 1 ) {
            bool stack_unwinding_in_progress = std::uncaught_exception();
            // Always attempt to do proper cleanup to avoid inevitable memory corruption 
            // in case of missing wait (for the sake of better testability & debuggability)
            if ( !is_canceling() )
                cancel();
            __TBB_TRY {
                my_root->wait_for_all();
            } __TBB_CATCH (...) {
                task::destroy(*my_root);
                __TBB_RETHROW();
            }
            task::destroy(*my_root);
            if ( !stack_unwinding_in_progress )
                internal::throw_exception( internal::eid_missing_wait );
        }
        else {
            task::destroy(*my_root);
        }
    }

    template<typename F>
    void run( task_handle<F>& h ) {
        internal_run< task_handle<F>, internal::task_handle_task<F> >( h );
    }

    task_group_status wait() {
        __TBB_TRY {
            my_root->wait_for_all();
        } __TBB_CATCH( ... ) {
            my_context.reset();
            __TBB_RETHROW();
        }
        if ( my_context.is_group_execution_cancelled() ) {
            // TODO: the reset method is not thread-safe. Ensure the correct behavior.
            my_context.reset();
            return canceled;
        }
        return complete;
    }

    bool is_canceling() {
        return my_context.is_group_execution_cancelled();
    }

    void cancel() {
        my_context.cancel_group_execution();
    }
}; // class task_group_base

} // namespace internal

class task_group : public internal::task_group_base {
public:
    task_group () : task_group_base( task_group_context::concurrent_wait ) {}

#if __SUNPRO_CC
    template<typename F>
    void run( task_handle<F>& h ) {
        internal_run< task_handle<F>, internal::task_handle_task<F> >( h );
    }
#else
    using task_group_base::run;
#endif

    template<typename F>
    void run( const F& f ) {
        internal_run< const F, internal::function_task<F> >( f );
    }

    template<typename F>
    task_group_status run_and_wait( const F& f ) {
        return internal_run_and_wait<const F>( f );
    }

    template<typename F>
    task_group_status run_and_wait( task_handle<F>& h ) {
      h.mark_scheduled();
      return internal_run_and_wait< task_handle<F> >( h );
    }
}; // class task_group

class structured_task_group : public internal::task_group_base {
public:
    template<typename F>
    task_group_status run_and_wait ( task_handle<F>& h ) {
        h.mark_scheduled();
        return internal_run_and_wait< task_handle<F> >( h );
    }

    task_group_status wait() {
        task_group_status res = task_group_base::wait();
        my_root->set_ref_count(1);
        return res;
    }
}; // class structured_task_group

inline 
bool is_current_task_group_canceling() {
    return task::self().is_cancelled();
}

template<class F>
task_handle<F> make_task( const F& f ) {
    return task_handle<F>( f );
}

} // namespace tbb

#endif /* __TBB_TASK_GROUP_CONTEXT */

#endif /* __TBB_task_group_H */
