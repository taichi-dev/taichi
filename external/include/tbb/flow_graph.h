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

#ifndef __TBB_flow_graph_H
#define __TBB_flow_graph_H

#include "tbb_stddef.h"
#include "atomic.h"
#include "spin_mutex.h"
#include "null_mutex.h"
#include "spin_rw_mutex.h"
#include "null_rw_mutex.h"
#include "task.h"
#include "cache_aligned_allocator.h"
#include "tbb_exception.h"
#include "internal/_template_helpers.h"
#include "internal/_aggregator_impl.h"
#include "tbb_profiling.h"

#if __TBB_PREVIEW_ASYNC_NODE
#include "task_arena.h"
#endif

#if TBB_DEPRECATED_FLOW_ENQUEUE
#define FLOW_SPAWN(a) tbb::task::enqueue((a))
#else
#define FLOW_SPAWN(a) tbb::task::spawn((a))
#endif

// use the VC10 or gcc version of tuple if it is available.
#if __TBB_CPP11_TUPLE_PRESENT
    #include <tuple>
namespace tbb {
    namespace flow {
        using std::tuple;
        using std::tuple_size;
        using std::tuple_element;
        using std::get;
    }
}
#else
    #include "compat/tuple"
#endif

#include<list>
#include<queue>

/** @file
  \brief The graph related classes and functions

  There are some applications that best express dependencies as messages
  passed between nodes in a graph.  These messages may contain data or
  simply act as signals that a predecessors has completed. The graph
  class and its associated node classes can be used to express such
  applications.
*/

namespace tbb {
namespace flow {

//! An enumeration the provides the two most common concurrency levels: unlimited and serial
enum concurrency { unlimited = 0, serial = 1 };

namespace interface8 {

namespace internal {
    template<typename T, typename M> class successor_cache;
    template<typename T, typename M> class broadcast_cache;
    template<typename T, typename M> class round_robin_cache;

#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
    template< typename C> class edge_container;
#endif
}

//A generic null type
struct null_type {};

//! An empty class used for messages that mean "I'm done"
class continue_msg {};

template< typename T > class sender;
template< typename T > class receiver;
class continue_receiver;

//! Pure virtual template class that defines a sender of messages of type T
template< typename T >
class sender {
public:
    //! The output type of this sender
    typedef T output_type;

    //! The successor type for this node
    typedef receiver<T> successor_type;

    virtual ~sender() {}

    //! Add a new successor to this node
    virtual bool register_successor( successor_type &r ) = 0;

    //! Removes a successor from this node
    virtual bool remove_successor( successor_type &r ) = 0;

    //! Request an item from the sender
    virtual bool try_get( T & ) { return false; }

    //! Reserves an item in the sender
    virtual bool try_reserve( T & ) { return false; }

    //! Releases the reserved item
    virtual bool try_release( ) { return false; }

    //! Consumes the reserved item
    virtual bool try_consume( ) { return false; }

#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
    //! interface to record edges for traversal & deletion
    typedef typename  internal::edge_container<successor_type> built_successors_type;
    typedef typename  built_successors_type::edge_list_type successor_list_type;
    virtual built_successors_type &built_successors()                   = 0;
    virtual void    internal_add_built_successor( successor_type & )    = 0;
    virtual void    internal_delete_built_successor( successor_type & ) = 0;
    virtual void    copy_successors( successor_list_type &)             = 0;
    virtual size_t  successor_count()                                   = 0;
#endif
};  // class sender<T>

template< typename T > class limiter_node;  // needed for resetting decrementer
template< typename R, typename B > class run_and_put_task;

static tbb::task * const SUCCESSFULLY_ENQUEUED = (task *)-1;

// flags to modify the behavior of the graph reset().  Can be combined.
enum reset_flags {
    rf_reset_protocol   = 0,
    rf_reset_bodies     = 1<<0,  // delete the current node body, reset to a copy of the initial node body.
    rf_clear_edges      = 1<<1   // delete edges
};

// enqueue left task if necessary.  Returns the non-enqueued task if there is one.
static inline tbb::task *combine_tasks( tbb::task * left, tbb::task * right) {
    // if no RHS task, don't change left.
    if(right == NULL) return left;
    // right != NULL
    if(left == NULL) return right;
    if(left == SUCCESSFULLY_ENQUEUED) return right;
    // left contains a task
    if(right != SUCCESSFULLY_ENQUEUED) {
        // both are valid tasks
        FLOW_SPAWN(*left);
        return right;
    }
    return left;
}

//! Pure virtual template class that defines a receiver of messages of type T
template< typename T >
class receiver {
public:
    //! The input type of this receiver
    typedef T input_type;

    //! The predecessor type for this node
    typedef sender<T> predecessor_type;

    //! Destructor
    virtual ~receiver() {}

    //! Put an item to the receiver
    bool try_put( const T& t ) {
        task *res = try_put_task(t);
        if (!res) return false;
        if (res != SUCCESSFULLY_ENQUEUED) FLOW_SPAWN(*res);
        return true;
    }

    //! put item to successor; return task to run the successor if possible.
protected:
    template< typename R, typename B > friend class run_and_put_task;
    template<typename X, typename Y> friend class internal::broadcast_cache;
    template<typename X, typename Y> friend class internal::round_robin_cache;
    virtual task *try_put_task(const T& t) = 0;
public:

    //! Add a predecessor to the node
    virtual bool register_predecessor( predecessor_type & ) { return false; }

    //! Remove a predecessor from the node
    virtual bool remove_predecessor( predecessor_type & ) { return false; }

#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
    typedef typename internal::edge_container<predecessor_type> built_predecessors_type;
    typedef typename built_predecessors_type::edge_list_type predecessor_list_type;
    virtual built_predecessors_type &built_predecessors()                  = 0;
    virtual void   internal_add_built_predecessor( predecessor_type & )    = 0;
    virtual void   internal_delete_built_predecessor( predecessor_type & ) = 0;
    virtual void   copy_predecessors( predecessor_list_type & )            = 0;
    virtual size_t predecessor_count()                                     = 0;
#endif

protected:
    //! put receiver back in initial state
    template<typename U> friend class limiter_node;
    virtual void reset_receiver(reset_flags f = rf_reset_protocol) = 0;

    template<typename TT, typename M> friend class internal::successor_cache;
    virtual bool is_continue_receiver() { return false; }

#if __TBB_PREVIEW_OPENCL_NODE
    template< typename, typename > friend class proxy_dependency_receiver;
#endif /* __TBB_PREVIEW_OPENCL_NODE */
}; // class receiver<T>

#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
//* holder of edges both for caches and for those nodes which do not have predecessor caches.
// C == receiver< ... > or sender< ... >, depending.
namespace internal {
template<typename C>
class edge_container {

public:
    typedef std::list<C *, tbb::tbb_allocator<C *> > edge_list_type;

    void add_edge( C &s) {
        built_edges.push_back( &s );
    }

    void delete_edge( C &s) {
        for ( typename edge_list_type::iterator i = built_edges.begin(); i != built_edges.end(); ++i ) {
            if ( *i == &s )  {
                (void)built_edges.erase(i);
                return;  // only remove one predecessor per request
            }
        }
    }

    void copy_edges( edge_list_type &v) {
        v = built_edges;
    }

    size_t edge_count() {
        return (size_t)(built_edges.size());
    }

    void clear() {
        built_edges.clear();
    }

    // methods remove the statement from all predecessors/successors liste in the edge
    // container.
    template< typename S > void sender_extract( S &s ); 
    template< typename R > void receiver_extract( R &r ); 
    
private: 
    edge_list_type built_edges;
};  // class edge_container
}  // namespace internal
#endif  /* TBB_PREVIEW_FLOW_GRAPH_FEATURES */

//! Base class for receivers of completion messages
/** These receivers automatically reset, but cannot be explicitly waited on */
class continue_receiver : public receiver< continue_msg > {
public:

    //! The input type
    typedef continue_msg input_type;

    //! The predecessor type for this node
    typedef sender< continue_msg > predecessor_type;

    //! Constructor
    continue_receiver( int number_of_predecessors = 0 ) {
        my_predecessor_count = my_initial_predecessor_count = number_of_predecessors;
        my_current_count = 0;
    }

    //! Copy constructor
    continue_receiver( const continue_receiver& src ) : receiver<continue_msg>() {
        my_predecessor_count = my_initial_predecessor_count = src.my_initial_predecessor_count;
        my_current_count = 0;
    }

    //! Destructor
    virtual ~continue_receiver() { }

    //! Increments the trigger threshold
    /* override */ bool register_predecessor( predecessor_type & ) {
        spin_mutex::scoped_lock l(my_mutex);
        ++my_predecessor_count;
        return true;
    }

    //! Decrements the trigger threshold
    /** Does not check to see if the removal of the predecessor now makes the current count
        exceed the new threshold.  So removing a predecessor while the graph is active can cause
        unexpected results. */
    /* override */ bool remove_predecessor( predecessor_type & ) {
        spin_mutex::scoped_lock l(my_mutex);
        --my_predecessor_count;
        return true;
    }

#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
    typedef internal::edge_container<predecessor_type> built_predecessors_type;
    typedef built_predecessors_type::edge_list_type predecessor_list_type;
    /*override*/ built_predecessors_type &built_predecessors() { return my_built_predecessors; }

    /*override*/ void internal_add_built_predecessor( predecessor_type &s) {
        spin_mutex::scoped_lock l(my_mutex);
        my_built_predecessors.add_edge( s );
    }

    /*override*/ void internal_delete_built_predecessor( predecessor_type &s) {
        spin_mutex::scoped_lock l(my_mutex);
        my_built_predecessors.delete_edge(s);
    }

    /*override*/ void copy_predecessors( predecessor_list_type &v) {
        spin_mutex::scoped_lock l(my_mutex);
        my_built_predecessors.copy_edges(v);
    }

    /*override*/ size_t predecessor_count() {
        spin_mutex::scoped_lock l(my_mutex);
        return my_built_predecessors.edge_count();
    }

#endif  /* TBB_PREVIEW_FLOW_GRAPH_FEATURES */

protected:
    template< typename R, typename B > friend class run_and_put_task;
    template<typename X, typename Y> friend class internal::broadcast_cache;
    template<typename X, typename Y> friend class internal::round_robin_cache;
    // execute body is supposed to be too small to create a task for.
    /* override */ task *try_put_task( const input_type & ) {
        {
            spin_mutex::scoped_lock l(my_mutex);
            if ( ++my_current_count < my_predecessor_count )
                return SUCCESSFULLY_ENQUEUED;
            else
                my_current_count = 0;
        }
        task * res = execute();
        return res? res : SUCCESSFULLY_ENQUEUED;
    }

#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
    // continue_receiver must contain its own built_predecessors because it does
    // not have a node_cache.
    built_predecessors_type my_built_predecessors;
#endif
    spin_mutex my_mutex;
    int my_predecessor_count;
    int my_current_count;
    int my_initial_predecessor_count;
    // the friend declaration in the base class did not eliminate the "protected class"
    // error in gcc 4.1.2
    template<typename U> friend class limiter_node;

    /*override*/void reset_receiver( reset_flags f ) {
        my_current_count = 0;
        if (f & rf_clear_edges) {
#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
            my_built_predecessors.clear();
#endif
            my_predecessor_count = my_initial_predecessor_count;
        }
    }

    //! Does whatever should happen when the threshold is reached
    /** This should be very fast or else spawn a task.  This is
        called while the sender is blocked in the try_put(). */
    virtual task * execute() = 0;
    template<typename TT, typename M> friend class internal::successor_cache;
    /*override*/ bool is_continue_receiver() { return true; }

}; // class continue_receiver
}  // interface8

#if __TBB_PREVIEW_MESSAGE_BASED_KEY_MATCHING
    template <typename K, typename T>
    K key_from_message( const T &t ) {
        return t.key();
    }
#endif /* __TBB_PREVIEW_MESSAGE_BASED_KEY_MATCHING */

    using interface8::sender;
    using interface8::receiver;
    using interface8::continue_receiver;
}  // flow
}  // tbb

#include "internal/_flow_graph_trace_impl.h"
#include "internal/_tbb_hash_compare_impl.h"

namespace tbb {
namespace flow {
namespace interface8 {

#include "internal/_flow_graph_impl.h"
#include "internal/_flow_graph_types_impl.h"
using namespace internal::graph_policy_namespace;

class graph;
class graph_node;

template <typename GraphContainerType, typename GraphNodeType>
class graph_iterator {
    friend class graph;
    friend class graph_node;
public:
    typedef size_t size_type;
    typedef GraphNodeType value_type;
    typedef GraphNodeType* pointer;
    typedef GraphNodeType& reference;
    typedef const GraphNodeType& const_reference;
    typedef std::forward_iterator_tag iterator_category;

    //! Default constructor
    graph_iterator() : my_graph(NULL), current_node(NULL) {}

    //! Copy constructor
    graph_iterator(const graph_iterator& other) :
        my_graph(other.my_graph), current_node(other.current_node)
    {}

    //! Assignment
    graph_iterator& operator=(const graph_iterator& other) {
        if (this != &other) {
            my_graph = other.my_graph;
            current_node = other.current_node;
        }
        return *this;
    }

    //! Dereference
    reference operator*() const;

    //! Dereference
    pointer operator->() const;

    //! Equality
    bool operator==(const graph_iterator& other) const {
        return ((my_graph == other.my_graph) && (current_node == other.current_node));
    }

    //! Inequality
    bool operator!=(const graph_iterator& other) const { return !(operator==(other)); }

    //! Pre-increment
    graph_iterator& operator++() {
        internal_forward();
        return *this;
    }

    //! Post-increment
    graph_iterator operator++(int) {
        graph_iterator result = *this;
        operator++();
        return result;
    }

private:
    // the graph over which we are iterating
    GraphContainerType *my_graph;
    // pointer into my_graph's my_nodes list
    pointer current_node;

    //! Private initializing constructor for begin() and end() iterators
    graph_iterator(GraphContainerType *g, bool begin);
    void internal_forward();
};  // class graph_iterator

//! The graph class
/** This class serves as a handle to the graph */
class graph : tbb::internal::no_copy {
    friend class graph_node;

    template< typename Body >
    class run_task : public task {
    public:
        run_task( Body& body ) : my_body(body) {}
        task *execute() {
            my_body();
            return NULL;
        }
    private:
        Body my_body;
    };

    template< typename Receiver, typename Body >
    class run_and_put_task : public task {
    public:
        run_and_put_task( Receiver &r, Body& body ) : my_receiver(r), my_body(body) {}
        task *execute() {
            task *res = my_receiver.try_put_task( my_body() );
            if (res == SUCCESSFULLY_ENQUEUED) res = NULL;
            return res;
        }
    private:
        Receiver &my_receiver;
        Body my_body;
    };
    typedef std::list<task *> task_list_type;

#if __TBB_PREVIEW_ASYNC_NODE
    class wait_functor {
        task* graph_root_task;
    public:
        wait_functor( task* t ) : graph_root_task(t) {}
        void operator()() const { graph_root_task->wait_for_all(); }
    };

    void prepare_task_arena() {
        my_task_arena = new tbb::task_arena(tbb::internal::attach());
        if (!my_task_arena->is_active())
            my_task_arena->initialize(); // create a new, default-initialized arena
        __TBB_ASSERT(my_task_arena->is_active(), NULL);
    }
#endif

public:
    //! Constructs a graph with isolated task_group_context
    graph() : my_nodes(NULL), my_nodes_last(NULL) {
#if __TBB_PREVIEW_ASYNC_NODE
        prepare_task_arena();
#endif
        own_context = true;
        cancelled = false;
        caught_exception = false;
        my_context = new task_group_context();
        my_root_task = ( new ( task::allocate_root(*my_context) ) empty_task );
        my_root_task->set_ref_count(1);
        tbb::internal::fgt_graph( this );
        my_is_active = true;
    }

    //! Constructs a graph with use_this_context as context
    explicit graph(task_group_context& use_this_context) :
      my_context(&use_this_context), my_nodes(NULL), my_nodes_last(NULL) {
#if __TBB_PREVIEW_ASYNC_NODE
        prepare_task_arena();
#endif
        own_context = false;
        my_root_task = ( new ( task::allocate_root(*my_context) ) empty_task );
        my_root_task->set_ref_count(1);
        tbb::internal::fgt_graph( this );
        my_is_active = true;
    }

    //! Destroys the graph.
    /** Calls wait_for_all, then destroys the root task and context. */
    ~graph() {
        wait_for_all();
        my_root_task->set_ref_count(0);
        task::destroy( *my_root_task );
        if (own_context) delete my_context;
#if __TBB_PREVIEW_ASYNC_NODE
        delete my_task_arena;
#endif
    }

#if TBB_PREVIEW_FLOW_GRAPH_TRACE
    void set_name( const char *name ) {
        tbb::internal::fgt_graph_desc( this, name );
    }
#endif

    //! Used to register that an external entity may still interact with the graph.
    /** The graph will not return from wait_for_all until a matching number of decrement_wait_count calls
        is made. */
    void increment_wait_count() {
        if (my_root_task)
            my_root_task->increment_ref_count();
    }

    //! Deregisters an external entity that may have interacted with the graph.
    /** The graph will not return from wait_for_all until all the number of decrement_wait_count calls
        matches the number of increment_wait_count calls. */
    void decrement_wait_count() {
        if (my_root_task)
            my_root_task->decrement_ref_count();
    }

    //! Spawns a task that runs a body and puts its output to a specific receiver
    /** The task is spawned as a child of the graph. This is useful for running tasks
        that need to block a wait_for_all() on the graph.  For example a one-off source. */
    template< typename Receiver, typename Body >
        void run( Receiver &r, Body body ) {
            if(is_active()) {
                FLOW_SPAWN( (* new ( task::allocate_additional_child_of( *root_task() ) )
                   run_and_put_task< Receiver, Body >( r, body )) );
            }
    }

    //! Spawns a task that runs a function object
    /** The task is spawned as a child of the graph. This is useful for running tasks
        that need to block a wait_for_all() on the graph. For example a one-off source. */
    template< typename Body >
    void run( Body body ) {
        if(is_active()) {
            FLOW_SPAWN( * new ( task::allocate_additional_child_of( *root_task() ) ) run_task< Body >( body ) );
        }
    }

    //! Wait until graph is idle and decrement_wait_count calls equals increment_wait_count calls.
    /** The waiting thread will go off and steal work while it is block in the wait_for_all. */
    void wait_for_all() {
        cancelled = false;
        caught_exception = false;
        if (my_root_task) {
#if TBB_USE_EXCEPTIONS
            try {
#endif
#if __TBB_PREVIEW_ASYNC_NODE
                my_task_arena->execute(wait_functor(my_root_task));
#else
                my_root_task->wait_for_all();
#endif
                cancelled = my_context->is_group_execution_cancelled();
#if TBB_USE_EXCEPTIONS
            }
            catch(...) {
                my_root_task->set_ref_count(1);
                my_context->reset();
                caught_exception = true;
                cancelled = true;
                throw;
            }
#endif
            // TODO: the "if" condition below is just a work-around to support the concurrent wait
            // mode. The cancelation and exception mechanisms are still broken in this mode.
            // Consider using task group not to re-implement the same functionality.
            if ( !(my_context->traits() & task_group_context::concurrent_wait) ) {
                my_context->reset();  // consistent with behavior in catch()
                my_root_task->set_ref_count(1);
            }
        }
    }

    //! Returns the root task of the graph
    task * root_task() {
        return my_root_task;
    }
    
    void set_active(bool a = true) {
       my_is_active = a;
    }

    bool is_active() {
       return my_is_active;
    }

    void add_task_to_reset_list(task *tp) {
        my_reset_task_list.push_back(tp);
    }

    // ITERATORS
    template<typename C, typename N>
    friend class graph_iterator;

    // Graph iterator typedefs
    typedef graph_iterator<graph,graph_node> iterator;
    typedef graph_iterator<const graph,const graph_node> const_iterator;

    // Graph iterator constructors
    //! start iterator
    iterator begin() { return iterator(this, true); }
    //! end iterator
    iterator end() { return iterator(this, false); }
     //! start const iterator
    const_iterator begin() const { return const_iterator(this, true); }
    //! end const iterator
    const_iterator end() const { return const_iterator(this, false); }
    //! start const iterator
    const_iterator cbegin() const { return const_iterator(this, true); }
    //! end const iterator
    const_iterator cend() const { return const_iterator(this, false); }

    //! return status of graph execution
    bool is_cancelled() { return cancelled; }
    bool exception_thrown() { return caught_exception; }

    // thread-unsafe state reset.
    void reset(reset_flags f = rf_reset_protocol);

private:
    task *my_root_task;
    task_group_context *my_context;
    bool own_context;
    bool cancelled;
    bool caught_exception;
    bool my_is_active;
    task_list_type my_reset_task_list;

    graph_node *my_nodes, *my_nodes_last;

    spin_mutex nodelist_mutex;
    void register_node(graph_node *n);
    void remove_node(graph_node *n);

#if __TBB_PREVIEW_ASYNC_NODE
    template < typename Input, typename Output, typename Policy, typename Allocator >
    friend class async_node;
    task_arena* my_task_arena;
#endif
};  // class graph

template <typename C, typename N>
graph_iterator<C,N>::graph_iterator(C *g, bool begin) : my_graph(g), current_node(NULL)
{
    if (begin) current_node = my_graph->my_nodes;
    //else it is an end iterator by default
}

template <typename C, typename N>
typename graph_iterator<C,N>::reference graph_iterator<C,N>::operator*() const {
    __TBB_ASSERT(current_node, "graph_iterator at end");
    return *operator->();
}

template <typename C, typename N>
typename graph_iterator<C,N>::pointer graph_iterator<C,N>::operator->() const {
    return current_node;
}


template <typename C, typename N>
void graph_iterator<C,N>::internal_forward() {
    if (current_node) current_node = current_node->next;
}

//! The base of all graph nodes.
class graph_node : tbb::internal::no_assign {
    friend class graph;
    template<typename C, typename N>
    friend class graph_iterator;
protected:
    graph& my_graph;
    graph_node *next, *prev;
public:
    graph_node(graph& g) : my_graph(g) {
        my_graph.register_node(this);
    }
    virtual ~graph_node() {
        my_graph.remove_node(this);
    }

#if TBB_PREVIEW_FLOW_GRAPH_TRACE
    virtual void set_name( const char *name ) = 0;
#endif

#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
    virtual void extract( ) = 0;
#endif

protected:
    // performs the reset on an individual node.
    virtual void reset_node(reset_flags f=rf_reset_protocol) = 0;
};  // class graph_node

inline void graph::register_node(graph_node *n) {
    n->next = NULL;
    {
        spin_mutex::scoped_lock lock(nodelist_mutex);
        n->prev = my_nodes_last;
        if (my_nodes_last) my_nodes_last->next = n;
        my_nodes_last = n;
        if (!my_nodes) my_nodes = n;
    }
}

inline void graph::remove_node(graph_node *n) {
    {
        spin_mutex::scoped_lock lock(nodelist_mutex);
        __TBB_ASSERT(my_nodes && my_nodes_last, "graph::remove_node: Error: no registered nodes");
        if (n->prev) n->prev->next = n->next;
        if (n->next) n->next->prev = n->prev;
        if (my_nodes_last == n) my_nodes_last = n->prev;
        if (my_nodes == n) my_nodes = n->next;
    }
    n->prev = n->next = NULL;
}

inline void graph::reset(  reset_flags f ) {
    // reset context
    set_active(false);
    if(my_context) my_context->reset();
    cancelled = false;
    caught_exception = false;
    // reset all the nodes comprising the graph
    for(iterator ii = begin(); ii != end(); ++ii) {
        graph_node *my_p = &(*ii);
        my_p->reset_node(f);
    }
#if __TBB_PREVIEW_ASYNC_NODE
    // Reattach the arena. Might be useful to run the graph in a particular task_arena
    // while not limiting graph lifetime to a single task_arena::execute() call.
    delete my_task_arena;
    prepare_task_arena();
#endif
    set_active(true);
    // now spawn the tasks necessary to start the graph
    for(task_list_type::iterator rti = my_reset_task_list.begin(); rti != my_reset_task_list.end(); ++rti) {
        FLOW_SPAWN(*(*rti));
    }
    my_reset_task_list.clear();
}


#include "internal/_flow_graph_node_impl.h"

//! An executable node that acts as a source, i.e. it has no predecessors
template < typename Output >
class source_node : public graph_node, public sender< Output > {
protected:
    using graph_node::my_graph;
public:
    //! The type of the output message, which is complete
    typedef Output output_type;

    //! The type of successors of this node
    typedef receiver< Output > successor_type;

    //Source node has no input type
    typedef null_type input_type;

#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
    typedef typename sender<output_type>::built_successors_type built_successors_type;
    typedef typename sender<output_type>::successor_list_type successor_list_type;
#endif

    //! Constructor for a node with a successor
    template< typename Body >
    source_node( graph &g, Body body, bool is_active = true )
        : graph_node(g), my_active(is_active), init_my_active(is_active),
        my_body( new internal::source_body_leaf< output_type, Body>(body) ),
        my_init_body( new internal::source_body_leaf< output_type, Body>(body) ),
        my_reserved(false), my_has_cached_item(false)
    {
        my_successors.set_owner(this);
        tbb::internal::fgt_node_with_body( tbb::internal::FLOW_SOURCE_NODE, &this->my_graph,
                                           static_cast<sender<output_type> *>(this), this->my_body );
    }

    //! Copy constructor
    source_node( const source_node& src ) :
        graph_node(src.my_graph), sender<Output>(),
        my_active(src.init_my_active),
        init_my_active(src.init_my_active), my_body( src.my_init_body->clone() ), my_init_body(src.my_init_body->clone() ),
        my_reserved(false), my_has_cached_item(false)
    {
        my_successors.set_owner(this);
        tbb::internal::fgt_node_with_body( tbb::internal::FLOW_SOURCE_NODE, &this->my_graph,
                                           static_cast<sender<output_type> *>(this), this->my_body );
    }

    //! The destructor
    ~source_node() { delete my_body; delete my_init_body; }

#if TBB_PREVIEW_FLOW_GRAPH_TRACE
    /* override */ void set_name( const char *name ) {
        tbb::internal::fgt_node_desc( this, name );
    }
#endif

    //! Add a new successor to this node
    /* override */ bool register_successor( successor_type &r ) {
        spin_mutex::scoped_lock lock(my_mutex);
        my_successors.register_successor(r);
        if ( my_active )
            spawn_put();
        return true;
    }

    //! Removes a successor from this node
    /* override */ bool remove_successor( successor_type &r ) {
        spin_mutex::scoped_lock lock(my_mutex);
        my_successors.remove_successor(r);
        return true;
    }

#if TBB_PREVIEW_FLOW_GRAPH_FEATURES

    /*override*/ built_successors_type &built_successors() { return my_successors.built_successors(); }

    /*override*/void internal_add_built_successor( successor_type &r) {
        spin_mutex::scoped_lock lock(my_mutex);
        my_successors.internal_add_built_successor(r);
    }

    /*override*/void internal_delete_built_successor( successor_type &r) {
        spin_mutex::scoped_lock lock(my_mutex);
        my_successors.internal_delete_built_successor(r);
    }

    /*override*/size_t successor_count() {
        spin_mutex::scoped_lock lock(my_mutex);
        return my_successors.successor_count();
    }

    /*override*/void copy_successors(successor_list_type &v) {
        spin_mutex::scoped_lock l(my_mutex);
        my_successors.copy_successors(v);
    }
#endif  /* TBB_PREVIEW_FLOW_GRAPH_FEATURES */

    //! Request an item from the node
    /*override */ bool try_get( output_type &v ) {
        spin_mutex::scoped_lock lock(my_mutex);
        if ( my_reserved )
            return false;

        if ( my_has_cached_item ) {
            v = my_cached_item;
            my_has_cached_item = false;
            return true;
        }
        // we've been asked to provide an item, but we have none.  enqueue a task to
        // provide one.
        spawn_put();
        return false;
    }

    //! Reserves an item.
    /* override */ bool try_reserve( output_type &v ) {
        spin_mutex::scoped_lock lock(my_mutex);
        if ( my_reserved ) {
            return false;
        }

        if ( my_has_cached_item ) {
            v = my_cached_item;
            my_reserved = true;
            return true;
        } else {
            return false;
        }
    }

    //! Release a reserved item.
    /** true = item has been released and so remains in sender, dest must request or reserve future items */
    /* override */ bool try_release( ) {
        spin_mutex::scoped_lock lock(my_mutex);
        __TBB_ASSERT( my_reserved && my_has_cached_item, "releasing non-existent reservation" );
        my_reserved = false;
        if(!my_successors.empty())
            spawn_put();
        return true;
    }

    //! Consumes a reserved item
    /* override */ bool try_consume( ) {
        spin_mutex::scoped_lock lock(my_mutex);
        __TBB_ASSERT( my_reserved && my_has_cached_item, "consuming non-existent reservation" );
        my_reserved = false;
        my_has_cached_item = false;
        if ( !my_successors.empty() ) {
            spawn_put();
        }
        return true;
    }

    //! Activates a node that was created in the inactive state
    void activate() {
        spin_mutex::scoped_lock lock(my_mutex);
        my_active = true;
        if ( !my_successors.empty() )
            spawn_put();
    }

    template<typename Body>
    Body copy_function_object() {
        internal::source_body<output_type> &body_ref = *this->my_body;
        return dynamic_cast< internal::source_body_leaf<output_type, Body> & >(body_ref).get_body();
    }

#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
    /*override*/void extract( ) {
        my_successors.built_successors().sender_extract(*this);   // removes "my_owner" == this from each successor
        my_active = init_my_active;
        my_reserved = false;
        if(my_has_cached_item) my_has_cached_item = false;
    }
#endif

protected:

    //! resets the source_node to its initial state
    /*override*/void reset_node( reset_flags f) {
        my_active = init_my_active;
        my_reserved =false;
        if(my_has_cached_item) {
            my_has_cached_item = false;
        }
        if(f & rf_clear_edges) my_successors.clear();
        if(f & rf_reset_bodies) {
            internal::source_body<output_type> *tmp = my_init_body->clone();
            delete my_body;
            my_body = tmp;
        }
        if(my_active)
            this->my_graph.add_task_to_reset_list(create_put_task());
    }

private:
    spin_mutex my_mutex;
    bool my_active;
    bool init_my_active;
    internal::source_body<output_type> *my_body;
    internal::source_body<output_type> *my_init_body;
    internal::broadcast_cache< output_type > my_successors;
    bool my_reserved;
    bool my_has_cached_item;
    output_type my_cached_item;

    // used by apply_body_bypass, can invoke body of node.
    bool try_reserve_apply_body(output_type &v) {
        spin_mutex::scoped_lock lock(my_mutex);
        if ( my_reserved ) {
            return false;
        }
        if ( !my_has_cached_item ) {
            tbb::internal::fgt_begin_body( my_body );
            bool r = (*my_body)(my_cached_item);
            tbb::internal::fgt_end_body( my_body );
            if (r) {
                my_has_cached_item = true;
            }
        }
        if ( my_has_cached_item ) {
            v = my_cached_item;
            my_reserved = true;
            return true;
        } else {
            return false;
        }
    }

    // when resetting, and if the source_node was created with my_active == true, then
    // when we reset the node we must store a task to run the node, and spawn it only
    // after the reset is complete and is_active() is again true.  This is why we don't
    // test for is_active() here.
    task* create_put_task() {
        return ( new ( task::allocate_additional_child_of( *(this->my_graph.root_task()) ) )
                        internal:: source_task_bypass < source_node< output_type > >( *this ) );
    }

    //! Spawns a task that applies the body
    /* override */ void spawn_put( ) {
        if(this->my_graph.is_active()) {
            FLOW_SPAWN( *create_put_task());
        }
    }

    friend class internal::source_task_bypass< source_node< output_type > >;
    //! Applies the body.  Returning SUCCESSFULLY_ENQUEUED okay; forward_task_bypass will handle it.
    /* override */ task * apply_body_bypass( ) {
        output_type v;
        if ( !try_reserve_apply_body(v) )
            return NULL;

        task *last_task = my_successors.try_put_task(v);
        if ( last_task )
            try_consume();
        else
            try_release();
        return last_task;
    }
};  // class source_node

template<typename T>
struct allocate_buffer {
    static const bool value = false;
};

template<>
struct allocate_buffer<queueing> {
    static const bool value = true;
};

//! Implements a function node that supports Input -> Output
template < typename Input, typename Output = continue_msg, typename Policy = queueing, typename Allocator=cache_aligned_allocator<Input> >
class function_node : public graph_node, public internal::function_input<Input,Output,Allocator>, public internal::function_output<Output> {
public:
    typedef Input input_type;
    typedef Output output_type;
    typedef sender< input_type > predecessor_type;
    typedef receiver< output_type > successor_type;
    typedef internal::function_input<input_type,output_type,Allocator> fInput_type;
    typedef internal::function_input_queue<input_type, Allocator> input_queue_type;
    typedef internal::function_output<output_type> fOutput_type;
#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
    using typename fInput_type::predecessor_list_type;
    using typename fOutput_type::successor_list_type;
#endif
    using fInput_type::my_predecessors;

    //! Constructor
    // input_queue_type is allocated here, but destroyed in the function_input_base.
    // TODO: pass the graph_buffer_policy to the function_input_base so it can all
    // be done in one place.  This would be an interface-breaking change.
    template< typename Body >
    function_node( graph &g, size_t concurrency, Body body ) :
        graph_node(g), fInput_type(g, concurrency, body, allocate_buffer<Policy>::value ?
               new input_queue_type( ) : NULL ) {
        tbb::internal::fgt_node_with_body( tbb::internal::FLOW_FUNCTION_NODE, &this->graph_node::my_graph,
                static_cast<receiver<input_type> *>(this), static_cast<sender<output_type> *>(this), this->my_body );
    }

    //! Copy constructor
    function_node( const function_node& src ) :
        graph_node(src.graph_node::my_graph),
        fInput_type(src, allocate_buffer<Policy>::value ? new input_queue_type : NULL),
        fOutput_type() {
        tbb::internal::fgt_node_with_body( tbb::internal::FLOW_FUNCTION_NODE, &this->graph_node::my_graph,
                static_cast<receiver<input_type> *>(this), static_cast<sender<output_type> *>(this), this->my_body );
    }

#if TBB_PREVIEW_FLOW_GRAPH_TRACE
    /* override */ void set_name( const char *name ) {
        tbb::internal::fgt_node_desc( this, name );
    }
#endif

#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
    /*override*/void extract( ) {
        my_predecessors.built_predecessors().receiver_extract(*this);
        successors().built_successors().sender_extract(*this);
    }
#endif

protected:
    template< typename R, typename B > friend class run_and_put_task;
    template<typename X, typename Y> friend class internal::broadcast_cache;
    template<typename X, typename Y> friend class internal::round_robin_cache;
    using fInput_type::try_put_task;

    /* override */ internal::broadcast_cache<output_type> &successors () { return fOutput_type::my_successors; }

    // override of graph_node's reset.
    /*override*/void reset_node(reset_flags f) {
        fInput_type::reset_function_input(f);
        // TODO: use clear() instead.
        if(f & rf_clear_edges) { 
            successors().clear();
            my_predecessors.clear();
        }
        __TBB_ASSERT(!(f & rf_clear_edges) || successors().empty(), "function_node successors not empty");
        __TBB_ASSERT(this->my_predecessors.empty(), "function_node predecessors not empty");
    }

};  // class function_node


//! implements a function node that supports Input -> (set of outputs)
// Output is a tuple of output types.
template < typename Input, typename Output, typename Policy = queueing, typename Allocator=cache_aligned_allocator<Input> >
class multifunction_node :
    public graph_node,
    public internal::multifunction_input
    <
        Input,
        typename internal::wrap_tuple_elements<
            tbb::flow::tuple_size<Output>::value,  // #elements in tuple
            internal::multifunction_output,  // wrap this around each element
            Output // the tuple providing the types
        >::type,
        Allocator
    > {
protected:
    using graph_node::my_graph;
    static const int N = tbb::flow::tuple_size<Output>::value;
public:
    typedef Input input_type;
    typedef null_type output_type;
    typedef typename internal::wrap_tuple_elements<N,internal::multifunction_output, Output>::type output_ports_type;
    typedef internal::multifunction_input<input_type, output_ports_type, Allocator> fInput_type;
    typedef internal::function_input_queue<input_type, Allocator> input_queue_type;
private:
    typedef typename internal::multifunction_input<input_type, output_ports_type, Allocator> base_type;
    using fInput_type::my_predecessors;
public:
    template<typename Body>
    multifunction_node( graph &g, size_t concurrency, Body body ) :
        graph_node(g), base_type(g,concurrency, body,  allocate_buffer<Policy>::value ? new input_queue_type : NULL) {
        tbb::internal::fgt_multioutput_node_with_body<N>( tbb::internal::FLOW_MULTIFUNCTION_NODE,
                &this->graph_node::my_graph, static_cast<receiver<input_type> *>(this),
                this->output_ports(), this->my_body );
    }

    multifunction_node( const multifunction_node &other) :
        graph_node(other.graph_node::my_graph), base_type(other,  allocate_buffer<Policy>::value ? new input_queue_type : NULL) {
        tbb::internal::fgt_multioutput_node_with_body<N>( tbb::internal::FLOW_MULTIFUNCTION_NODE,
                &this->graph_node::my_graph, static_cast<receiver<input_type> *>(this),
                this->output_ports(), this->my_body );
    }

#if TBB_PREVIEW_FLOW_GRAPH_TRACE
    /* override */ void set_name( const char *name ) {
        tbb::internal::fgt_multioutput_node_desc( this, name );
    }
#endif

#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
    void extract( ) {
        my_predecessors.built_predecessors().receiver_extract(*this);
        base_type::extract();
    }
#endif
    // all the guts are in multifunction_input...
protected:
    /*override*/void reset_node(reset_flags f) { base_type::reset(f); }
};  // multifunction_node

//! split_node: accepts a tuple as input, forwards each element of the tuple to its
//  successors.  The node has unlimited concurrency, so though it is marked as
//  "rejecting" it does not reject inputs.
template<typename TupleType, typename Allocator=cache_aligned_allocator<TupleType> >
class split_node : public multifunction_node<TupleType, TupleType, rejecting, Allocator> {
    static const int N = tbb::flow::tuple_size<TupleType>::value;
    typedef multifunction_node<TupleType,TupleType,rejecting,Allocator> base_type;
public:
    typedef typename base_type::output_ports_type output_ports_type;
    typedef typename base_type::output_type output_type;
private:
    struct splitting_body {
        void operator()(const TupleType& t, output_ports_type &p) {
            internal::emit_element<N>::emit_this(t, p);
        }
    };
public:
    typedef TupleType input_type;
    typedef Allocator allocator_type;
    split_node(graph &g) : base_type(g, unlimited, splitting_body()) {
        tbb::internal::fgt_multioutput_node<N>( tbb::internal::FLOW_SPLIT_NODE, &this->graph_node::my_graph,
                                                          static_cast<receiver<input_type> *>(this), this->output_ports() );
    }

    split_node( const split_node & other) : base_type(other) {
        tbb::internal::fgt_multioutput_node<N>( tbb::internal::FLOW_SPLIT_NODE, &this->graph_node::my_graph,
                                                          static_cast<receiver<input_type> *>(this), this->output_ports() );
    }

#if TBB_PREVIEW_FLOW_GRAPH_TRACE
    /* override */ void set_name( const char *name ) {
        tbb::internal::fgt_multioutput_node_desc( this, name );
    }
#endif

};

//! Implements an executable node that supports continue_msg -> Output
template <typename Output>
class continue_node : public graph_node, public internal::continue_input<Output>, public internal::function_output<Output> {
protected:
    using graph_node::my_graph;
public:
    typedef continue_msg input_type;
    typedef Output output_type;
    typedef sender< input_type > predecessor_type;
    typedef receiver< output_type > successor_type;
    typedef internal::continue_input<Output> fInput_type;
    typedef internal::function_output<output_type> fOutput_type;

    //! Constructor for executable node with continue_msg -> Output
    template <typename Body >
    continue_node( graph &g, Body body ) :
        graph_node(g), internal::continue_input<output_type>( g, body ) {
        tbb::internal::fgt_node_with_body( tbb::internal::FLOW_CONTINUE_NODE, &this->my_graph,
                                           static_cast<receiver<input_type> *>(this),
                                           static_cast<sender<output_type> *>(this), this->my_body );
    }


    //! Constructor for executable node with continue_msg -> Output
    template <typename Body >
    continue_node( graph &g, int number_of_predecessors, Body body ) :
        graph_node(g), internal::continue_input<output_type>( g, number_of_predecessors, body ) {
        tbb::internal::fgt_node_with_body( tbb::internal::FLOW_CONTINUE_NODE, &this->my_graph,
                                           static_cast<receiver<input_type> *>(this),
                                           static_cast<sender<output_type> *>(this), this->my_body );
    }

    //! Copy constructor
    continue_node( const continue_node& src ) :
        graph_node(src.graph_node::my_graph), internal::continue_input<output_type>(src),
        internal::function_output<Output>() {
        tbb::internal::fgt_node_with_body( tbb::internal::FLOW_CONTINUE_NODE, &this->my_graph,
                                           static_cast<receiver<input_type> *>(this),
                                           static_cast<sender<output_type> *>(this), this->my_body );
    }

#if TBB_PREVIEW_FLOW_GRAPH_TRACE
    /* override */ void set_name( const char *name ) {
        tbb::internal::fgt_node_desc( this, name );
    }
#endif

#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
    /*override graph_node*/ void extract() {
        fInput_type::my_built_predecessors.receiver_extract(*this);
        successors().built_successors().sender_extract(*this);
    }
#endif

protected:
    template< typename R, typename B > friend class run_and_put_task;
    template<typename X, typename Y> friend class internal::broadcast_cache;
    template<typename X, typename Y> friend class internal::round_robin_cache;
    using fInput_type::try_put_task;
    /* override */ internal::broadcast_cache<output_type> &successors () { return fOutput_type::my_successors; }

    /*override*/void reset_node(reset_flags f) {
        fInput_type::reset_receiver(f);
        if(f & rf_clear_edges)successors().clear();
        __TBB_ASSERT(!(f & rf_clear_edges) || successors().empty(), "continue_node not reset");
    }

};  // continue_node

template< typename T >
class overwrite_node : public graph_node, public receiver<T>, public sender<T> {
protected:
    using graph_node::my_graph;
public:
    typedef T input_type;
    typedef T output_type;
    typedef sender< input_type > predecessor_type;
    typedef receiver< output_type > successor_type;
#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
    typedef typename receiver<input_type>::built_predecessors_type built_predecessors_type;
    typedef typename sender<output_type>::built_successors_type built_successors_type;
    typedef typename receiver<input_type>::predecessor_list_type predecessor_list_type;
    typedef typename sender<output_type>::successor_list_type successor_list_type;
#endif

    overwrite_node(graph &g) : graph_node(g), my_buffer_is_valid(false) {
        my_successors.set_owner( this );
        tbb::internal::fgt_node( tbb::internal::FLOW_OVERWRITE_NODE, &this->my_graph,
                                 static_cast<receiver<input_type> *>(this), static_cast<sender<output_type> *>(this) );
    }

    // Copy constructor; doesn't take anything from src; default won't work
    overwrite_node( const overwrite_node& src ) :
        graph_node(src.my_graph), receiver<T>(), sender<T>(), my_buffer_is_valid(false)
    {
        my_successors.set_owner( this );
        tbb::internal::fgt_node( tbb::internal::FLOW_OVERWRITE_NODE, &this->my_graph,
                                 static_cast<receiver<input_type> *>(this), static_cast<sender<output_type> *>(this) );
    }

    ~overwrite_node() {}

#if TBB_PREVIEW_FLOW_GRAPH_TRACE
    /* override */ void set_name( const char *name ) {
        tbb::internal::fgt_node_desc( this, name );
    }
#endif

    /* override */ bool register_successor( successor_type &s ) {
        spin_mutex::scoped_lock l( my_mutex );
        if (my_buffer_is_valid && this->my_graph.is_active()) {
            // We have a valid value that must be forwarded immediately.
            if ( s.try_put( my_buffer ) || !s.register_predecessor( *this  ) ) {
                // We add the successor: it accepted our put or it rejected it but won't let us become a predecessor
                my_successors.register_successor( s );
            } else {
                // We don't add the successor: it rejected our put and we became its predecessor instead
                return false;
            }
        } else {
            // No valid value yet, just add as successor
            my_successors.register_successor( s );
        }
        return true;
    }

    /* override */ bool remove_successor( successor_type &s ) {
        spin_mutex::scoped_lock l( my_mutex );
        my_successors.remove_successor(s);
        return true;
    }

#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
    /*override*/built_predecessors_type &built_predecessors() { return my_built_predecessors; }
    /*override*/built_successors_type   &built_successors()   { return my_successors.built_successors(); }

    /*override*/void internal_add_built_successor( successor_type &s) {
        spin_mutex::scoped_lock l( my_mutex );
        my_successors.internal_add_built_successor(s);
    }

    /*override*/void internal_delete_built_successor( successor_type &s) {
        spin_mutex::scoped_lock l( my_mutex );
        my_successors.internal_delete_built_successor(s);
    }

    /*override*/size_t successor_count() {
        spin_mutex::scoped_lock l( my_mutex );
        return my_successors.successor_count();
    }

    /*override*/ void copy_successors(successor_list_type &v) {
        spin_mutex::scoped_lock l( my_mutex );
        my_successors.copy_successors(v);
    }

    /*override*/ void internal_add_built_predecessor( predecessor_type &p) {
        spin_mutex::scoped_lock l( my_mutex );
        my_built_predecessors.add_edge(p);
    }

    /*override*/ void internal_delete_built_predecessor( predecessor_type &p) {
        spin_mutex::scoped_lock l( my_mutex );
        my_built_predecessors.delete_edge(p);
    }

    /*override*/size_t predecessor_count() {
        spin_mutex::scoped_lock l( my_mutex );
        return my_built_predecessors.edge_count();
    }

    /*override*/void copy_predecessors(predecessor_list_type &v) {
        spin_mutex::scoped_lock l( my_mutex );
        my_built_predecessors.copy_edges(v);
    }

    /*override*/ void extract() {
        my_buffer_is_valid = false;
        built_successors().sender_extract(*this);
        built_predecessors().receiver_extract(*this);
    }

#endif  /* TBB_PREVIEW_FLOW_GRAPH_FEATURES */

    /* override */ bool try_get( input_type &v ) {
        spin_mutex::scoped_lock l( my_mutex );
        if ( my_buffer_is_valid ) {
            v = my_buffer;
            return true;
        }
        return false;
    }

    bool is_valid() {
       spin_mutex::scoped_lock l( my_mutex );
       return my_buffer_is_valid;
    }

    void clear() {
       spin_mutex::scoped_lock l( my_mutex );
       my_buffer_is_valid = false;
    }

protected:
    template< typename R, typename B > friend class run_and_put_task;
    template<typename X, typename Y> friend class internal::broadcast_cache;
    template<typename X, typename Y> friend class internal::round_robin_cache;
    /* override */ task * try_put_task( const input_type &v ) {
        spin_mutex::scoped_lock l( my_mutex );
        my_buffer = v;
        my_buffer_is_valid = true;
        task * rtask = my_successors.try_put_task(v);
        if (!rtask) rtask = SUCCESSFULLY_ENQUEUED;
        return rtask;
    }

    spin_mutex my_mutex;
    internal::broadcast_cache< input_type, null_rw_mutex > my_successors;
#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
    internal::edge_container<predecessor_type> my_built_predecessors;
#endif
    input_type my_buffer;
    bool my_buffer_is_valid;
    /*override*/void reset_receiver(reset_flags /*f*/) {}

    /*override*/void reset_node( reset_flags f) {
        my_buffer_is_valid = false;
       if (f&rf_clear_edges) {
           my_successors.clear();
       }
    }
};  // overwrite_node

template< typename T >
class write_once_node : public overwrite_node<T> {
public:
    typedef T input_type;
    typedef T output_type;
    typedef sender< input_type > predecessor_type;
    typedef receiver< output_type > successor_type;

    //! Constructor
    write_once_node(graph& g) : overwrite_node<T>(g) {
        tbb::internal::fgt_node( tbb::internal::FLOW_WRITE_ONCE_NODE, &(this->my_graph),
                                 static_cast<receiver<input_type> *>(this),
                                 static_cast<sender<output_type> *>(this) );
    }

    //! Copy constructor: call base class copy constructor
    write_once_node( const write_once_node& src ) : overwrite_node<T>(src) {
        tbb::internal::fgt_node( tbb::internal::FLOW_WRITE_ONCE_NODE, &(this->my_graph),
                                 static_cast<receiver<input_type> *>(this),
                                 static_cast<sender<output_type> *>(this) );
    }

#if TBB_PREVIEW_FLOW_GRAPH_TRACE
    /* override */ void set_name( const char *name ) {
        tbb::internal::fgt_node_desc( this, name );
    }
#endif

protected:
    template< typename R, typename B > friend class run_and_put_task;
    template<typename X, typename Y> friend class internal::broadcast_cache;
    template<typename X, typename Y> friend class internal::round_robin_cache;
    /* override */ task *try_put_task( const T &v ) {
        spin_mutex::scoped_lock l( this->my_mutex );
        if ( this->my_buffer_is_valid ) {
            return NULL;
        } else {
            this->my_buffer = v;
            this->my_buffer_is_valid = true;
            task *res = this->my_successors.try_put_task(v);
            if (!res) res = SUCCESSFULLY_ENQUEUED;
            return res;
        }
    }
};

//! Forwards messages of type T to all successors
template <typename T>
class broadcast_node : public graph_node, public receiver<T>, public sender<T> {
protected:
    using graph_node::my_graph;
public:
    typedef T input_type;
    typedef T output_type;
    typedef sender< input_type > predecessor_type;
    typedef receiver< output_type > successor_type;
#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
    typedef typename receiver<input_type>::predecessor_list_type predecessor_list_type;
    typedef typename sender<output_type>::successor_list_type successor_list_type;
#endif
private:
    internal::broadcast_cache<input_type> my_successors;
#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
    internal::edge_container<predecessor_type> my_built_predecessors;
    spin_mutex pred_mutex;  // serialize accesses on edge_container
#endif
public:

    broadcast_node(graph& g) : graph_node(g) {
        my_successors.set_owner( this );
        tbb::internal::fgt_node( tbb::internal::FLOW_BROADCAST_NODE, &this->my_graph,
                                 static_cast<receiver<input_type> *>(this), static_cast<sender<output_type> *>(this) );
    }

    // Copy constructor
    broadcast_node( const broadcast_node& src ) :
        graph_node(src.my_graph), receiver<T>(), sender<T>()
    {
        my_successors.set_owner( this );
        tbb::internal::fgt_node( tbb::internal::FLOW_BROADCAST_NODE, &this->my_graph,
                                 static_cast<receiver<input_type> *>(this), static_cast<sender<output_type> *>(this) );
    }

#if TBB_PREVIEW_FLOW_GRAPH_TRACE
    /* override */ void set_name( const char *name ) {
        tbb::internal::fgt_node_desc( this, name );
    }
#endif

    //! Adds a successor
    virtual bool register_successor( receiver<T> &r ) {
        my_successors.register_successor( r );
        return true;
    }

    //! Removes s as a successor
    virtual bool remove_successor( receiver<T> &r ) {
        my_successors.remove_successor( r );
        return true;
    }

#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
    typedef typename sender<T>::built_successors_type built_successors_type;

    /*override sender*/ built_successors_type &built_successors() { return my_successors.built_successors(); }

    /*override sender*/ void internal_add_built_successor(successor_type &r) {
        my_successors.internal_add_built_successor(r);
    }

    /*override sender*/ void internal_delete_built_successor(successor_type &r) {
        my_successors.internal_delete_built_successor(r);
    }

    /*override sender*/ size_t successor_count() {
        return my_successors.successor_count();
    }

    /*override*/ void copy_successors(successor_list_type &v) {
        my_successors.copy_successors(v);
    }

    typedef typename receiver<T>::built_predecessors_type built_predecessors_type;

    /*override receiver*/ built_predecessors_type &built_predecessors() { return my_built_predecessors; } 

    /*override*/ void internal_add_built_predecessor( predecessor_type &p) {
        spin_mutex::scoped_lock l(pred_mutex);
        my_built_predecessors.add_edge(p);
    }

    /*override*/ void internal_delete_built_predecessor( predecessor_type &p) {
        spin_mutex::scoped_lock l(pred_mutex);
        my_built_predecessors.delete_edge(p);
    }

    /*override*/ size_t predecessor_count() {
        spin_mutex::scoped_lock l(pred_mutex);
        return my_built_predecessors.edge_count();
    }

    /*override*/ void copy_predecessors(predecessor_list_type &v) {
        spin_mutex::scoped_lock l(pred_mutex);
        my_built_predecessors.copy_edges(v);
    }

    /*override graph_node*/ void extract() {
        my_built_predecessors.receiver_extract(*this);
        my_successors.built_successors().sender_extract(*this);
    }
#endif  /* TBB_PREVIEW_FLOW_GRAPH_FEATURES */

protected:
    template< typename R, typename B > friend class run_and_put_task;
    template<typename X, typename Y> friend class internal::broadcast_cache;
    template<typename X, typename Y> friend class internal::round_robin_cache;
    //! build a task to run the successor if possible.  Default is old behavior.
    /*override*/ task *try_put_task(const T& t) {
        task *new_task = my_successors.try_put_task(t);
        if (!new_task) new_task = SUCCESSFULLY_ENQUEUED;
        return new_task;
    }

    /*override*/void reset_receiver(reset_flags /*f*/) {}

    /*override*/void reset_node(reset_flags f) {
        if (f&rf_clear_edges) {
           my_successors.clear();
#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
           my_built_predecessors.clear();
#endif
        }
        __TBB_ASSERT(!(f & rf_clear_edges) || my_successors.empty(), "Error resetting broadcast_node");
    }
};  // broadcast_node

//! Forwards messages in arbitrary order
template <typename T, typename A=cache_aligned_allocator<T> >
class buffer_node : public graph_node, public internal::reservable_item_buffer<T, A>, public receiver<T>, public sender<T> {
protected:
    using graph_node::my_graph;
public:
    typedef T input_type;
    typedef T output_type;
    typedef sender< input_type > predecessor_type;
    typedef receiver< output_type > successor_type;
    typedef buffer_node<T, A> class_type;
#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
    typedef typename receiver<input_type>::predecessor_list_type predecessor_list_type;
    typedef typename sender<output_type>::successor_list_type successor_list_type;
#endif
protected:
    typedef size_t size_type;
    internal::round_robin_cache< T, null_rw_mutex > my_successors;

#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
    internal::edge_container<predecessor_type> my_built_predecessors;
#endif

    friend class internal::forward_task_bypass< buffer_node< T, A > >;

    enum op_type {reg_succ, rem_succ, req_item, res_item, rel_res, con_res, put_item, try_fwd_task
#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
        , add_blt_succ, del_blt_succ,
        add_blt_pred, del_blt_pred,
        blt_succ_cnt, blt_pred_cnt,
        blt_succ_cpy, blt_pred_cpy   // create vector copies of preds and succs
#endif
    };
    enum op_stat {WAIT=0, SUCCEEDED, FAILED};

    // implements the aggregator_operation concept
    class buffer_operation : public internal::aggregated_operation< buffer_operation > {
    public:
        char type;
#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
        task * ltask;
        union {
            input_type *elem;
            successor_type *r;
            predecessor_type *p;
            size_t cnt_val;
            successor_list_type *svec;
            predecessor_list_type *pvec;
        };
#else
        T *elem;
        task * ltask;
        successor_type *r;
#endif
        buffer_operation(const T& e, op_type t) : type(char(t))

#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
                                                  , ltask(NULL), elem(const_cast<T*>(&e))
#else
                                                  , elem(const_cast<T*>(&e)) , ltask(NULL)
#endif
        {}
        buffer_operation(op_type t) : type(char(t)),  ltask(NULL) {}
    };

    bool forwarder_busy;
    typedef internal::aggregating_functor<class_type, buffer_operation> handler_type;
    friend class internal::aggregating_functor<class_type, buffer_operation>;
    internal::aggregator< handler_type, buffer_operation> my_aggregator;

    virtual void handle_operations(buffer_operation *op_list) {
        buffer_operation *tmp = NULL;
        bool try_forwarding=false;
        while (op_list) {
            tmp = op_list;
            op_list = op_list->next;
            switch (tmp->type) {
            case reg_succ: internal_reg_succ(tmp);  try_forwarding = true; break;
            case rem_succ: internal_rem_succ(tmp); break;
            case req_item: internal_pop(tmp); break;
            case res_item: internal_reserve(tmp); break;
            case rel_res:  internal_release(tmp);  try_forwarding = true; break;
            case con_res:  internal_consume(tmp);  try_forwarding = true; break;
            case put_item: internal_push(tmp);  try_forwarding = (tmp->status == SUCCEEDED); break;
            case try_fwd_task: internal_forward_task(tmp); break;
#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
            // edge recording
            case add_blt_succ: internal_add_built_succ(tmp); break;
            case del_blt_succ: internal_del_built_succ(tmp); break;
            case add_blt_pred: internal_add_built_pred(tmp); break;
            case del_blt_pred: internal_del_built_pred(tmp); break;
            case blt_succ_cnt: internal_succ_cnt(tmp); break;
            case blt_pred_cnt: internal_pred_cnt(tmp); break;
            case blt_succ_cpy: internal_copy_succs(tmp); break;
            case blt_pred_cpy: internal_copy_preds(tmp); break;
#endif
            }
        }
        if (try_forwarding && !forwarder_busy) {
            if(this->my_graph.is_active()) {
                forwarder_busy = true;
                task *new_task = new(task::allocate_additional_child_of(*(this->my_graph.root_task()))) internal::
                        forward_task_bypass
                        < buffer_node<input_type, A> >(*this);
                // tmp should point to the last item handled by the aggregator.  This is the operation
                // the handling thread enqueued.  So modifying that record will be okay.
                tbb::task *z = tmp->ltask;
                tmp->ltask = combine_tasks(z, new_task);  // in case the op generated a task
            }
        }
    }  // handle_operations

    inline task *grab_forwarding_task( buffer_operation &op_data) {
        return op_data.ltask;
    }

    inline bool enqueue_forwarding_task(buffer_operation &op_data) {
        task *ft = grab_forwarding_task(op_data);
        if(ft) {
            FLOW_SPAWN(*ft);
            return true;
        }
        return false;
    }

    //! This is executed by an enqueued task, the "forwarder"
    virtual task *forward_task() {
        buffer_operation op_data(try_fwd_task);
        task *last_task = NULL;
        do {
            op_data.status = WAIT;
            op_data.ltask = NULL;
            my_aggregator.execute(&op_data);
            tbb::task *xtask = op_data.ltask;
            last_task = combine_tasks(last_task, xtask);
        } while (op_data.status == SUCCEEDED);
        return last_task;
    }

    //! Register successor
    virtual void internal_reg_succ(buffer_operation *op) {
        my_successors.register_successor(*(op->r));
        __TBB_store_with_release(op->status, SUCCEEDED);
    }

    //! Remove successor
    virtual void internal_rem_succ(buffer_operation *op) {
        my_successors.remove_successor(*(op->r));
        __TBB_store_with_release(op->status, SUCCEEDED);
    }

#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
    typedef typename sender<T>::built_successors_type built_successors_type;

    /*override sender*/ built_successors_type &built_successors() { return my_successors.built_successors(); }

    virtual void internal_add_built_succ(buffer_operation *op) {
        my_successors.internal_add_built_successor(*(op->r));
        __TBB_store_with_release(op->status, SUCCEEDED);
    }

    virtual void internal_del_built_succ(buffer_operation *op) {
        my_successors.internal_delete_built_successor(*(op->r));
        __TBB_store_with_release(op->status, SUCCEEDED);
    }

    typedef typename receiver<T>::built_predecessors_type built_predecessors_type;

    /*override receiver*/ built_predecessors_type &built_predecessors() { return my_built_predecessors; } 

    virtual void internal_add_built_pred(buffer_operation *op) {
        my_built_predecessors.add_edge(*(op->p));
        __TBB_store_with_release(op->status, SUCCEEDED);
    }

    virtual void internal_del_built_pred(buffer_operation *op) {
        my_built_predecessors.delete_edge(*(op->p));
        __TBB_store_with_release(op->status, SUCCEEDED);
    }

    virtual void internal_succ_cnt(buffer_operation *op) {
        op->cnt_val = my_successors.successor_count();
        __TBB_store_with_release(op->status, SUCCEEDED);
    }

    virtual void internal_pred_cnt(buffer_operation *op) {
        op->cnt_val = my_built_predecessors.edge_count();
        __TBB_store_with_release(op->status, SUCCEEDED);
    }

    virtual void internal_copy_succs(buffer_operation *op) {
        my_successors.copy_successors(*(op->svec));
        __TBB_store_with_release(op->status, SUCCEEDED);
    }

    virtual void internal_copy_preds(buffer_operation *op) {
        my_built_predecessors.copy_edges(*(op->pvec));
        __TBB_store_with_release(op->status, SUCCEEDED);
    }

#endif  /* TBB_PREVIEW_FLOW_GRAPH_FEATURES */

    //! Tries to forward valid items to successors
    virtual void internal_forward_task(buffer_operation *op) {
        if (this->my_reserved || !this->my_item_valid(this->my_tail-1)) {
            __TBB_store_with_release(op->status, FAILED);
            this->forwarder_busy = false;
            return;
        }
        T i_copy;
        task * last_task = NULL;
        size_type counter = my_successors.size();
        // Try forwarding, giving each successor a chance
        while (counter>0 && !this->buffer_empty() && this->my_item_valid(this->my_tail-1)) {
            this->copy_back(i_copy);
            task *new_task = my_successors.try_put_task(i_copy);
            if(new_task) {
                last_task = combine_tasks(last_task, new_task);
                this->destroy_back();
            }
            --counter;
        }
        op->ltask = last_task;  // return task
        if (last_task && !counter) {
            __TBB_store_with_release(op->status, SUCCEEDED);
        }
        else {
            __TBB_store_with_release(op->status, FAILED);
            forwarder_busy = false;
        }
    }

    virtual void internal_push(buffer_operation *op) {
        this->push_back(*(op->elem));
        __TBB_store_with_release(op->status, SUCCEEDED);
    }

    virtual void internal_pop(buffer_operation *op) {
        if(this->pop_back(*(op->elem))) {
            __TBB_store_with_release(op->status, SUCCEEDED);
        }
        else {
            __TBB_store_with_release(op->status, FAILED);
        }
    }

    virtual void internal_reserve(buffer_operation *op) {
        if(this->reserve_front(*(op->elem))) {
            __TBB_store_with_release(op->status, SUCCEEDED);
        }
        else {
            __TBB_store_with_release(op->status, FAILED);
        }
    }

    virtual void internal_consume(buffer_operation *op) {
        this->consume_front();
        __TBB_store_with_release(op->status, SUCCEEDED);
    }

    virtual void internal_release(buffer_operation *op) {
        this->release_front();
        __TBB_store_with_release(op->status, SUCCEEDED);
    }

public:
    //! Constructor
    buffer_node( graph &g ) : graph_node(g), internal::reservable_item_buffer<T>(),
        forwarder_busy(false) {
        my_successors.set_owner(this);
        my_aggregator.initialize_handler(handler_type(this));
        tbb::internal::fgt_node( tbb::internal::FLOW_BUFFER_NODE, &this->my_graph,
                                 static_cast<receiver<input_type> *>(this), static_cast<sender<output_type> *>(this) );
    }

    //! Copy constructor
    buffer_node( const buffer_node& src ) : graph_node(src.my_graph),
        internal::reservable_item_buffer<T>(), receiver<T>(), sender<T>() {
        forwarder_busy = false;
        my_successors.set_owner(this);
        my_aggregator.initialize_handler(handler_type(this));
        tbb::internal::fgt_node( tbb::internal::FLOW_BUFFER_NODE, &this->my_graph,
                                 static_cast<receiver<input_type> *>(this), static_cast<sender<output_type> *>(this) );
    }

    virtual ~buffer_node() {}

#if TBB_PREVIEW_FLOW_GRAPH_TRACE
    /* override */ void set_name( const char *name ) {
        tbb::internal::fgt_node_desc( this, name );
    }
#endif

    //
    // message sender implementation
    //

    //! Adds a new successor.
    /** Adds successor r to the list of successors; may forward tasks.  */
    /* override */ bool register_successor( successor_type &r ) {
        buffer_operation op_data(reg_succ);
        op_data.r = &r;
        my_aggregator.execute(&op_data);
        (void)enqueue_forwarding_task(op_data);
        return true;
    }

#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
    /*override*/ void internal_add_built_successor( successor_type &r) {
        buffer_operation op_data(add_blt_succ);
        op_data.r = &r;
        my_aggregator.execute(&op_data);
    }

    /*override*/ void internal_delete_built_successor( successor_type &r) {
        buffer_operation op_data(del_blt_succ);
        op_data.r = &r;
        my_aggregator.execute(&op_data);
    }

    /*override*/ void internal_add_built_predecessor( predecessor_type &p) {
        buffer_operation op_data(add_blt_pred);
        op_data.p = &p;
        my_aggregator.execute(&op_data);
    }

    /*override*/ void internal_delete_built_predecessor( predecessor_type &p) {
        buffer_operation op_data(del_blt_pred);
        op_data.p = &p;
        my_aggregator.execute(&op_data);
    }

    /*override*/ size_t predecessor_count() {
        buffer_operation op_data(blt_pred_cnt);
        my_aggregator.execute(&op_data);
        return op_data.cnt_val;
    }

    /*override*/ size_t successor_count() {
        buffer_operation op_data(blt_succ_cnt);
        my_aggregator.execute(&op_data);
        return op_data.cnt_val;
    }

    /*override*/ void copy_predecessors( predecessor_list_type &v ) {
        buffer_operation op_data(blt_pred_cpy);
        op_data.pvec = &v;
        my_aggregator.execute(&op_data);
    }

    /*override*/ void copy_successors( successor_list_type &v ) {
        buffer_operation op_data(blt_succ_cpy);
        op_data.svec = &v;
        my_aggregator.execute(&op_data);
    }

#endif

    //! Removes a successor.
    /** Removes successor r from the list of successors.
        It also calls r.remove_predecessor(*this) to remove this node as a predecessor. */
    /* override */ bool remove_successor( successor_type &r ) {
        r.remove_predecessor(*this);
        buffer_operation op_data(rem_succ);
        op_data.r = &r;
        my_aggregator.execute(&op_data);
        // even though this operation does not cause a forward, if we are the handler, and
        // a forward is scheduled, we may be the first to reach this point after the aggregator,
        // and so should check for the task.
        (void)enqueue_forwarding_task(op_data);
        return true;
    }

    //! Request an item from the buffer_node
    /**  true = v contains the returned item<BR>
         false = no item has been returned */
    /* override */ bool try_get( T &v ) {
        buffer_operation op_data(req_item);
        op_data.elem = &v;
        my_aggregator.execute(&op_data);
        (void)enqueue_forwarding_task(op_data);
        return (op_data.status==SUCCEEDED);
    }

    //! Reserves an item.
    /**  false = no item can be reserved<BR>
         true = an item is reserved */
    /* override */ bool try_reserve( T &v ) {
        buffer_operation op_data(res_item);
        op_data.elem = &v;
        my_aggregator.execute(&op_data);
        (void)enqueue_forwarding_task(op_data);
        return (op_data.status==SUCCEEDED);
    }

    //! Release a reserved item.
    /**  true = item has been released and so remains in sender */
    /* override */ bool try_release() {
        buffer_operation op_data(rel_res);
        my_aggregator.execute(&op_data);
        (void)enqueue_forwarding_task(op_data);
        return true;
    }

    //! Consumes a reserved item.
    /** true = item is removed from sender and reservation removed */
    /* override */ bool try_consume() {
        buffer_operation op_data(con_res);
        my_aggregator.execute(&op_data);
        (void)enqueue_forwarding_task(op_data);
        return true;
    }

protected:

    template< typename R, typename B > friend class run_and_put_task;
    template<typename X, typename Y> friend class internal::broadcast_cache;
    template<typename X, typename Y> friend class internal::round_robin_cache;
    //! receive an item, return a task *if possible
    /* override */ task *try_put_task(const T &t) {
        buffer_operation op_data(t, put_item);
        my_aggregator.execute(&op_data);
        task *ft = grab_forwarding_task(op_data);
        // sequencer_nodes can return failure (if an item has been previously inserted)
        // We have to spawn the returned task if our own operation fails.

        if(ft && op_data.status == FAILED) {
            // we haven't succeeded queueing the item, but for some reason the
            // call returned a task (if another request resulted in a successful
            // forward this could happen.)  Queue the task and reset the pointer.
            FLOW_SPAWN(*ft); ft = NULL;
        }
        else if(!ft && op_data.status == SUCCEEDED) {
            ft = SUCCESSFULLY_ENQUEUED;
        }
        return ft;
    }

    /*override*/void reset_receiver(reset_flags /*f*/) { }

#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
public:
    /* override*/ void extract() {
        my_built_predecessors.receiver_extract(*this);
        my_successors.built_successors().sender_extract(*this);
    }
#endif

protected:
    /*override*/void reset_node( reset_flags f) {
        internal::reservable_item_buffer<T, A>::reset();
        // TODO: just clear structures
        if (f&rf_clear_edges) {
            my_successors.clear();
#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
            my_built_predecessors.clear();
#endif
        }
        forwarder_busy = false;
    }


};  // buffer_node

//! Forwards messages in FIFO order
template <typename T, typename A=cache_aligned_allocator<T> >
class queue_node : public buffer_node<T, A> {
protected:
    typedef buffer_node<T, A> base_type;
    typedef typename base_type::size_type size_type;
    typedef typename base_type::buffer_operation queue_operation;

    enum op_stat {WAIT=0, SUCCEEDED, FAILED};

    /* override */ void internal_forward_task(queue_operation *op) {
        if (this->my_reserved || !this->my_item_valid(this->my_head)) {
            __TBB_store_with_release(op->status, FAILED);
            this->forwarder_busy = false;
            return;
        }
        T i_copy;
        task *last_task = NULL;
        size_type counter = this->my_successors.size();
        // Keep trying to send items while there is at least one accepting successor
        while (counter>0 && this->my_item_valid(this->my_head)) {
            this->copy_front(i_copy);
            task *new_task = this->my_successors.try_put_task(i_copy);
            if(new_task) {
                this->destroy_front();
                last_task = combine_tasks(last_task, new_task);
            }
            --counter;
        }
        op->ltask = last_task;
        if (last_task && !counter)
            __TBB_store_with_release(op->status, SUCCEEDED);
        else {
            __TBB_store_with_release(op->status, FAILED);
            this->forwarder_busy = false;
        }
    }

    /* override */ void internal_pop(queue_operation *op) {
        if ( this->my_reserved || !this->my_item_valid(this->my_head)){
            __TBB_store_with_release(op->status, FAILED);
        }
        else {
            this->pop_front(*(op->elem));
            __TBB_store_with_release(op->status, SUCCEEDED);
        }
    }
    /* override */ void internal_reserve(queue_operation *op) {
        if (this->my_reserved || !this->my_item_valid(this->my_head)) {
            __TBB_store_with_release(op->status, FAILED);
        }
        else {
            this->reserve_front(*(op->elem));
            __TBB_store_with_release(op->status, SUCCEEDED);
        }
    }
    /* override */ void internal_consume(queue_operation *op) {
        this->consume_front();
        __TBB_store_with_release(op->status, SUCCEEDED);
    }

public:
    typedef T input_type;
    typedef T output_type;
    typedef sender< input_type > predecessor_type;
    typedef receiver< output_type > successor_type;

    //! Constructor
    queue_node( graph &g ) : base_type(g) {
        tbb::internal::fgt_node( tbb::internal::FLOW_QUEUE_NODE, &(this->my_graph),
                                 static_cast<receiver<input_type> *>(this),
                                 static_cast<sender<output_type> *>(this) );
    }

    //! Copy constructor
    queue_node( const queue_node& src) : base_type(src) {
        tbb::internal::fgt_node( tbb::internal::FLOW_QUEUE_NODE, &(this->my_graph),
                                 static_cast<receiver<input_type> *>(this),
                                 static_cast<sender<output_type> *>(this) );
    }

#if TBB_PREVIEW_FLOW_GRAPH_TRACE
    /* override */ void set_name( const char *name ) {
        tbb::internal::fgt_node_desc( this, name );
    }
#endif

protected:
    /*override*/void reset_node( reset_flags f) {
        base_type::reset_node(f);
    }
};  // queue_node

//! Forwards messages in sequence order
template< typename T, typename A=cache_aligned_allocator<T> >
class sequencer_node : public queue_node<T, A> {
    internal::function_body< T, size_t > *my_sequencer;
    // my_sequencer should be a benign function and must be callable
    // from a parallel context.  Does this mean it needn't be reset?
public:
    typedef T input_type;
    typedef T output_type;
    typedef sender< input_type > predecessor_type;
    typedef receiver< output_type > successor_type;

    //! Constructor
    template< typename Sequencer >
    sequencer_node( graph &g, const Sequencer& s ) : queue_node<T, A>(g),
        my_sequencer(new internal::function_body_leaf< T, size_t, Sequencer>(s) ) {
        tbb::internal::fgt_node( tbb::internal::FLOW_SEQUENCER_NODE, &(this->my_graph),
                                 static_cast<receiver<input_type> *>(this),
                                 static_cast<sender<output_type> *>(this) );
    }

    //! Copy constructor
    sequencer_node( const sequencer_node& src ) : queue_node<T, A>(src),
        my_sequencer( src.my_sequencer->clone() ) {
        tbb::internal::fgt_node( tbb::internal::FLOW_SEQUENCER_NODE, &(this->my_graph),
                                 static_cast<receiver<input_type> *>(this),
                                 static_cast<sender<output_type> *>(this) );
    }

    //! Destructor
    ~sequencer_node() { delete my_sequencer; }

#if TBB_PREVIEW_FLOW_GRAPH_TRACE
    /* override */ void set_name( const char *name ) {
        tbb::internal::fgt_node_desc( this, name );
    }
#endif

protected:
    typedef typename buffer_node<T, A>::size_type size_type;
    typedef typename buffer_node<T, A>::buffer_operation sequencer_operation;

    enum op_stat {WAIT=0, SUCCEEDED, FAILED};

private:
    /* override */ void internal_push(sequencer_operation *op) {
        size_type tag = (*my_sequencer)(*(op->elem));
#if !TBB_DEPRECATED_SEQUENCER_DUPLICATES
        if(tag < this->my_head) {
            // have already emitted a message with this tag
            __TBB_store_with_release(op->status, FAILED);
            return;
        }
#endif
        // cannot modify this->my_tail now; the buffer would be inconsistent.
        size_t new_tail = (tag+1 > this->my_tail) ? tag+1 : this->my_tail;

        if(this->size(new_tail) > this->capacity()) {
            this->grow_my_array(this->size(new_tail));
        }
        this->my_tail = new_tail;
        if(this->place_item(tag,*(op->elem))) {
            __TBB_store_with_release(op->status, SUCCEEDED);
        }
        else {
            // already have a message with this tag
            __TBB_store_with_release(op->status, FAILED);
        }
    }
};  // sequencer_node

//! Forwards messages in priority order
template< typename T, typename Compare = std::less<T>, typename A=cache_aligned_allocator<T> >
class priority_queue_node : public buffer_node<T, A> {
public:
    typedef T input_type;
    typedef T output_type;
    typedef buffer_node<T,A> base_type;
    typedef sender< input_type > predecessor_type;
    typedef receiver< output_type > successor_type;

    //! Constructor
    priority_queue_node( graph &g ) : buffer_node<T, A>(g), mark(0) {
        tbb::internal::fgt_node( tbb::internal::FLOW_PRIORITY_QUEUE_NODE, &(this->my_graph),
                                 static_cast<receiver<input_type> *>(this),
                                 static_cast<sender<output_type> *>(this) );
    }

    //! Copy constructor
    priority_queue_node( const priority_queue_node &src ) : buffer_node<T, A>(src), mark(0) {
        tbb::internal::fgt_node( tbb::internal::FLOW_PRIORITY_QUEUE_NODE, &(this->my_graph),
                                 static_cast<receiver<input_type> *>(this),
                                 static_cast<sender<output_type> *>(this) );
    }

#if TBB_PREVIEW_FLOW_GRAPH_TRACE
    /* override */ void set_name( const char *name ) {
        tbb::internal::fgt_node_desc( this, name );
    }
#endif


protected:

    /*override*/void reset_node( reset_flags f) {
        mark = 0;
        base_type::reset_node(f);
    }

    typedef typename buffer_node<T, A>::size_type size_type;
    typedef typename buffer_node<T, A>::item_type item_type;
    typedef typename buffer_node<T, A>::buffer_operation prio_operation;

    enum op_stat {WAIT=0, SUCCEEDED, FAILED};

    /* override */ void handle_operations(prio_operation *op_list) {
        prio_operation *tmp = op_list /*, *pop_list*/ ;
        bool try_forwarding=false;
        while (op_list) {
            tmp = op_list;
            op_list = op_list->next;
            switch (tmp->type) {
            case buffer_node<T, A>::reg_succ: this->internal_reg_succ(tmp); try_forwarding = true; break;
            case buffer_node<T, A>::rem_succ: this->internal_rem_succ(tmp); break;
            case buffer_node<T, A>::put_item: internal_push(tmp); try_forwarding = true; break;
            case buffer_node<T, A>::try_fwd_task: internal_forward_task(tmp); break;
            case buffer_node<T, A>::rel_res: internal_release(tmp); try_forwarding = true; break;
            case buffer_node<T, A>::con_res: internal_consume(tmp); try_forwarding = true; break;
            case buffer_node<T, A>::req_item: internal_pop(tmp); break;
            case buffer_node<T, A>::res_item: internal_reserve(tmp); break;
#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
            case buffer_node<T, A>::add_blt_succ: this->internal_add_built_succ(tmp); break;
            case buffer_node<T, A>::del_blt_succ: this->internal_del_built_succ(tmp); break;
            case buffer_node<T, A>::add_blt_pred: this->internal_add_built_pred(tmp); break;
            case buffer_node<T, A>::del_blt_pred: this->internal_del_built_pred(tmp); break;
            case buffer_node<T, A>::blt_succ_cnt: this->internal_succ_cnt(tmp); break;
            case buffer_node<T, A>::blt_pred_cnt: this->internal_pred_cnt(tmp); break;
            case buffer_node<T, A>::blt_succ_cpy: this->internal_copy_succs(tmp); break;
            case buffer_node<T, A>::blt_pred_cpy: this->internal_copy_preds(tmp); break;
#endif
            }
        }
        // process pops!  for now, no special pop processing
        // concurrent_priority_queue handles pushes first, then pops.
        // that is the genesis of this comment
        if (mark<this->my_tail) heapify();
        __TBB_ASSERT(mark == this->my_tail, "mark unequal after heapify");
        if (try_forwarding && !this->forwarder_busy) {  // could we also test for this->my_tail (queue non-empty)?
            if(this->my_graph.is_active()) {
                this->forwarder_busy = true;
                task *new_task = new(task::allocate_additional_child_of(*(this->my_graph.root_task()))) internal::
                        forward_task_bypass
                        < buffer_node<input_type, A> >(*this);
                // tmp should point to the last item handled by the aggregator.  This is the operation
                // the handling thread enqueued.  So modifying that record will be okay.
                tbb::task *tmp1 = tmp->ltask;
                tmp->ltask = combine_tasks(tmp1, new_task);
            }
        }
    }

    //! Tries to forward valid items to successors
    /* override */ void internal_forward_task(prio_operation *op) {
        T i_copy;
        task * last_task = NULL; // flagged when a successor accepts
        size_type counter = this->my_successors.size();

        if (this->my_reserved || this->my_tail == 0) {
            __TBB_store_with_release(op->status, FAILED);
            this->forwarder_busy = false;
            return;
        }
        // Keep trying to send while there exists an accepting successor
        while (counter>0 && this->my_tail > 0) {
            prio_copy(i_copy);
            task * new_task = this->my_successors.try_put_task(i_copy);
            if ( new_task ) {
                last_task = combine_tasks(last_task, new_task);
                prio_pop();
            }
            --counter;
        }
        op->ltask = last_task;
        if (last_task && !counter)
            __TBB_store_with_release(op->status, SUCCEEDED);
        else {
            __TBB_store_with_release(op->status, FAILED);
            this->forwarder_busy = false;
        }
    }

    /* override */ void internal_push(prio_operation *op) {
        prio_push(*(op->elem));
        __TBB_store_with_release(op->status, SUCCEEDED);
    }

    /* override */ void internal_pop(prio_operation *op) {
        // if empty or already reserved, don't pop
        if ( this->my_reserved == true || this->my_tail == 0 ) {
            __TBB_store_with_release(op->status, FAILED);
            return;
        }

        prio_copy(*(op->elem));
        __TBB_store_with_release(op->status, SUCCEEDED);
        prio_pop();

    }

    // pops the highest-priority item, saves copy
    /* override */ void internal_reserve(prio_operation *op) {
        if (this->my_reserved == true || this->my_tail == 0) {
            __TBB_store_with_release(op->status, FAILED);
            return;
        }
        this->my_reserved = true;
        prio_copy(*(op->elem));
        reserved_item = *(op->elem);
        __TBB_store_with_release(op->status, SUCCEEDED);
        prio_pop();
    }

    /* override */ void internal_consume(prio_operation *op) {
        __TBB_store_with_release(op->status, SUCCEEDED);
        this->my_reserved = false;
        reserved_item = input_type();
    }

    /* override */ void internal_release(prio_operation *op) {
        __TBB_store_with_release(op->status, SUCCEEDED);
        prio_push(reserved_item);
        this->my_reserved = false;
        reserved_item = input_type();
    }
private:
    Compare compare;
    size_type mark;

    input_type reserved_item;

    // in case a reheap has not been done after a push, check if the mark item is higher than the 0'th item
    bool prio_use_tail() {
        __TBB_ASSERT(mark <= this->my_tail, "mark outside bounds before test");
        return mark < this->my_tail && compare(this->get_my_item(0), this->get_my_item(this->my_tail - 1));
    }

    // prio_push: checks that the item will fit, expand array if necessary, put at end
    void prio_push(const T &src) {
        if ( this->my_tail >= this->my_array_size )
            this->grow_my_array( this->my_tail + 1 );
        (void) this->place_item(this->my_tail, src);
        ++(this->my_tail);
        __TBB_ASSERT(mark < this->my_tail, "mark outside bounds after push");
    }

    // prio_pop: deletes highest priority item from the array, and if it is item
    // 0, move last item to 0 and reheap.  If end of array, just destroy and decrement tail
    // and mark.  Assumes the array has already been tested for emptiness; no failure.
    void prio_pop()  {
        if (prio_use_tail()) {
            // there are newly pushed elems; last one higher than top
            // copy the data
            this->destroy_item(this->my_tail-1);
            --(this->my_tail);
            __TBB_ASSERT(mark <= this->my_tail, "mark outside bounds after pop");
            return;
        }
        this->destroy_item(0);
        if(this->my_tail > 1) {
            // push the last element down heap
            __TBB_ASSERT(this->my_item_valid(this->my_tail - 1), NULL);
            this->move_item(0,this->my_tail - 1);
        }
        --(this->my_tail);
        if(mark > this->my_tail) --mark;
        if (this->my_tail > 1) // don't reheap for heap of size 1
            reheap();
        __TBB_ASSERT(mark <= this->my_tail, "mark outside bounds after pop");
    }

    void prio_copy(T &res) {
        if (prio_use_tail()) {
            res = this->get_my_item(this->my_tail - 1);
        }
        else {
            res = this->get_my_item(0);
        }
    }

    // turn array into heap
    void heapify() {
        if(this->my_tail == 0) {
            mark = 0;
            return;
        }
        if (!mark) mark = 1;
        for (; mark<this->my_tail; ++mark) { // for each unheaped element
            size_type cur_pos = mark;
            input_type to_place;
            this->fetch_item(mark,to_place);
            do { // push to_place up the heap
                size_type parent = (cur_pos-1)>>1;
                if (!compare(this->get_my_item(parent), to_place))
                    break;
                this->move_item(cur_pos, parent);
                cur_pos = parent;
            } while( cur_pos );
            (void) this->place_item(cur_pos, to_place);
        }
    }

    // otherwise heapified array with new root element; rearrange to heap
    void reheap() {
        size_type cur_pos=0, child=1;
        while (child < mark) {
            size_type target = child;
            if (child+1<mark &&
                compare(this->get_my_item(child),
                        this->get_my_item(child+1)))
                ++target;
            // target now has the higher priority child
            if (compare(this->get_my_item(target),
                        this->get_my_item(cur_pos)))
                break;
            // swap
            this->swap_items(cur_pos, target);
            cur_pos = target;
            child = (cur_pos<<1)+1;
        }
    }
};  // priority_queue_node

//! Forwards messages only if the threshold has not been reached
/** This node forwards items until its threshold is reached.
    It contains no buffering.  If the downstream node rejects, the
    message is dropped. */
template< typename T >
class limiter_node : public graph_node, public receiver< T >, public sender< T > {
protected:
    using graph_node::my_graph;
public:
    typedef T input_type;
    typedef T output_type;
    typedef sender< input_type > predecessor_type;
    typedef receiver< output_type > successor_type;
#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
    typedef typename receiver<input_type>::built_predecessors_type built_predecessors_type;
    typedef typename sender<output_type>::built_successors_type built_successors_type;
    typedef typename receiver<input_type>::predecessor_list_type predecessor_list_type;
    typedef typename sender<output_type>::successor_list_type successor_list_type;
#endif

private:
    size_t my_threshold;
    size_t my_count; //number of successful puts
    size_t my_tries; //number of active put attempts
    internal::reservable_predecessor_cache< T, spin_mutex > my_predecessors;
    spin_mutex my_mutex;
    internal::broadcast_cache< T > my_successors;
    int init_decrement_predecessors;

    friend class internal::forward_task_bypass< limiter_node<T> >;

    // Let decrementer call decrement_counter()
    friend class internal::decrementer< limiter_node<T> >;

    bool check_conditions() {  // always called under lock
        return ( my_count + my_tries < my_threshold && !my_predecessors.empty() && !my_successors.empty() );
    }

    // only returns a valid task pointer or NULL, never SUCCESSFULLY_ENQUEUED
    task *forward_task() {
        input_type v;
        task *rval = NULL;
        bool reserved = false;
            {
                spin_mutex::scoped_lock lock(my_mutex);
                if ( check_conditions() )
                    ++my_tries;
                else
                    return NULL;
            }

        //SUCCESS
        // if we can reserve and can put, we consume the reservation 
        // we increment the count and decrement the tries
        if ( (my_predecessors.try_reserve(v)) == true ){
            reserved=true;
            if ( (rval = my_successors.try_put_task(v)) != NULL ){
                {
                    spin_mutex::scoped_lock lock(my_mutex);
                    ++my_count;
                    --my_tries;
                    my_predecessors.try_consume();
                    if ( check_conditions() ) {
                        if ( this->my_graph.is_active() ) {
                            task *rtask = new ( task::allocate_additional_child_of( *(this->my_graph.root_task()) ) )
                                internal::forward_task_bypass< limiter_node<T> >( *this );
                            FLOW_SPAWN (*rtask);
                        }
                    }
                }
                return rval;
            }
        }
        //FAILURE
        //if we can't reserve, we decrement the tries
        //if we can reserve but can't put, we decrement the tries and release the reservation
        {
            spin_mutex::scoped_lock lock(my_mutex);
            --my_tries;
            if (reserved) my_predecessors.try_release();
            if ( check_conditions() ) {
                if ( this->my_graph.is_active() ) {
                    task *rtask = new ( task::allocate_additional_child_of( *(this->my_graph.root_task()) ) )
                        internal::forward_task_bypass< limiter_node<T> >( *this );
                    __TBB_ASSERT(!rval, "Have two tasks to handle");
                    return rtask;
                }
            }
            return rval;
        }
    }

    void forward() {
        __TBB_ASSERT(false, "Should never be called");
        return;
    }

    task * decrement_counter() {
        {
            spin_mutex::scoped_lock lock(my_mutex);
            if(my_count) --my_count;
        }
        return forward_task();
    }

public:
    //! The internal receiver< continue_msg > that decrements the count
    internal::decrementer< limiter_node<T> > decrement;

    //! Constructor
    limiter_node(graph &g, size_t threshold, int num_decrement_predecessors=0) :
        graph_node(g), my_threshold(threshold), my_count(0), my_tries(0),
        init_decrement_predecessors(num_decrement_predecessors),
        decrement(num_decrement_predecessors)
    {
        my_predecessors.set_owner(this);
        my_successors.set_owner(this);
        decrement.set_owner(this);
        tbb::internal::fgt_node( tbb::internal::FLOW_LIMITER_NODE, &this->my_graph,
                                 static_cast<receiver<input_type> *>(this), static_cast<receiver<continue_msg> *>(&decrement),
                                 static_cast<sender<output_type> *>(this) );
    }

    //! Copy constructor
    limiter_node( const limiter_node& src ) :
        graph_node(src.my_graph), receiver<T>(), sender<T>(),
        my_threshold(src.my_threshold), my_count(0), my_tries(0),
        init_decrement_predecessors(src.init_decrement_predecessors),
        decrement(src.init_decrement_predecessors)
    {
        my_predecessors.set_owner(this);
        my_successors.set_owner(this);
        decrement.set_owner(this);
        tbb::internal::fgt_node( tbb::internal::FLOW_LIMITER_NODE, &this->my_graph,
                                 static_cast<receiver<input_type> *>(this), static_cast<receiver<continue_msg> *>(&decrement),
                                 static_cast<sender<output_type> *>(this) );
    }

#if TBB_PREVIEW_FLOW_GRAPH_TRACE
    /* override */ void set_name( const char *name ) {
        tbb::internal::fgt_node_desc( this, name );
    }
#endif

    //! Replace the current successor with this new successor
    /* override */ bool register_successor( receiver<output_type> &r ) {
        spin_mutex::scoped_lock lock(my_mutex);
        bool was_empty = my_successors.empty();
        my_successors.register_successor(r);
        //spawn a forward task if this is the only successor
        if ( was_empty && !my_predecessors.empty() && my_count + my_tries < my_threshold ) {
            if ( this->my_graph.is_active() ) {
                FLOW_SPAWN( (* new ( task::allocate_additional_child_of( *(this->my_graph.root_task()) ) )
                            internal::forward_task_bypass < limiter_node<T> >( *this ) ) );
            }
        }
        return true;
    }

    //! Removes a successor from this node
    /** r.remove_predecessor(*this) is also called. */
    /* override */ bool remove_successor( receiver<output_type> &r ) {
        r.remove_predecessor(*this);
        my_successors.remove_successor(r);
        return true;
    }

#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
    /*override*/ built_successors_type &built_successors() { return my_successors.built_successors(); }
    /*override*/ built_predecessors_type &built_predecessors() { return my_predecessors.built_predecessors(); }

    /*override*/void internal_add_built_successor(receiver<output_type> &src) {
        my_successors.internal_add_built_successor(src);
    }

    /*override*/void internal_delete_built_successor(receiver<output_type> &src) {
        my_successors.internal_delete_built_successor(src);
    }

    /*override*/size_t successor_count() { return my_successors.successor_count(); }

    /*override*/ void copy_successors(successor_list_type &v) {
        my_successors.copy_successors(v);
    }

    /*override*/void internal_add_built_predecessor(sender<output_type> &src) {
        my_predecessors.internal_add_built_predecessor(src);
    }

    /*override*/void internal_delete_built_predecessor(sender<output_type> &src) {
        my_predecessors.internal_delete_built_predecessor(src);
    }

    /*override*/size_t predecessor_count() { return my_predecessors.predecessor_count(); }

    /*override*/ void copy_predecessors(predecessor_list_type &v) {
        my_predecessors.copy_predecessors(v);
    }

    /*override*/void extract() {
        my_count = 0;
        my_successors.built_successors().sender_extract(*this);
        my_predecessors.built_predecessors().receiver_extract(*this);
        decrement.built_predecessors().receiver_extract(decrement);
    }
#endif  /* TBB_PREVIEW_FLOW_GRAPH_FEATURES */

    //! Adds src to the list of cached predecessors.
    /* override */ bool register_predecessor( predecessor_type &src ) {
        spin_mutex::scoped_lock lock(my_mutex);
        my_predecessors.add( src );
        if ( my_count + my_tries < my_threshold && !my_successors.empty() && this->my_graph.is_active() ) {
            FLOW_SPAWN( (* new ( task::allocate_additional_child_of( *(this->my_graph.root_task()) ) )
                        internal::forward_task_bypass < limiter_node<T> >( *this ) ) );
        }
        return true;
    }

    //! Removes src from the list of cached predecessors.
    /* override */ bool remove_predecessor( predecessor_type &src ) {
        my_predecessors.remove( src );
        return true;
    }

protected:

    template< typename R, typename B > friend class run_and_put_task;
    template<typename X, typename Y> friend class internal::broadcast_cache;
    template<typename X, typename Y> friend class internal::round_robin_cache;
    //! Puts an item to this receiver
    /* override */ task *try_put_task( const T &t ) {
        {
            spin_mutex::scoped_lock lock(my_mutex);
            if ( my_count + my_tries >= my_threshold )
                return NULL;
            else
                ++my_tries;
        }

        task * rtask = my_successors.try_put_task(t);

        if ( !rtask ) {  // try_put_task failed.
            spin_mutex::scoped_lock lock(my_mutex);
            --my_tries;
            if ( check_conditions() && this->my_graph.is_active() ) {
                rtask = new ( task::allocate_additional_child_of( *(this->my_graph.root_task()) ) )
                    internal::forward_task_bypass< limiter_node<T> >( *this );
            }
        }
        else {
            spin_mutex::scoped_lock lock(my_mutex);
            ++my_count;
            --my_tries;
             }
        return rtask;
    }

    /*override*/void reset_receiver(reset_flags /*f*/) { 
        __TBB_ASSERT(false,NULL);  // should never be called
    }

    /*override*/void reset_node( reset_flags f) {
        my_count = 0;
        if(f & rf_clear_edges) {
            my_predecessors.clear();
            my_successors.clear();
        }
        else
        {
            my_predecessors.reset( );
        }
        decrement.reset_receiver(f);
    }
};  // limiter_node

#include "internal/_flow_graph_join_impl.h"

using internal::reserving_port;
using internal::queueing_port;
using internal::key_matching_port;
using internal::input_port;
using internal::tag_value;

template<typename OutputTuple, typename JP=queueing> class join_node;

template<typename OutputTuple>
class join_node<OutputTuple,reserving>: public internal::unfolded_join_node<tbb::flow::tuple_size<OutputTuple>::value, reserving_port, OutputTuple, reserving> {
private:
    static const int N = tbb::flow::tuple_size<OutputTuple>::value;
    typedef typename internal::unfolded_join_node<N, reserving_port, OutputTuple, reserving> unfolded_type;
public:
    typedef OutputTuple output_type;
    typedef typename unfolded_type::input_ports_type input_ports_type;
    join_node(graph &g) : unfolded_type(g) {
        tbb::internal::fgt_multiinput_node<N>( tbb::internal::FLOW_JOIN_NODE_RESERVING, &this->my_graph,
                                            this->input_ports(), static_cast< sender< output_type > *>(this) );
    }
    join_node(const join_node &other) : unfolded_type(other) {
        tbb::internal::fgt_multiinput_node<N>( tbb::internal::FLOW_JOIN_NODE_RESERVING, &this->my_graph,
                                            this->input_ports(), static_cast< sender< output_type > *>(this) );
    }

#if TBB_PREVIEW_FLOW_GRAPH_TRACE
    /* override */ void set_name( const char *name ) {
        tbb::internal::fgt_node_desc( this, name );
    }
#endif

};

template<typename OutputTuple>
class join_node<OutputTuple,queueing>: public internal::unfolded_join_node<tbb::flow::tuple_size<OutputTuple>::value, queueing_port, OutputTuple, queueing> {
private:
    static const int N = tbb::flow::tuple_size<OutputTuple>::value;
    typedef typename internal::unfolded_join_node<N, queueing_port, OutputTuple, queueing> unfolded_type;
public:
    typedef OutputTuple output_type;
    typedef typename unfolded_type::input_ports_type input_ports_type;
    join_node(graph &g) : unfolded_type(g) {
        tbb::internal::fgt_multiinput_node<N>( tbb::internal::FLOW_JOIN_NODE_QUEUEING, &this->my_graph,
                                            this->input_ports(), static_cast< sender< output_type > *>(this) );
    }
    join_node(const join_node &other) : unfolded_type(other) {
        tbb::internal::fgt_multiinput_node<N>( tbb::internal::FLOW_JOIN_NODE_QUEUEING, &this->my_graph,
                                            this->input_ports(), static_cast< sender< output_type > *>(this) );
    }

#if TBB_PREVIEW_FLOW_GRAPH_TRACE
    /* override */ void set_name( const char *name ) {
        tbb::internal::fgt_node_desc( this, name );
    }
#endif

};

// template for key_matching join_node
// tag_matching join_node is a specialization of key_matching, and is source-compatible.
template<typename OutputTuple, typename K, typename KHash>
class join_node<OutputTuple, key_matching<K, KHash> > : public internal::unfolded_join_node<tbb::flow::tuple_size<OutputTuple>::value,
      key_matching_port, OutputTuple, key_matching<K,KHash> > {
private:
    static const int N = tbb::flow::tuple_size<OutputTuple>::value;
    typedef typename internal::unfolded_join_node<N, key_matching_port, OutputTuple, key_matching<K,KHash> > unfolded_type;
public:
    typedef OutputTuple output_type;
    typedef typename unfolded_type::input_ports_type input_ports_type;

#if __TBB_PREVIEW_MESSAGE_BASED_KEY_MATCHING
    join_node(graph &g) : unfolded_type(g) {}
#endif  /* __TBB_PREVIEW_MESSAGE_BASED_KEY_MATCHING */

    template<typename __TBB_B0, typename __TBB_B1>
    join_node(graph &g, __TBB_B0 b0, __TBB_B1 b1) : unfolded_type(g, b0, b1) {
        tbb::internal::fgt_multiinput_node<N>( tbb::internal::FLOW_JOIN_NODE_TAG_MATCHING, &this->my_graph,
                                                           this->input_ports(), static_cast< sender< output_type > *>(this) );
    }
    template<typename __TBB_B0, typename __TBB_B1, typename __TBB_B2>
    join_node(graph &g, __TBB_B0 b0, __TBB_B1 b1, __TBB_B2 b2) : unfolded_type(g, b0, b1, b2) {
        tbb::internal::fgt_multiinput_node<N>( tbb::internal::FLOW_JOIN_NODE_TAG_MATCHING, &this->my_graph,
                                                           this->input_ports(), static_cast< sender< output_type > *>(this) );
    }
    template<typename __TBB_B0, typename __TBB_B1, typename __TBB_B2, typename __TBB_B3>
    join_node(graph &g, __TBB_B0 b0, __TBB_B1 b1, __TBB_B2 b2, __TBB_B3 b3) : unfolded_type(g, b0, b1, b2, b3) {
        tbb::internal::fgt_multiinput_node<N>( tbb::internal::FLOW_JOIN_NODE_TAG_MATCHING, &this->my_graph,
                                                           this->input_ports(), static_cast< sender< output_type > *>(this) );
    }
    template<typename __TBB_B0, typename __TBB_B1, typename __TBB_B2, typename __TBB_B3, typename __TBB_B4>
    join_node(graph &g, __TBB_B0 b0, __TBB_B1 b1, __TBB_B2 b2, __TBB_B3 b3, __TBB_B4 b4) :
            unfolded_type(g, b0, b1, b2, b3, b4) {
        tbb::internal::fgt_multiinput_node<N>( tbb::internal::FLOW_JOIN_NODE_TAG_MATCHING, &this->my_graph,
                                                           this->input_ports(), static_cast< sender< output_type > *>(this) );
    }
#if __TBB_VARIADIC_MAX >= 6
    template<typename __TBB_B0, typename __TBB_B1, typename __TBB_B2, typename __TBB_B3, typename __TBB_B4,
        typename __TBB_B5>
    join_node(graph &g, __TBB_B0 b0, __TBB_B1 b1, __TBB_B2 b2, __TBB_B3 b3, __TBB_B4 b4, __TBB_B5 b5) :
            unfolded_type(g, b0, b1, b2, b3, b4, b5) {
        tbb::internal::fgt_multiinput_node<N>( tbb::internal::FLOW_JOIN_NODE_TAG_MATCHING, &this->my_graph,
                                                           this->input_ports(), static_cast< sender< output_type > *>(this) );
    }
#endif
#if __TBB_VARIADIC_MAX >= 7
    template<typename __TBB_B0, typename __TBB_B1, typename __TBB_B2, typename __TBB_B3, typename __TBB_B4,
        typename __TBB_B5, typename __TBB_B6>
    join_node(graph &g, __TBB_B0 b0, __TBB_B1 b1, __TBB_B2 b2, __TBB_B3 b3, __TBB_B4 b4, __TBB_B5 b5, __TBB_B6 b6) :
            unfolded_type(g, b0, b1, b2, b3, b4, b5, b6) {
        tbb::internal::fgt_multiinput_node<N>( tbb::internal::FLOW_JOIN_NODE_TAG_MATCHING, &this->my_graph,
                                                           this->input_ports(), static_cast< sender< output_type > *>(this) );
    }
#endif
#if __TBB_VARIADIC_MAX >= 8
    template<typename __TBB_B0, typename __TBB_B1, typename __TBB_B2, typename __TBB_B3, typename __TBB_B4,
        typename __TBB_B5, typename __TBB_B6, typename __TBB_B7>
    join_node(graph &g, __TBB_B0 b0, __TBB_B1 b1, __TBB_B2 b2, __TBB_B3 b3, __TBB_B4 b4, __TBB_B5 b5, __TBB_B6 b6,
            __TBB_B7 b7) : unfolded_type(g, b0, b1, b2, b3, b4, b5, b6, b7) {
        tbb::internal::fgt_multiinput_node<N>( tbb::internal::FLOW_JOIN_NODE_TAG_MATCHING, &this->my_graph,
                                                           this->input_ports(), static_cast< sender< output_type > *>(this) );
    }
#endif
#if __TBB_VARIADIC_MAX >= 9
    template<typename __TBB_B0, typename __TBB_B1, typename __TBB_B2, typename __TBB_B3, typename __TBB_B4,
        typename __TBB_B5, typename __TBB_B6, typename __TBB_B7, typename __TBB_B8>
    join_node(graph &g, __TBB_B0 b0, __TBB_B1 b1, __TBB_B2 b2, __TBB_B3 b3, __TBB_B4 b4, __TBB_B5 b5, __TBB_B6 b6,
            __TBB_B7 b7, __TBB_B8 b8) : unfolded_type(g, b0, b1, b2, b3, b4, b5, b6, b7, b8) {
        tbb::internal::fgt_multiinput_node<N>( tbb::internal::FLOW_JOIN_NODE_TAG_MATCHING, &this->my_graph,
                                                           this->input_ports(), static_cast< sender< output_type > *>(this) );
    }
#endif
#if __TBB_VARIADIC_MAX >= 10
    template<typename __TBB_B0, typename __TBB_B1, typename __TBB_B2, typename __TBB_B3, typename __TBB_B4,
        typename __TBB_B5, typename __TBB_B6, typename __TBB_B7, typename __TBB_B8, typename __TBB_B9>
    join_node(graph &g, __TBB_B0 b0, __TBB_B1 b1, __TBB_B2 b2, __TBB_B3 b3, __TBB_B4 b4, __TBB_B5 b5, __TBB_B6 b6,
            __TBB_B7 b7, __TBB_B8 b8, __TBB_B9 b9) : unfolded_type(g, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9) {
        tbb::internal::fgt_multiinput_node<N>( tbb::internal::FLOW_JOIN_NODE_TAG_MATCHING, &this->my_graph,
                                                           this->input_ports(), static_cast< sender< output_type > *>(this) );
    }
#endif
    join_node(const join_node &other) : unfolded_type(other) {
        tbb::internal::fgt_multiinput_node<N>( tbb::internal::FLOW_JOIN_NODE_TAG_MATCHING, &this->my_graph,
                                                           this->input_ports(), static_cast< sender< output_type > *>(this) );
    }

#if TBB_PREVIEW_FLOW_GRAPH_TRACE
    /* override */ void set_name( const char *name ) {
        tbb::internal::fgt_node_desc( this, name );
    }
#endif

};

// indexer node
#include "internal/_flow_graph_indexer_impl.h"

template<typename T0, typename T1=null_type, typename T2=null_type, typename T3=null_type,
                      typename T4=null_type, typename T5=null_type, typename T6=null_type,
                      typename T7=null_type, typename T8=null_type, typename T9=null_type> class indexer_node;

//indexer node specializations
template<typename T0>
class indexer_node<T0> : public internal::unfolded_indexer_node<tuple<T0> > {
private:
    static const int N = 1;
public:
    typedef tuple<T0> InputTuple;
    typedef typename internal::tagged_msg<size_t, T0> output_type;
    typedef typename internal::unfolded_indexer_node<InputTuple> unfolded_type;
    indexer_node(graph& g) : unfolded_type(g) {
        tbb::internal::fgt_multiinput_node<N>( tbb::internal::FLOW_INDEXER_NODE, &this->my_graph,
                                           this->input_ports(), static_cast< sender< output_type > *>(this) );
    }
    // Copy constructor
    indexer_node( const indexer_node& other ) : unfolded_type(other) {
        tbb::internal::fgt_multiinput_node<N>( tbb::internal::FLOW_INDEXER_NODE, &this->my_graph,
                                           this->input_ports(), static_cast< sender< output_type > *>(this) );
    }

#if TBB_PREVIEW_FLOW_GRAPH_TRACE
     void set_name( const char *name ) {
        tbb::internal::fgt_node_desc( this, name );
    }
#endif
};

template<typename T0, typename T1>
class indexer_node<T0, T1> : public internal::unfolded_indexer_node<tuple<T0, T1> > {
private:
    static const int N = 2;
public:
    typedef tuple<T0, T1> InputTuple;
    typedef typename internal::tagged_msg<size_t, T0, T1> output_type;
    typedef typename internal::unfolded_indexer_node<InputTuple> unfolded_type;
    indexer_node(graph& g) : unfolded_type(g) {
        tbb::internal::fgt_multiinput_node<N>( tbb::internal::FLOW_INDEXER_NODE, &this->my_graph,
                                           this->input_ports(), static_cast< sender< output_type > *>(this) );
    }
    // Copy constructor
    indexer_node( const indexer_node& other ) : unfolded_type(other) {
        tbb::internal::fgt_multiinput_node<N>( tbb::internal::FLOW_INDEXER_NODE, &this->my_graph,
                                           this->input_ports(), static_cast< sender< output_type > *>(this) );
    }

#if TBB_PREVIEW_FLOW_GRAPH_TRACE
     void set_name( const char *name ) {
        tbb::internal::fgt_node_desc( this, name );
    }
#endif
};

template<typename T0, typename T1, typename T2>
class indexer_node<T0, T1, T2> : public internal::unfolded_indexer_node<tuple<T0, T1, T2> > {
private:
    static const int N = 3;
public:
    typedef tuple<T0, T1, T2> InputTuple;
    typedef typename internal::tagged_msg<size_t, T0, T1, T2> output_type;
    typedef typename internal::unfolded_indexer_node<InputTuple> unfolded_type;
    indexer_node(graph& g) : unfolded_type(g) {
        tbb::internal::fgt_multiinput_node<N>( tbb::internal::FLOW_INDEXER_NODE, &this->my_graph,
                                           this->input_ports(), static_cast< sender< output_type > *>(this) );
    }
    // Copy constructor
    indexer_node( const indexer_node& other ) : unfolded_type(other) {
        tbb::internal::fgt_multiinput_node<N>( tbb::internal::FLOW_INDEXER_NODE, &this->my_graph,
                                           this->input_ports(), static_cast< sender< output_type > *>(this) );
    }

#if TBB_PREVIEW_FLOW_GRAPH_TRACE
        void set_name( const char *name ) {
        tbb::internal::fgt_node_desc( this, name );
    }
#endif
};

template<typename T0, typename T1, typename T2, typename T3>
class indexer_node<T0, T1, T2, T3> : public internal::unfolded_indexer_node<tuple<T0, T1, T2, T3> > {
private:
    static const int N = 4;
public:
    typedef tuple<T0, T1, T2, T3> InputTuple;
    typedef typename internal::tagged_msg<size_t, T0, T1, T2, T3> output_type;
    typedef typename internal::unfolded_indexer_node<InputTuple> unfolded_type;
    indexer_node(graph& g) : unfolded_type(g) {
        tbb::internal::fgt_multiinput_node<N>( tbb::internal::FLOW_INDEXER_NODE, &this->my_graph,
                                           this->input_ports(), static_cast< sender< output_type > *>(this) );
    }
    // Copy constructor
    indexer_node( const indexer_node& other ) : unfolded_type(other) {
        tbb::internal::fgt_multiinput_node<N>( tbb::internal::FLOW_INDEXER_NODE, &this->my_graph,
                                           this->input_ports(), static_cast< sender< output_type > *>(this) );
    }

#if TBB_PREVIEW_FLOW_GRAPH_TRACE
    /* override */ void set_name( const char *name ) {
        tbb::internal::fgt_node_desc( this, name );
    }
#endif
};

template<typename T0, typename T1, typename T2, typename T3, typename T4>
class indexer_node<T0, T1, T2, T3, T4> : public internal::unfolded_indexer_node<tuple<T0, T1, T2, T3, T4> > {
private:
    static const int N = 5;
public:
    typedef tuple<T0, T1, T2, T3, T4> InputTuple;
    typedef typename internal::tagged_msg<size_t, T0, T1, T2, T3, T4> output_type;
    typedef typename internal::unfolded_indexer_node<InputTuple> unfolded_type;
    indexer_node(graph& g) : unfolded_type(g) {
        tbb::internal::fgt_multiinput_node<N>( tbb::internal::FLOW_INDEXER_NODE, &this->my_graph,
                                           this->input_ports(), static_cast< sender< output_type > *>(this) );
    }
    // Copy constructor
    indexer_node( const indexer_node& other ) : unfolded_type(other) {
        tbb::internal::fgt_multiinput_node<N>( tbb::internal::FLOW_INDEXER_NODE, &this->my_graph,
                                           this->input_ports(), static_cast< sender< output_type > *>(this) );
    }

#if TBB_PREVIEW_FLOW_GRAPH_TRACE
    /* override */ void set_name( const char *name ) {
        tbb::internal::fgt_node_desc( this, name );
    }
#endif
};

#if __TBB_VARIADIC_MAX >= 6
template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5>
class indexer_node<T0, T1, T2, T3, T4, T5> : public internal::unfolded_indexer_node<tuple<T0, T1, T2, T3, T4, T5> > {
private:
    static const int N = 6;
public:
    typedef tuple<T0, T1, T2, T3, T4, T5> InputTuple;
    typedef typename internal::tagged_msg<size_t, T0, T1, T2, T3, T4, T5> output_type;
    typedef typename internal::unfolded_indexer_node<InputTuple> unfolded_type;
    indexer_node(graph& g) : unfolded_type(g) {
        tbb::internal::fgt_multiinput_node<N>( tbb::internal::FLOW_INDEXER_NODE, &this->my_graph,
                                           this->input_ports(), static_cast< sender< output_type > *>(this) );
    }
    // Copy constructor
    indexer_node( const indexer_node& other ) : unfolded_type(other) {
        tbb::internal::fgt_multiinput_node<N>( tbb::internal::FLOW_INDEXER_NODE, &this->my_graph,
                                           this->input_ports(), static_cast< sender< output_type > *>(this) );
    }

#if TBB_PREVIEW_FLOW_GRAPH_TRACE
    /* override */ void set_name( const char *name ) {
        tbb::internal::fgt_node_desc( this, name );
    }
#endif
};
#endif //variadic max 6

#if __TBB_VARIADIC_MAX >= 7
template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5,
         typename T6>
class indexer_node<T0, T1, T2, T3, T4, T5, T6> : public internal::unfolded_indexer_node<tuple<T0, T1, T2, T3, T4, T5, T6> > {
private:
    static const int N = 7;
public:
    typedef tuple<T0, T1, T2, T3, T4, T5, T6> InputTuple;
    typedef typename internal::tagged_msg<size_t, T0, T1, T2, T3, T4, T5, T6> output_type;
    typedef typename internal::unfolded_indexer_node<InputTuple> unfolded_type;
    indexer_node(graph& g) : unfolded_type(g) {
        tbb::internal::fgt_multiinput_node<N>( tbb::internal::FLOW_INDEXER_NODE, &this->my_graph,
                                           this->input_ports(), static_cast< sender< output_type > *>(this) );
    }
    // Copy constructor
    indexer_node( const indexer_node& other ) : unfolded_type(other) {
        tbb::internal::fgt_multiinput_node<N>( tbb::internal::FLOW_INDEXER_NODE, &this->my_graph,
                                           this->input_ports(), static_cast< sender< output_type > *>(this) );
    }

#if TBB_PREVIEW_FLOW_GRAPH_TRACE
    /* override */ void set_name( const char *name ) {
        tbb::internal::fgt_node_desc( this, name );
    }
#endif
};
#endif //variadic max 7

#if __TBB_VARIADIC_MAX >= 8
template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5,
         typename T6, typename T7>
class indexer_node<T0, T1, T2, T3, T4, T5, T6, T7> : public internal::unfolded_indexer_node<tuple<T0, T1, T2, T3, T4, T5, T6, T7> > {
private:
    static const int N = 8;
public:
    typedef tuple<T0, T1, T2, T3, T4, T5, T6, T7> InputTuple;
    typedef typename internal::tagged_msg<size_t, T0, T1, T2, T3, T4, T5, T6, T7> output_type;
    typedef typename internal::unfolded_indexer_node<InputTuple> unfolded_type;
    indexer_node(graph& g) : unfolded_type(g) {
        tbb::internal::fgt_multiinput_node<N>( tbb::internal::FLOW_INDEXER_NODE, &this->my_graph,
                                           this->input_ports(), static_cast< sender< output_type > *>(this) );
    }
    // Copy constructor
    indexer_node( const indexer_node& other ) : unfolded_type(other) {
        tbb::internal::fgt_multiinput_node<N>( tbb::internal::FLOW_INDEXER_NODE, &this->my_graph,
                                           this->input_ports(), static_cast< sender< output_type > *>(this) );
    }

#if TBB_PREVIEW_FLOW_GRAPH_TRACE
    /* override */ void set_name( const char *name ) {
        tbb::internal::fgt_node_desc( this, name );
    }
#endif
};
#endif //variadic max 8

#if __TBB_VARIADIC_MAX >= 9
template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5,
         typename T6, typename T7, typename T8>
class indexer_node<T0, T1, T2, T3, T4, T5, T6, T7, T8> : public internal::unfolded_indexer_node<tuple<T0, T1, T2, T3, T4, T5, T6, T7, T8> > {
private:
    static const int N = 9;
public:
    typedef tuple<T0, T1, T2, T3, T4, T5, T6, T7, T8> InputTuple;
    typedef typename internal::tagged_msg<size_t, T0, T1, T2, T3, T4, T5, T6, T7, T8> output_type;
    typedef typename internal::unfolded_indexer_node<InputTuple> unfolded_type;
    indexer_node(graph& g) : unfolded_type(g) {
        tbb::internal::fgt_multiinput_node<N>( tbb::internal::FLOW_INDEXER_NODE, &this->my_graph,
                                           this->input_ports(), static_cast< sender< output_type > *>(this) );
    }
    // Copy constructor
    indexer_node( const indexer_node& other ) : unfolded_type(other) {
        tbb::internal::fgt_multiinput_node<N>( tbb::internal::FLOW_INDEXER_NODE, &this->my_graph,
                                           this->input_ports(), static_cast< sender< output_type > *>(this) );
    }

#if TBB_PREVIEW_FLOW_GRAPH_TRACE
    /* override */ void set_name( const char *name ) {
        tbb::internal::fgt_node_desc( this, name );
    }
#endif
};
#endif //variadic max 9

#if __TBB_VARIADIC_MAX >= 10
template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5,
         typename T6, typename T7, typename T8, typename T9>
class indexer_node/*default*/ : public internal::unfolded_indexer_node<tuple<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9> > {
private:
    static const int N = 10;
public:
    typedef tuple<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9> InputTuple;
    typedef typename internal::tagged_msg<size_t, T0, T1, T2, T3, T4, T5, T6, T7, T8, T9> output_type;
    typedef typename internal::unfolded_indexer_node<InputTuple> unfolded_type;
    indexer_node(graph& g) : unfolded_type(g) {
        tbb::internal::fgt_multiinput_node<N>( tbb::internal::FLOW_INDEXER_NODE, &this->my_graph,
                                           this->input_ports(), static_cast< sender< output_type > *>(this) );
    }
    // Copy constructor
    indexer_node( const indexer_node& other ) : unfolded_type(other) {
        tbb::internal::fgt_multiinput_node<N>( tbb::internal::FLOW_INDEXER_NODE, &this->my_graph,
                                           this->input_ports(), static_cast< sender< output_type > *>(this) );
    }

#if TBB_PREVIEW_FLOW_GRAPH_TRACE
    /* override */ void set_name( const char *name ) {
        tbb::internal::fgt_node_desc( this, name );
    }
#endif
};
#endif //variadic max 10

//! Makes an edge between a single predecessor and a single successor
template< typename T >
inline void make_edge( sender<T> &p, receiver<T> &s ) {
#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
    s.internal_add_built_predecessor(p);
    p.internal_add_built_successor(s);
#endif
    p.register_successor( s );
    tbb::internal::fgt_make_edge( &p, &s );
}

#if __TBB_FLOW_GRAPH_CPP11_FEATURES
//Makes an edge from port 0 of a multi-output predecessor to port 0 of a multi-input successor.
template< typename T, typename V,
          typename = typename T::output_ports_type, typename = typename V::input_ports_type >
inline void make_edge( T& output, V& input) {
    make_edge(get<0>(output.output_ports()), get<0>(input.input_ports()));
}

//Makes an edge from port 0 of a multi-output predecessor to a receiver.
template< typename T, typename R,
          typename = typename T::output_ports_type >
inline void make_edge( T& output, receiver<R>& input) {
     make_edge(get<0>(output.output_ports()), input);
}

//Makes an edge from a sender to port 0 of a multi-input successor.
template< typename S,  typename V,
          typename = typename V::input_ports_type >
inline void make_edge( sender<S>& output, V& input) {
     make_edge(output, get<0>(input.input_ports()));
}
#endif

//! Removes an edge between a single predecessor and a single successor
template< typename T >
inline void remove_edge( sender<T> &p, receiver<T> &s ) {
    p.remove_successor( s );
#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
    // TODO: should we try to remove p from the predecessor list of s, in case the edge is reversed?
    p.internal_delete_built_successor(s);
    s.internal_delete_built_predecessor(p);
#endif
    tbb::internal::fgt_remove_edge( &p, &s );
}

#if __TBB_FLOW_GRAPH_CPP11_FEATURES
//Removes an edge between port 0 of a multi-output predecessor and port 0 of a multi-input successor.
template< typename T, typename V,
          typename = typename T::output_ports_type, typename = typename V::input_ports_type >
inline void remove_edge( T& output, V& input) {
    remove_edge(get<0>(output.output_ports()), get<0>(input.input_ports()));
}

//Removes an edge between port 0 of a multi-output predecessor and a receiver.
template< typename T, typename R,
          typename = typename T::output_ports_type >
inline void remove_edge( T& output, receiver<R>& input) {
     remove_edge(get<0>(output.output_ports()), input);
}
//Removes an edge between a sender and port 0 of a multi-input successor.
template< typename S,  typename V,
          typename = typename V::input_ports_type >
inline void remove_edge( sender<S>& output, V& input) {
     remove_edge(output, get<0>(input.input_ports()));
}
#endif

#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
template<typename C >
template< typename S >
void internal::edge_container<C>::sender_extract( S &s ) {
    edge_list_type e = built_edges;
    for ( typename edge_list_type::iterator i = e.begin(); i != e.end(); ++i ) {
        remove_edge(s, **i);
    }
}

template<typename C >
template< typename R >
void internal::edge_container<C>::receiver_extract( R &r ) {
    edge_list_type e = built_edges;
    for ( typename edge_list_type::iterator i = e.begin(); i != e.end(); ++i ) {
        remove_edge(**i, r);
    }
}
#endif  /* TBB_PREVIEW_FLOW_GRAPH_FEATURES */

//! Returns a copy of the body from a function or continue node
template< typename Body, typename Node >
Body copy_body( Node &n ) {
    return n.template copy_function_object<Body>();
}

#if __TBB_FLOW_GRAPH_CPP11_FEATURES

//composite_node
template< typename InputTuple, typename OutputTuple > class composite_node;

template< typename... InputTypes, typename... OutputTypes>
class composite_node <tbb::flow::tuple<InputTypes...>, tbb::flow::tuple<OutputTypes...> > : public graph_node, tbb::internal::no_copy {

public:
    typedef tbb::flow::tuple< receiver<InputTypes>&... > input_ports_type;
    typedef tbb::flow::tuple< sender<OutputTypes>&... > output_ports_type;

private:
#if TBB_PREVIEW_FLOW_GRAPH_TRACE
    const char *my_type_name;
#endif
    input_ports_type *my_input_ports;
    output_ports_type *my_output_ports;

    static const size_t NUM_INPUTS = sizeof...(InputTypes);
    static const size_t NUM_OUTPUTS = sizeof...(OutputTypes);

protected:
    /*override*/void reset_node(reset_flags) {}

public:
#if TBB_PREVIEW_FLOW_GRAPH_TRACE
    composite_node( graph &g, const char *type_name = "composite_node") : graph_node(g), my_type_name(type_name), my_input_ports(NULL), my_output_ports(NULL)  {
        tbb::internal::itt_make_task_group( tbb::internal::ITT_DOMAIN_FLOW, this, tbb::internal::FLOW_NODE, &g, tbb::internal::FLOW_GRAPH, tbb::internal::FLOW_COMPOSITE_NODE );
        tbb::internal::fgt_multiinput_multioutput_node_desc( this, my_type_name );
    }
#else
    composite_node( graph &g) : graph_node(g), my_input_ports(NULL), my_output_ports(NULL) {}
#endif

   template<typename T1, typename T2>
   void set_external_ports(T1&& input_ports_tuple, T2&& output_ports_tuple) {
       __TBB_STATIC_ASSERT(NUM_INPUTS == tbb::flow::tuple_size<input_ports_type>::value, "number of arguments does not match number of input ports");
       __TBB_STATIC_ASSERT(NUM_OUTPUTS == tbb::flow::tuple_size<output_ports_type>::value, "number of arguments does not match number of output ports");
      my_input_ports =  new input_ports_type(std::forward<T1>(input_ports_tuple));
      my_output_ports = new output_ports_type(std::forward<T2>(output_ports_tuple));

#if TBB_PREVIEW_FLOW_GRAPH_TRACE
      tbb::internal::fgt_internal_input_helper<T1, NUM_INPUTS>::register_port( this, input_ports_tuple);
      tbb::internal::fgt_internal_output_helper<T2, NUM_OUTPUTS>::register_port( this, output_ports_tuple);
#endif
   }

#if TBB_PREVIEW_FLOW_GRAPH_TRACE
    template< typename... NodeTypes >
    void add_visible_nodes(const NodeTypes&... n) { internal::add_nodes_impl(this, true, n...); }

    template< typename... NodeTypes >
    void add_nodes(const NodeTypes&... n) { internal::add_nodes_impl(this, false, n...); }
#else
    template<typename... Nodes> void add_nodes(Nodes&...) { }
    template<typename... Nodes> void add_visible_nodes(Nodes&...) { }
#endif

#if TBB_PREVIEW_FLOW_GRAPH_TRACE
    /* override */ void set_name( const char *name ) {
        tbb::internal::fgt_multiinput_multioutput_node_desc( this, name );
    }
#endif

    input_ports_type input_ports() {
         __TBB_ASSERT(my_input_ports, "input ports not set, call set_external_ports to set input ports");
         return *my_input_ports;
    }

    output_ports_type output_ports() {
         __TBB_ASSERT(my_output_ports, "output ports not set, call set_external_ports to set output ports");
         return *my_output_ports;
    }

    virtual ~composite_node() {
        if(my_input_ports) delete my_input_ports;
        if(my_output_ports) delete my_output_ports;
    }

#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
    /*override*/void extract() {
        __TBB_ASSERT(false, "Current composite_node implementation does not support extract");
    }
#endif
};  // class composite_node

//composite_node with only input ports
template< typename... InputTypes>
class composite_node <tbb::flow::tuple<InputTypes...>, tbb::flow::tuple<> > : public graph_node, tbb::internal::no_copy {
public:
    typedef tbb::flow::tuple< receiver<InputTypes>&... > input_ports_type;

private:
#if TBB_PREVIEW_FLOW_GRAPH_TRACE
    const char *my_type_name;
#endif 
    input_ports_type *my_input_ports;
    static const size_t NUM_INPUTS = sizeof...(InputTypes);

protected:
    /*override*/void reset_node(reset_flags) {}

public:
#if TBB_PREVIEW_FLOW_GRAPH_TRACE
    composite_node( graph &g, const char *type_name = "composite_node") : graph_node(g), my_type_name(type_name), my_input_ports(NULL)  {
        tbb::internal::itt_make_task_group( tbb::internal::ITT_DOMAIN_FLOW, this, tbb::internal::FLOW_NODE, &g, tbb::internal::FLOW_GRAPH, tbb::internal::FLOW_COMPOSITE_NODE );
        tbb::internal::fgt_multiinput_multioutput_node_desc( this, my_type_name );
    }
#else
    composite_node( graph &g) : graph_node(g), my_input_ports(NULL) {}
#endif

   template<typename T>
   void set_external_ports(T&& input_ports_tuple) {
       __TBB_STATIC_ASSERT(NUM_INPUTS == tbb::flow::tuple_size<input_ports_type>::value, "number of arguments does not match number of input ports");

      my_input_ports =  new input_ports_type(std::forward<T>(input_ports_tuple));

#if TBB_PREVIEW_FLOW_GRAPH_TRACE
      tbb::internal::fgt_internal_input_helper<T, NUM_INPUTS>::register_port( this, std::forward<T>(input_ports_tuple));
#endif
   }

#if TBB_PREVIEW_FLOW_GRAPH_TRACE
    template< typename... NodeTypes >
    void add_visible_nodes(const NodeTypes&... n) { internal::add_nodes_impl(this, true, n...); }

    template< typename... NodeTypes >
    void add_nodes( const NodeTypes&... n) { internal::add_nodes_impl(this, false, n...); }
#else
    template<typename... Nodes> void add_nodes(Nodes&...) {}
    template<typename... Nodes> void add_visible_nodes(Nodes&...) {}
#endif

#if TBB_PREVIEW_FLOW_GRAPH_TRACE
    /* override */ void set_name( const char *name ) {
        tbb::internal::fgt_multiinput_multioutput_node_desc( this, name );
    }
#endif

    input_ports_type input_ports() {
         __TBB_ASSERT(my_input_ports, "input ports not set, call set_external_ports to set input ports");
         return *my_input_ports;
    }

    virtual ~composite_node() {
        if(my_input_ports) delete my_input_ports;
    }

#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
    /*override*/void extract() {
        __TBB_ASSERT(false, "Current composite_node implementation does not support extract");
    }
#endif

};  // class composite_node

//composite_nodes with only output_ports
template<typename... OutputTypes>
class composite_node <tbb::flow::tuple<>, tbb::flow::tuple<OutputTypes...> > : public graph_node, tbb::internal::no_copy {
public:
    typedef tbb::flow::tuple< sender<OutputTypes>&... > output_ports_type;

private:
#if TBB_PREVIEW_FLOW_GRAPH_TRACE
    const char *my_type_name;
#endif
    output_ports_type *my_output_ports;
    static const size_t NUM_OUTPUTS = sizeof...(OutputTypes);

protected:
    /*override*/void reset_node(reset_flags) {}

public:
#if TBB_PREVIEW_FLOW_GRAPH_TRACE
    composite_node( graph &g, const char *type_name = "composite_node") : graph_node(g), my_type_name(type_name), my_output_ports(NULL) {
        tbb::internal::itt_make_task_group( tbb::internal::ITT_DOMAIN_FLOW, this, tbb::internal::FLOW_NODE, &g, tbb::internal::FLOW_GRAPH, tbb::internal::FLOW_COMPOSITE_NODE );
        tbb::internal::fgt_multiinput_multioutput_node_desc( this, my_type_name );
    }
#else
    composite_node( graph &g) : graph_node(g), my_output_ports(NULL) {}
#endif

   template<typename T>
   void set_external_ports(T&& output_ports_tuple) {
       __TBB_STATIC_ASSERT(NUM_OUTPUTS == tbb::flow::tuple_size<output_ports_type>::value, "number of arguments does not match number of output ports");

      my_output_ports = new output_ports_type(std::forward<T>(output_ports_tuple));

#if TBB_PREVIEW_FLOW_GRAPH_TRACE
      tbb::internal::fgt_internal_output_helper<T, NUM_OUTPUTS>::register_port( this, std::forward<T>(output_ports_tuple));
#endif
   }

#if TBB_PREVIEW_FLOW_GRAPH_TRACE
    template<typename... NodeTypes >
    void add_visible_nodes(const NodeTypes&... n) { internal::add_nodes_impl(this, true, n...); }

    template<typename... NodeTypes >
    void add_nodes(const NodeTypes&... n) { internal::add_nodes_impl(this, false, n...); }
#else
    template<typename... Nodes> void add_nodes(Nodes&...) {}
    template<typename... Nodes> void add_visible_nodes(Nodes&...) {}
#endif

#if TBB_PREVIEW_FLOW_GRAPH_TRACE
    /* override */ void set_name( const char *name ) {
        tbb::internal::fgt_multiinput_multioutput_node_desc( this, name );
    }
#endif

    output_ports_type output_ports() {
         __TBB_ASSERT(my_output_ports, "output ports not set, call set_external_ports to set output ports");
         return *my_output_ports;
    }

    virtual ~composite_node() {
        if(my_output_ports) delete my_output_ports;
    }

#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
    /*override*/void extract() {
        __TBB_ASSERT(false, "Current composite_node implementation does not support extract");
    }
#endif

};  // class composite_node

#endif // __TBB_FLOW_GRAPH_CPP11_FEATURES

#if __TBB_PREVIEW_ASYNC_NODE
namespace internal {
//! Pure virtual template class that defines interface for async communication
template < typename Output >
class async_gateway {
public:
    typedef Output output_type;

    //! Submit signal from an asynchronous activity to FG
    virtual bool async_try_put( const output_type &i ) = 0;

    //! Increment reference count of graph to prevent premature return from wait_for_all
    virtual void async_reserve() = 0;

    //! Decrement reference count of graph to allow return from wait_for_all
    virtual void async_commit() = 0;

    virtual ~async_gateway() {}
};

template<typename Input, typename Ports, typename AsyncGateway, typename Body>
class async_body { 
public:
    typedef AsyncGateway async_gateway_type;

    async_body(const Body &body, async_gateway_type *gateway) : my_body(body), my_async_gateway(gateway) { }
 
    async_body(const async_body &other) : my_body(other.my_body), my_async_gateway(other.my_async_gateway) { }

    void operator()( const Input &v, Ports & ) {
        my_body(v, *my_async_gateway);
    }

    Body get_body() { return my_body; }

    void set_async_gateway(async_gateway_type *gateway) {
        my_async_gateway = gateway;
    }

private:
    Body my_body;
    async_gateway_type *my_async_gateway;
};

}

//! Implements async node
template < typename Input, typename Output, typename Policy = queueing, typename Allocator=cache_aligned_allocator<Input> >
class async_node : public multifunction_node< Input, tuple< Output >, Policy, Allocator >, public internal::async_gateway<Output>, public sender< Output > {
protected:
    typedef multifunction_node< Input, tuple< Output >, Policy, Allocator > base_type;

public:
    typedef Input input_type;
    typedef Output output_type;
    typedef sender< input_type > predecessor_type;
    typedef receiver< output_type > successor_type;
    typedef internal::async_gateway< output_type > async_gateway_type;

protected:
    typedef typename internal::multifunction_input<Input, typename base_type::output_ports_type, Allocator> mfn_input_type;

    struct try_put_functor {
        typedef internal::multifunction_output<Output> output_port_type;
        output_port_type *port;
        const Output *value;
        bool result;
        try_put_functor(output_port_type &p, const Output &v) : port(&p), value(&v), result(false) { }
        void operator()() {
            result = port->try_put(*value);
        }
    };

public:
    template<typename Body>
    async_node( graph &g, size_t concurrency, Body body ) :
        base_type( g, concurrency, internal::async_body<Input, typename base_type::output_ports_type, async_gateway_type, Body>(body, this) ) {
        tbb::internal::fgt_multioutput_node<1>( tbb::internal::FLOW_ASYNC_NODE,
                                                &this->graph_node::my_graph,
                                                static_cast<receiver<input_type> *>(this),
                                                this->output_ports() );
    }

    async_node( const async_node &other ) : base_type(other) {
        typedef internal::multifunction_body<input_type, typename base_type::output_ports_type> mfn_body_type;
        mfn_body_type &body_ref = *this->my_body;
        body_ref.set_gateway(static_cast<async_gateway_type *>(this));
        mfn_body_type &init_body_ref = *this->my_init_body;
        init_body_ref.set_gateway(static_cast<async_gateway_type *>(this));
        tbb::internal::fgt_multioutput_node<1>( tbb::internal::FLOW_ASYNC_NODE, &this->graph_node::my_graph, static_cast<receiver<input_type> *>(this), this->output_ports() );
    }

    virtual ~async_node() {}

    /* override */ async_gateway_type& async_gateway() {
        return static_cast< async_gateway_type& >(*this);
    }

    //! Implements async_gateway::async_try_put for an external activity to submit a message to FG
    /*override*/ bool async_try_put(const output_type &i ) {
        internal::multifunction_output<output_type> &port_0 = internal::output_port<0>(*this);
        graph &g = this->graph_node::my_graph;
        tbb::internal::fgt_async_try_put_begin(static_cast<receiver<input_type> *>(this), &port_0);
        __TBB_ASSERT(g.my_task_arena && g.my_task_arena->is_active(), NULL);
        try_put_functor tpf(port_0, i);
        g.my_task_arena->execute(tpf);
        tbb::internal::fgt_async_try_put_end(static_cast<receiver<input_type> *>(this), &port_0);
        return tpf.result;
    }

    /*override*/ void async_reserve() {
        this->graph_node::my_graph.increment_wait_count();
        tbb::internal::fgt_async_reserve(static_cast<receiver<input_type> *>(this), &this->graph_node::my_graph);
    }

    /*override*/ void async_commit() {
        this->graph_node::my_graph.decrement_wait_count();
        tbb::internal::fgt_async_commit(static_cast<receiver<input_type> *>(this), &this->graph_node::my_graph);
    }

#if TBB_PREVIEW_FLOW_GRAPH_TRACE
    /* override */ void set_name( const char *name ) {
            tbb::internal::fgt_node_desc( this, name );
    }
#endif

    // Define sender< Output >

    //! Add a new successor to this node
    /* override */ bool register_successor( successor_type &r ) {
        return internal::output_port<0>(*this).register_successor(r);
    }

    //! Removes a successor from this node
    /* override */  bool remove_successor( successor_type &r ) {
        return internal::output_port<0>(*this).remove_successor(r);
    }

    template<typename Body>
    Body copy_function_object() {
        typedef internal::multifunction_body<input_type, typename base_type::output_ports_type> mfn_body_type;
        typedef internal::async_body<Input, typename base_type::output_ports_type, async_gateway_type, Body> async_body_type;
        mfn_body_type &body_ref = *this->my_body;
        async_body_type ab = dynamic_cast< internal::multifunction_body_leaf<input_type, typename base_type::output_ports_type, async_body_type> & >(body_ref).get_body(); 
        return ab.get_body();
    }

#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
    //! interface to record edges for traversal & deletion
    typedef typename  internal::edge_container<successor_type> built_successors_type;
    typedef typename  built_successors_type::edge_list_type successor_list_type;
    /* override */ built_successors_type &built_successors() {
        return internal::output_port<0>(*this).built_successors();
    }

    /* override */ void    internal_add_built_successor( successor_type &r ) {
        internal::output_port<0>(*this).internal_add_built_successor(r);
    }

    /* override */ void    internal_delete_built_successor( successor_type &r ) {
        internal::output_port<0>(*this).internal_delete_built_successor(r);
    }

    /* override */ void    copy_successors( successor_list_type &l ) {
        internal::output_port<0>(*this).copy_successors(l);
    }

    /* override */ size_t  successor_count() {
        return internal::output_port<0>(*this).successor_count();
    }
#endif

protected:

    /*override*/ void reset_node( reset_flags f) {
       base_type::reset_node(f);
    }

}; 

#endif // __TBB_PREVIEW_ASYNC_NODE

} // interface8

    using interface8::reset_flags;
    using interface8::rf_reset_protocol;
    using interface8::rf_reset_bodies;
    using interface8::rf_clear_edges;

    using interface8::graph;
    using interface8::graph_node;
    using interface8::continue_msg;

    using interface8::source_node;
    using interface8::function_node;
    using interface8::multifunction_node;
    using interface8::split_node;
    using interface8::internal::output_port;
    using interface8::indexer_node;
    using interface8::internal::tagged_msg;
    using interface8::internal::cast_to;
    using interface8::internal::is_a;
    using interface8::continue_node;
    using interface8::overwrite_node;
    using interface8::write_once_node;
    using interface8::broadcast_node;
    using interface8::buffer_node;
    using interface8::queue_node;
    using interface8::sequencer_node;
    using interface8::priority_queue_node;
    using interface8::limiter_node;
    using namespace interface8::internal::graph_policy_namespace;
    using interface8::join_node;
    using interface8::input_port;
    using interface8::copy_body;
    using interface8::make_edge;
    using interface8::remove_edge;
    using interface8::internal::tag_value;
#if __TBB_FLOW_GRAPH_CPP11_FEATURES
     using interface8::composite_node;
#endif
#if __TBB_PREVIEW_ASYNC_NODE
    using interface8::async_node;
#endif
} // flow
} // tbb

#undef __TBB_PFG_RESET_ARG
#undef __TBB_COMMA

#endif // __TBB_flow_graph_H
