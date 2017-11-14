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

#ifndef __TBB__flow_graph_impl_H
#define __TBB__flow_graph_impl_H

#ifndef __TBB_flow_graph_H
#error Do not #include this internal file directly; use public TBB headers instead.
#endif

// included in namespace tbb::flow::interfaceX (in flow_graph.h)

namespace internal {

    typedef tbb::internal::uint64_t tag_value;

    using tbb::internal::strip;

    namespace graph_policy_namespace {

        struct rejecting { };
        struct reserving { };
        struct queueing  { };

        // K == type of field used for key-matching.  Each tag-matching port will be provided
        // functor that, given an object accepted by the port, will return the
        /// field of type K being used for matching.
        template<typename K, typename KHash=tbb_hash_compare<typename strip<K>::type > >
        struct key_matching {
            typedef K key_type;
            typedef typename strip<K>::type base_key_type;
            typedef KHash hash_compare_type;
        };

        // old tag_matching join's new specifier
        typedef key_matching<tag_value> tag_matching;
    }

// -------------- function_body containers ----------------------

    //! A functor that takes no input and generates a value of type Output
    template< typename Output >
    class source_body : tbb::internal::no_assign {
    public:
        virtual ~source_body() {}
        virtual bool operator()(Output &output) = 0;
        virtual source_body* clone() = 0;
    };

    //! The leaf for source_body
    template< typename Output, typename Body>
    class source_body_leaf : public source_body<Output> {
    public:
        source_body_leaf( const Body &_body ) : body(_body) { }
        /*override*/ bool operator()(Output &output) { return body( output ); }
        /*override*/ source_body_leaf* clone() {
            return new source_body_leaf< Output, Body >(body);
        }
        Body get_body() { return body; }
    private:
        Body body;
    };

    //! A functor that takes an Input and generates an Output
    template< typename Input, typename Output >
    class function_body : tbb::internal::no_assign {
    public:
        virtual ~function_body() {}
        virtual Output operator()(const Input &input) = 0;
        virtual function_body* clone() = 0;
    };

    //! the leaf for function_body
    template <typename Input, typename Output, typename B>
    class function_body_leaf : public function_body< Input, Output > {
    public:
        function_body_leaf( const B &_body ) : body(_body) { }
        Output operator()(const Input &i) { return body(i); }
        B get_body() { return body; }
        /*override*/ function_body_leaf* clone() {
            return new function_body_leaf< Input, Output, B >(body);
        }
    private:
        B body;
    };

    //! the leaf for function_body specialized for Input and output of continue_msg
    template <typename B>
    class function_body_leaf< continue_msg, continue_msg, B> : public function_body< continue_msg, continue_msg > {
    public:
        function_body_leaf( const B &_body ) : body(_body) { }
        continue_msg operator()( const continue_msg &i ) {
            body(i);
            return i;
        }
        B get_body() { return body; }
        /*override*/ function_body_leaf* clone() {
           return new function_body_leaf< continue_msg, continue_msg, B >(body);
        }
    private:
        B body;
    };

    //! the leaf for function_body specialized for Output of continue_msg
    template <typename Input, typename B>
    class function_body_leaf< Input, continue_msg, B> : public function_body< Input, continue_msg > {
    public:
        function_body_leaf( const B &_body ) : body(_body) { }
        continue_msg operator()(const Input &i) {
            body(i);
            return continue_msg();
        }
        B get_body() { return body; }
        /*override*/ function_body_leaf* clone() {
            return new function_body_leaf< Input, continue_msg, B >(body);
        }
    private:
        B body;
    };

    //! the leaf for function_body specialized for Input of continue_msg
    template <typename Output, typename B>
    class function_body_leaf< continue_msg, Output, B > : public function_body< continue_msg, Output > {
    public:
        function_body_leaf( const B &_body ) : body(_body) { }
        Output operator()(const continue_msg &i) {
            return body(i);
        }
        B get_body() { return body; }
        /*override*/ function_body_leaf* clone() {
            return new function_body_leaf< continue_msg, Output, B >(body);
        }
    private:
        B body;
    };

#if __TBB_PREVIEW_ASYNC_NODE
template< typename T, typename = typename T::async_gateway_type >
void set_async_gateway(T *body, void *g) {
    body->set_async_gateway(static_cast<typename T::async_gateway_type *>(g));
}

void set_async_gateway(...) { }
#endif

    //! function_body that takes an Input and a set of output ports
    template<typename Input, typename OutputSet>
    class multifunction_body : tbb::internal::no_assign {
    public:
        virtual ~multifunction_body () {}
        virtual void operator()(const Input &/* input*/, OutputSet &/*oset*/) = 0;
        virtual multifunction_body* clone() = 0;
#if __TBB_PREVIEW_ASYNC_NODE
        virtual void set_gateway(void *gateway) = 0;
#endif
    };

    //! leaf for multifunction.  OutputSet can be a std::tuple or a vector.
    template<typename Input, typename OutputSet, typename B >
    class multifunction_body_leaf : public multifunction_body<Input, OutputSet> {
    public:
        multifunction_body_leaf(const B &_body) : body(_body) { }
        void operator()(const Input &input, OutputSet &oset) {
            body(input, oset); // body may explicitly put() to one or more of oset.
        }
        B get_body() { return body; }
		
#if __TBB_PREVIEW_ASYNC_NODE
        /*override*/  void set_gateway(void *gateway) {
           set_async_gateway(&body, gateway);
        }
#endif
        /*override*/ multifunction_body_leaf* clone() {
            return new multifunction_body_leaf<Input, OutputSet,B>(body);
        }

    private:
        B body;
    };

// ------ function bodies for hash_buffers and key-matching joins.

template<typename Input, typename Output>
class type_to_key_function_body : tbb::internal::no_assign {
    public:
        virtual ~type_to_key_function_body() {}
        virtual Output operator()(const Input &input) = 0;  // returns an Output
        virtual type_to_key_function_body* clone() = 0;
};

// specialization for ref output
template<typename Input, typename Output>
class type_to_key_function_body<Input,Output&> : tbb::internal::no_assign {
    public:
        virtual ~type_to_key_function_body() {}
        virtual const Output & operator()(const Input &input) = 0;  // returns a const Output&
        virtual type_to_key_function_body* clone() = 0;
};

template <typename Input, typename Output, typename B>
class type_to_key_function_body_leaf : public type_to_key_function_body<Input, Output> {
public:
    type_to_key_function_body_leaf( const B &_body ) : body(_body) { }
    /*override*/Output operator()(const Input &i) { return body(i); }
    B get_body() { return body; }
    /*override*/ type_to_key_function_body_leaf* clone() {
        return new type_to_key_function_body_leaf< Input, Output, B>(body);
    }
private:
    B body;
};

template <typename Input, typename Output, typename B>
class type_to_key_function_body_leaf<Input,Output&,B> : public type_to_key_function_body< Input, Output&> {
public:
    type_to_key_function_body_leaf( const B &_body ) : body(_body) { }

    /*override*/const Output& operator()(const Input &i) {
        return body(i);
    }

    B get_body() { return body; }

    /*override*/ type_to_key_function_body_leaf* clone() {
        return new type_to_key_function_body_leaf< Input, Output&, B>(body);
    }

private:
    B body;
};

// --------------------------- end of function_body containers ------------------------

// --------------------------- node task bodies ---------------------------------------

    //! A task that calls a node's forward_task function
    template< typename NodeType >
    class forward_task_bypass : public task {

        NodeType &my_node;

    public:

        forward_task_bypass( NodeType &n ) : my_node(n) {}

        task *execute() {
            task * new_task = my_node.forward_task();
            if (new_task == SUCCESSFULLY_ENQUEUED) new_task = NULL;
            return new_task;
        }
    };

    //! A task that calls a node's apply_body_bypass function, passing in an input of type Input
    //  return the task* unless it is SUCCESSFULLY_ENQUEUED, in which case return NULL
    template< typename NodeType, typename Input >
    class apply_body_task_bypass : public task {

        NodeType &my_node;
        Input my_input;

    public:

        apply_body_task_bypass( NodeType &n, const Input &i ) : my_node(n), my_input(i) {}

        task *execute() {
            task * next_task = my_node.apply_body_bypass( my_input );
            if(next_task == SUCCESSFULLY_ENQUEUED) next_task = NULL;
            return next_task;
        }
    };

    //! A task that calls a node's apply_body_bypass function with no input
    template< typename NodeType >
    class source_task_bypass : public task {

        NodeType &my_node;

    public:

        source_task_bypass( NodeType &n ) : my_node(n) {}

        task *execute() {
            task *new_task = my_node.apply_body_bypass( );
            if(new_task == SUCCESSFULLY_ENQUEUED) return NULL;
            return new_task;
        }
    };

// ------------------------ end of node task bodies -----------------------------------

    //! An empty functor that takes an Input and returns a default constructed Output
    template< typename Input, typename Output >
    struct empty_body {
       Output operator()( const Input & ) const { return Output(); }
    };

    //! A node_cache maintains a std::queue of elements of type T.  Each operation is protected by a lock.
    template< typename T, typename M=spin_mutex >
    class node_cache {
        public:

        typedef size_t size_type;

        bool empty() {
            typename mutex_type::scoped_lock lock( my_mutex );
            return internal_empty();
        }

        void add( T &n ) {
            typename mutex_type::scoped_lock lock( my_mutex );
            internal_push(n);
        }

        void remove( T &n ) {
            typename mutex_type::scoped_lock lock( my_mutex );
            for ( size_t i = internal_size(); i != 0; --i ) {
                T &s = internal_pop();
                if ( &s == &n )  return;  // only remove one predecessor per request
                internal_push(s);
            }
        }

        void clear() {
            while( !my_q.empty()) (void)my_q.pop();
#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
            my_built_predecessors.clear();
#endif
        }

#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
        typedef edge_container<T> built_predecessors_type;
        built_predecessors_type &built_predecessors() { return my_built_predecessors; }

        typedef typename edge_container<T>::edge_list_type predecessor_list_type;
        void internal_add_built_predecessor( T &n ) {
            typename mutex_type::scoped_lock lock( my_mutex );
            my_built_predecessors.add_edge(n);
        }

        void internal_delete_built_predecessor( T &n ) {
            typename mutex_type::scoped_lock lock( my_mutex );
            my_built_predecessors.delete_edge(n);
        }

        void copy_predecessors( predecessor_list_type &v) {
            typename mutex_type::scoped_lock lock( my_mutex );
            my_built_predecessors.copy_edges(v);
        }

        size_t predecessor_count() {
            typename mutex_type::scoped_lock lock(my_mutex);
            return (size_t)(my_built_predecessors.edge_count());
        }
#endif  /* TBB_PREVIEW_FLOW_GRAPH_FEATURES */

    protected:

        typedef M mutex_type;
        mutex_type my_mutex;
        std::queue< T * > my_q;
#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
        built_predecessors_type my_built_predecessors;
#endif

        // Assumes lock is held
        inline bool internal_empty( )  {
            return my_q.empty();
        }

        // Assumes lock is held
        inline size_type internal_size( )  {
            return my_q.size();
        }

        // Assumes lock is held
        inline void internal_push( T &n )  {
            my_q.push(&n);
        }

        // Assumes lock is held
        inline T &internal_pop() {
            T *v = my_q.front();
            my_q.pop();
            return *v;
        }

    };

    //! A cache of predecessors that only supports try_get
    template< typename T, typename M=spin_mutex >
    class predecessor_cache : public node_cache< sender<T>, M > {
    public:
        typedef M mutex_type;
        typedef T output_type;
        typedef sender<output_type> predecessor_type;
        typedef receiver<output_type> successor_type;

        predecessor_cache( ) : my_owner( NULL ) { }

        void set_owner( successor_type *owner ) { my_owner = owner; }

        bool get_item( output_type &v ) {

            bool msg = false;

            do {
                predecessor_type *src;
                {
                    typename mutex_type::scoped_lock lock(this->my_mutex);
                    if ( this->internal_empty() ) {
                        break;
                    }
                    src = &this->internal_pop();
                }

                // Try to get from this sender
                msg = src->try_get( v );

                if (msg == false) {
                    // Relinquish ownership of the edge
                    if (my_owner)
                        src->register_successor( *my_owner );
                } else {
                    // Retain ownership of the edge
                    this->add(*src);
                }
            } while ( msg == false );
            return msg;
        }

        // If we are removing arcs (rf_clear_edges), call clear() rather than reset().
        void reset() {
            if (my_owner) {
                for(;;) {
                    predecessor_type *src;
                    {
                        if (this->internal_empty()) break;
                        src = &this->internal_pop();
                    }
                    src->register_successor( *my_owner );
                }
            }
        }

    protected:

#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
        using node_cache< sender<T>, M >::my_built_predecessors;
#endif
        successor_type *my_owner;
    };

    //! An cache of predecessors that supports requests and reservations
    template< typename T, typename M=spin_mutex >
    class reservable_predecessor_cache : public predecessor_cache< T, M > {
    public:
        typedef M mutex_type;
        typedef T output_type;
        typedef sender<T> predecessor_type;
        typedef receiver<T> successor_type;

        reservable_predecessor_cache( ) : reserved_src(NULL) { }

        bool
        try_reserve( output_type &v ) {
            bool msg = false;

            do {
                {
                    typename mutex_type::scoped_lock lock(this->my_mutex);
                    if ( reserved_src || this->internal_empty() )
                        return false;

                    reserved_src = &this->internal_pop();
                }

                // Try to get from this sender
                msg = reserved_src->try_reserve( v );

                if (msg == false) {
                    typename mutex_type::scoped_lock lock(this->my_mutex);
                    // Relinquish ownership of the edge
                    reserved_src->register_successor( *this->my_owner );
                    reserved_src = NULL;
                } else {
                    // Retain ownership of the edge
                    this->add( *reserved_src );
                }
            } while ( msg == false );

            return msg;
        }

        bool
        try_release( ) {
            reserved_src->try_release( );
            reserved_src = NULL;
            return true;
        }

        bool
        try_consume( ) {
            reserved_src->try_consume( );
            reserved_src = NULL;
            return true;
        }

        void reset( ) {
            reserved_src = NULL;
            predecessor_cache<T,M>::reset( );
        }

        void clear() {
            reserved_src = NULL;
            predecessor_cache<T,M>::clear();
        }

    private:
        predecessor_type *reserved_src;
    };


    //! An abstract cache of successors
    template<typename T, typename M=spin_rw_mutex >
    class successor_cache : tbb::internal::no_copy {
    protected:

        typedef M mutex_type;
        mutex_type my_mutex;

        typedef receiver<T> successor_type;
        typedef receiver<T> *pointer_type;
        typedef std::list< pointer_type > successors_type;
#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
        edge_container<successor_type> my_built_successors;
#endif
        successors_type my_successors;

        sender<T> *my_owner;

    public:
#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
        typedef typename edge_container<successor_type>::edge_list_type successor_list_type;

        edge_container<successor_type> &built_successors() { return my_built_successors; }

        void internal_add_built_successor( successor_type &r) {
            typename mutex_type::scoped_lock l(my_mutex, true);
            my_built_successors.add_edge( r );
        }

        void internal_delete_built_successor( successor_type &r) {
            typename mutex_type::scoped_lock l(my_mutex, true);
            my_built_successors.delete_edge(r);
        }

        void copy_successors( successor_list_type &v) {
            typename mutex_type::scoped_lock l(my_mutex, false);
            my_built_successors.copy_edges(v);
        }

        size_t successor_count() {
            typename mutex_type::scoped_lock l(my_mutex,false);
            return my_built_successors.edge_count();
        }

#endif /* TBB_PREVIEW_FLOW_GRAPH_FEATURES */

        successor_cache( ) : my_owner(NULL) {}

        void set_owner( sender<T> *owner ) { my_owner = owner; }

        virtual ~successor_cache() {}

        void register_successor( successor_type &r ) {
            typename mutex_type::scoped_lock l(my_mutex, true);
            my_successors.push_back( &r );
        }

        void remove_successor( successor_type &r ) {
            typename mutex_type::scoped_lock l(my_mutex, true);
            for ( typename successors_type::iterator i = my_successors.begin();
                  i != my_successors.end(); ++i ) {
                if ( *i == & r ) {
                    my_successors.erase(i);
                    break;
                }
            }
        }

        bool empty() {
            typename mutex_type::scoped_lock l(my_mutex, false);
            return my_successors.empty();
        }

        void clear() {
            my_successors.clear();
#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
            my_built_successors.clear();
#endif
        }

        virtual task * try_put_task( const T &t ) = 0;
     };  // successor_cache<T>

    //! An abstract cache of successors, specialized to continue_msg
    template<>
    class successor_cache< continue_msg > : tbb::internal::no_copy {
    protected:

        typedef spin_rw_mutex mutex_type;
        mutex_type my_mutex;

        typedef receiver<continue_msg> successor_type;
        typedef receiver<continue_msg> *pointer_type;
        typedef std::list< pointer_type > successors_type;
        successors_type my_successors;
#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
        edge_container<successor_type> my_built_successors;
        typedef edge_container<successor_type>::edge_list_type successor_list_type;
#endif

        sender<continue_msg> *my_owner;

    public:

#if TBB_PREVIEW_FLOW_GRAPH_FEATURES

        edge_container<successor_type> &built_successors() { return my_built_successors; }

        void internal_add_built_successor( successor_type &r) {
            mutex_type::scoped_lock l(my_mutex, true);
            my_built_successors.add_edge( r );
        }

        void internal_delete_built_successor( successor_type &r) {
            mutex_type::scoped_lock l(my_mutex, true);
            my_built_successors.delete_edge(r);
        }

        void copy_successors( successor_list_type &v) {
            mutex_type::scoped_lock l(my_mutex, false);
            my_built_successors.copy_edges(v);
        }

        size_t successor_count() {
            mutex_type::scoped_lock l(my_mutex,false);
            return my_built_successors.edge_count();
        }

#endif  /* TBB_PREVIEW_FLOW_GRAPH_FEATURES */

        successor_cache( ) : my_owner(NULL) {}

        void set_owner( sender<continue_msg> *owner ) { my_owner = owner; }

        virtual ~successor_cache() {}

        void register_successor( successor_type &r ) {
            mutex_type::scoped_lock l(my_mutex, true);
            my_successors.push_back( &r );
            if ( my_owner && r.is_continue_receiver() ) {
                r.register_predecessor( *my_owner );
            }
        }

        void remove_successor( successor_type &r ) {
            mutex_type::scoped_lock l(my_mutex, true);
            for ( successors_type::iterator i = my_successors.begin();
                  i != my_successors.end(); ++i ) {
                if ( *i == & r ) {
                    // TODO: Check if we need to test for continue_receiver before
                    // removing from r.
                    if ( my_owner )
                        r.remove_predecessor( *my_owner );
                    my_successors.erase(i);
                    break;
                }
            }
        }

        bool empty() {
            mutex_type::scoped_lock l(my_mutex, false);
            return my_successors.empty();
        }

        void clear() {
            my_successors.clear();
#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
            my_built_successors.clear();
#endif
        }

        virtual task * try_put_task( const continue_msg &t ) = 0;

    };  // successor_cache< continue_msg >

    //! A cache of successors that are broadcast to
    template<typename T, typename M=spin_rw_mutex>
    class broadcast_cache : public successor_cache<T, M> {
        typedef M mutex_type;
        typedef typename successor_cache<T,M>::successors_type successors_type;

    public:

        broadcast_cache( ) {}

        // as above, but call try_put_task instead, and return the last task we received (if any)
        /*override*/ task * try_put_task( const T &t ) {
            task * last_task = NULL;
            bool upgraded = true;
            typename mutex_type::scoped_lock l(this->my_mutex, upgraded);
            typename successors_type::iterator i = this->my_successors.begin();
            while ( i != this->my_successors.end() ) {
                task *new_task = (*i)->try_put_task(t);
                last_task = combine_tasks(last_task, new_task);  // enqueue if necessary
                if(new_task) {
                    ++i;
                }
                else {  // failed
                    if ( (*i)->register_predecessor(*this->my_owner) ) {
                        if (!upgraded) {
                            l.upgrade_to_writer();
                            upgraded = true;
                        }
                        i = this->my_successors.erase(i);
                    } else {
                        ++i;
                    }
                }
            }
            return last_task;
        }

    };

    //! A cache of successors that are put in a round-robin fashion
    template<typename T, typename M=spin_rw_mutex >
    class round_robin_cache : public successor_cache<T, M> {
        typedef size_t size_type;
        typedef M mutex_type;
        typedef typename successor_cache<T,M>::successors_type successors_type;

    public:

        round_robin_cache( ) {}

        size_type size() {
            typename mutex_type::scoped_lock l(this->my_mutex, false);
            return this->my_successors.size();
        }

        /*override*/task *try_put_task( const T &t ) {
            bool upgraded = true;
            typename mutex_type::scoped_lock l(this->my_mutex, upgraded);
            typename successors_type::iterator i = this->my_successors.begin();
            while ( i != this->my_successors.end() ) {
                task *new_task = (*i)->try_put_task(t);
                if ( new_task ) {
                    return new_task;
                } else {
                   if ( (*i)->register_predecessor(*this->my_owner) ) {
                       if (!upgraded) {
                           l.upgrade_to_writer();
                           upgraded = true;
                       }
                       i = this->my_successors.erase(i);
                   }
                   else {
                       ++i;
                   }
                }
            }
            return NULL;
        }
    };

    template<typename T>
    class decrementer : public continue_receiver, tbb::internal::no_copy {

        T *my_node;

        task *execute() {
            return my_node->decrement_counter();
        }

    public:

        typedef continue_msg input_type;
        typedef continue_msg output_type;
        decrementer( int number_of_predecessors = 0 ) : continue_receiver( number_of_predecessors ) { }
        void set_owner( T *node ) { my_node = node; }
    };

}

#endif // __TBB__flow_graph_impl_H

