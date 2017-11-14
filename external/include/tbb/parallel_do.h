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

#ifndef __TBB_parallel_do_H
#define __TBB_parallel_do_H

#include "internal/_range_iterator.h"
#include "internal/_template_helpers.h"
#include "task.h"
#include "aligned_space.h"
#include <iterator>

namespace tbb {

//! @cond INTERNAL
namespace internal {
    template<typename Body, typename Item> class parallel_do_feeder_impl;
    template<typename Body> class do_group_task;
} // namespace internal
//! @endcond

//! Class the user supplied algorithm body uses to add new tasks
/** \param Item Work item type **/
template<typename Item>
class parallel_do_feeder: internal::no_copy
{
    parallel_do_feeder() {}
    virtual ~parallel_do_feeder () {}
    virtual void internal_add( const Item& item ) = 0;
    template<typename Body_, typename Item_> friend class internal::parallel_do_feeder_impl;
public:
    //! Add a work item to a running parallel_do.
    // TODO: add an overload for r-value reference
    void add( const Item& item ) {internal_add(item);}
};

//! @cond INTERNAL
namespace internal {
    //! For internal use only.
    /** Selects one of the two possible forms of function call member operator.
        @ingroup algorithms **/
    template<class Body, typename Item>
    class parallel_do_operator_selector
    {
        typedef parallel_do_feeder<Item> Feeder;
        template<typename A1, typename A2, typename CvItem >
        static void internal_call( const Body& obj, A1& arg1, A2&, void (Body::*)(CvItem) const ) {
            obj(arg1);
        }
        template<typename A1, typename A2, typename CvItem >
        static void internal_call( const Body& obj, A1& arg1, A2& arg2, void (Body::*)(CvItem, parallel_do_feeder<Item>&) const ) {
            obj(arg1, arg2);
        }

    public:
        template<typename A1, typename A2 >
        static void call( const Body& obj, A1& arg1, A2& arg2 )
        {
            internal_call( obj, arg1, arg2, &Body::operator() );
        }
    };

    //! For internal use only.
    /** Executes one iteration of a do.
        @ingroup algorithms */
    template<typename Body, typename Item>
    class do_iteration_task: public task
    {
        typedef parallel_do_feeder_impl<Body, Item> feeder_type;

        Item my_value;
        feeder_type& my_feeder;

        do_iteration_task( const Item& value, feeder_type& feeder ) : 
            my_value(value), my_feeder(feeder)
        {}

        /*override*/ 
        task* execute()
        {
            // TODO: use move semantics for my_value
            parallel_do_operator_selector<Body, Item>::call(*my_feeder.my_body, my_value, my_feeder);
            return NULL;
        }

        template<typename Body_, typename Item_> friend class parallel_do_feeder_impl;
    }; // class do_iteration_task

    template<typename Iterator, typename Body, typename Item>
    class do_iteration_task_iter: public task
    {
        typedef parallel_do_feeder_impl<Body, Item> feeder_type;

        Iterator my_iter;
        feeder_type& my_feeder;

        do_iteration_task_iter( const Iterator& iter, feeder_type& feeder ) : 
            my_iter(iter), my_feeder(feeder)
        {}

        /*override*/ 
        task* execute()
        {
            parallel_do_operator_selector<Body, Item>::call(*my_feeder.my_body, *my_iter, my_feeder);
            return NULL;
        }

        template<typename Iterator_, typename Body_, typename Item_> friend class do_group_task_forward;
        template<typename Body_, typename Item_> friend class do_group_task_input;
        template<typename Iterator_, typename Body_, typename Item_> friend class do_task_iter;
    }; // class do_iteration_task_iter

    //! For internal use only.
    /** Implements new task adding procedure.
        @ingroup algorithms **/
    template<class Body, typename Item>
    class parallel_do_feeder_impl : public parallel_do_feeder<Item>
    {
        /*override*/ 
        void internal_add( const Item& item )
        {
            typedef do_iteration_task<Body, Item> iteration_type;

            iteration_type& t = *new (task::allocate_additional_child_of(*my_barrier)) iteration_type(item, *this);

            t.spawn( t );
        }
    public:
        const Body* my_body;
        empty_task* my_barrier;

        parallel_do_feeder_impl()
        {
            my_barrier = new( task::allocate_root() ) empty_task();
            __TBB_ASSERT(my_barrier, "root task allocation failed");
        }

#if __TBB_TASK_GROUP_CONTEXT
        parallel_do_feeder_impl(tbb::task_group_context &context)
        {
            my_barrier = new( task::allocate_root(context) ) empty_task();
            __TBB_ASSERT(my_barrier, "root task allocation failed");
        }
#endif

        ~parallel_do_feeder_impl()
        {
            my_barrier->destroy(*my_barrier);
        }
    }; // class parallel_do_feeder_impl


    //! For internal use only
    /** Unpacks a block of iterations.
        @ingroup algorithms */

    template<typename Iterator, typename Body, typename Item>
    class do_group_task_forward: public task
    {
        static const size_t max_arg_size = 4;

        typedef parallel_do_feeder_impl<Body, Item> feeder_type;

        feeder_type& my_feeder;
        Iterator my_first;
        size_t my_size;

        do_group_task_forward( Iterator first, size_t size, feeder_type& feeder ) 
            : my_feeder(feeder), my_first(first), my_size(size)
        {}

        /*override*/ task* execute()
        {
            typedef do_iteration_task_iter<Iterator, Body, Item> iteration_type;
            __TBB_ASSERT( my_size>0, NULL );
            task_list list;
            task* t; 
            size_t k=0; 
            for(;;) {
                t = new( allocate_child() ) iteration_type( my_first, my_feeder );
                ++my_first;
                if( ++k==my_size ) break;
                list.push_back(*t);
            }
            set_ref_count(int(k+1));
            spawn(list);
            spawn_and_wait_for_all(*t);
            return NULL;
        }

        template<typename Iterator_, typename Body_, typename _Item> friend class do_task_iter;
    }; // class do_group_task_forward

    template<typename Body, typename Item>
    class do_group_task_input: public task
    {
        static const size_t max_arg_size = 4;

        typedef parallel_do_feeder_impl<Body, Item> feeder_type;

        feeder_type& my_feeder;
        size_t my_size;
        aligned_space<Item, max_arg_size> my_arg;

        do_group_task_input( feeder_type& feeder )
            : my_feeder(feeder), my_size(0)
        {}

        /*override*/ task* execute()
        {
            typedef do_iteration_task_iter<Item*, Body, Item> iteration_type;
            __TBB_ASSERT( my_size>0, NULL );
            task_list list;
            task* t; 
            size_t k=0; 
            for(;;) {
                t = new( allocate_child() ) iteration_type( my_arg.begin() + k, my_feeder );
                if( ++k==my_size ) break;
                list.push_back(*t);
            }
            set_ref_count(int(k+1));
            spawn(list);
            spawn_and_wait_for_all(*t);
            return NULL;
        }

        ~do_group_task_input(){
            for( size_t k=0; k<my_size; ++k)
                (my_arg.begin() + k)->~Item();
        }

        template<typename Iterator_, typename Body_, typename Item_> friend class do_task_iter;
    }; // class do_group_task_input

    //! For internal use only.
    /** Gets block of iterations and packages them into a do_group_task.
        @ingroup algorithms */
    template<typename Iterator, typename Body, typename Item>
    class do_task_iter: public task
    {
        typedef parallel_do_feeder_impl<Body, Item> feeder_type;

    public:
        do_task_iter( Iterator first, Iterator last , feeder_type& feeder ) : 
            my_first(first), my_last(last), my_feeder(feeder)
        {}

    private:
        Iterator my_first;
        Iterator my_last;
        feeder_type& my_feeder;

        /* Do not merge run(xxx) and run_xxx() methods. They are separated in order
            to make sure that compilers will eliminate unused argument of type xxx
            (that is will not put it on stack). The sole purpose of this argument 
            is overload resolution.

            An alternative could be using template functions, but explicit specialization 
            of member function templates is not supported for non specialized class 
            templates. Besides template functions would always fall back to the least 
            efficient variant (the one for input iterators) in case of iterators having 
            custom tags derived from basic ones. */
        /*override*/ task* execute()
        {
            typedef typename std::iterator_traits<Iterator>::iterator_category iterator_tag;
            return run( (iterator_tag*)NULL );
        }

        /** This is the most restricted variant that operates on input iterators or
            iterators with unknown tags (tags not derived from the standard ones). **/
        inline task* run( void* ) { return run_for_input_iterator(); }

        task* run_for_input_iterator() {
            typedef do_group_task_input<Body, Item> block_type;

            block_type& t = *new( allocate_additional_child_of(*my_feeder.my_barrier) ) block_type(my_feeder);
            size_t k=0; 
            while( !(my_first == my_last) ) {
                // TODO: move *my_first
                new (t.my_arg.begin() + k) Item(*my_first);
                ++my_first;
                if( ++k==block_type::max_arg_size ) {
                    if ( !(my_first == my_last) )
                        recycle_to_reexecute();
                    break;
                }
            }
            if( k==0 ) {
                destroy(t);
                return NULL;
            } else {
                t.my_size = k;
                return &t;
            }
        }

        inline task* run( std::forward_iterator_tag* ) { return run_for_forward_iterator(); }

        task* run_for_forward_iterator() {
            typedef do_group_task_forward<Iterator, Body, Item> block_type;

            Iterator first = my_first;
            size_t k=0; 
            while( !(my_first==my_last) ) {
                ++my_first;
                if( ++k==block_type::max_arg_size ) {
                    if ( !(my_first==my_last) )
                        recycle_to_reexecute();
                    break;
                }
            }
            return k==0 ? NULL : new( allocate_additional_child_of(*my_feeder.my_barrier) ) block_type(first, k, my_feeder);
        }

        inline task* run( std::random_access_iterator_tag* ) { return run_for_random_access_iterator(); }

        task* run_for_random_access_iterator() {
            typedef do_group_task_forward<Iterator, Body, Item> block_type;
            typedef do_iteration_task_iter<Iterator, Body, Item> iteration_type;

            size_t k = static_cast<size_t>(my_last-my_first); 
            if( k > block_type::max_arg_size ) {
                Iterator middle = my_first + k/2;

                empty_task& c = *new( allocate_continuation() ) empty_task;
                do_task_iter& b = *new( c.allocate_child() ) do_task_iter(middle, my_last, my_feeder);
                recycle_as_child_of(c);

                my_last = middle;
                c.set_ref_count(2);
                c.spawn(b);
                return this;
            }else if( k != 0 ) {
                task_list list;
                task* t; 
                size_t k1=0; 
                for(;;) {
                    t = new( allocate_child() ) iteration_type(my_first, my_feeder);
                    ++my_first;
                    if( ++k1==k ) break;
                    list.push_back(*t);
                }
                set_ref_count(int(k+1));
                spawn(list);
                spawn_and_wait_for_all(*t);
            }
            return NULL;
        }
    }; // class do_task_iter

    //! For internal use only.
    /** Implements parallel iteration over a range.
        @ingroup algorithms */
    template<typename Iterator, typename Body, typename Item> 
    void run_parallel_do( Iterator first, Iterator last, const Body& body
#if __TBB_TASK_GROUP_CONTEXT
        , task_group_context& context
#endif
        )
    {
        typedef do_task_iter<Iterator, Body, Item> root_iteration_task;
#if __TBB_TASK_GROUP_CONTEXT
        parallel_do_feeder_impl<Body, Item> feeder(context);
#else
        parallel_do_feeder_impl<Body, Item> feeder;
#endif
        feeder.my_body = &body;

        root_iteration_task &t = *new( feeder.my_barrier->allocate_child() ) root_iteration_task(first, last, feeder);

        feeder.my_barrier->set_ref_count(2);
        feeder.my_barrier->spawn_and_wait_for_all(t);
    }

    //! For internal use only.
    /** Detects types of Body's operator function arguments.
        @ingroup algorithms **/
    template<typename Iterator, typename Body, typename Item> 
    void select_parallel_do( Iterator first, Iterator last, const Body& body, void (Body::*)(Item) const
#if __TBB_TASK_GROUP_CONTEXT
        , task_group_context& context 
#endif // __TBB_TASK_GROUP_CONTEXT 
        )
    {
        run_parallel_do<Iterator, Body, typename strip<Item>::type>( first, last, body
#if __TBB_TASK_GROUP_CONTEXT
            , context
#endif // __TBB_TASK_GROUP_CONTEXT 
            );
    }

    //! For internal use only.
    /** Detects types of Body's operator function arguments.
        @ingroup algorithms **/
    template<typename Iterator, typename Body, typename Item, typename _Item> 
    void select_parallel_do( Iterator first, Iterator last, const Body& body, void (Body::*)(Item, parallel_do_feeder<_Item>&) const
#if __TBB_TASK_GROUP_CONTEXT
        , task_group_context& context 
#endif // __TBB_TASK_GROUP_CONTEXT
        )
    {
        run_parallel_do<Iterator, Body, typename strip<Item>::type>( first, last, body
#if __TBB_TASK_GROUP_CONTEXT
            , context
#endif // __TBB_TASK_GROUP_CONTEXT
            );
    }

} // namespace internal
//! @endcond


/** \page parallel_do_body_req Requirements on parallel_do body
    Class \c Body implementing the concept of parallel_do body must define:
    - \code 
        B::operator()( 
                cv_item_type item,
                parallel_do_feeder<item_type>& feeder
        ) const

        OR

        B::operator()( cv_item_type& item ) const
      \endcode                                               Process item. 
                                                             May be invoked concurrently  for the same \c this but different \c item.

    - \code item_type( const item_type& ) \endcode 
                                                             Copy a work item.
    - \code ~item_type() \endcode                            Destroy a work item
**/

/** \name parallel_do
    See also requirements on \ref parallel_do_body_req "parallel_do Body". **/
//@{
//! Parallel iteration over a range, with optional addition of more work.
/** @ingroup algorithms */
template<typename Iterator, typename Body> 
void parallel_do( Iterator first, Iterator last, const Body& body )
{
    if ( first == last )
        return;
#if __TBB_TASK_GROUP_CONTEXT
    task_group_context context;
#endif // __TBB_TASK_GROUP_CONTEXT
    internal::select_parallel_do( first, last, body, &Body::operator()
#if __TBB_TASK_GROUP_CONTEXT
        , context
#endif // __TBB_TASK_GROUP_CONTEXT
        );
}

template<typename Range, typename Body>
void parallel_do(Range& rng, const Body& body) {
    parallel_do(tbb::internal::first(rng), tbb::internal::last(rng), body);
}

template<typename Range, typename Body>
void parallel_do(const Range& rng, const Body& body) {
    parallel_do(tbb::internal::first(rng), tbb::internal::last(rng), body);
}

#if __TBB_TASK_GROUP_CONTEXT
//! Parallel iteration over a range, with optional addition of more work and user-supplied context
/** @ingroup algorithms */
template<typename Iterator, typename Body> 
void parallel_do( Iterator first, Iterator last, const Body& body, task_group_context& context  )
{
    if ( first == last )
        return;
    internal::select_parallel_do( first, last, body, &Body::operator(), context );
}

template<typename Range, typename Body>
void parallel_do(Range& rng, const Body& body, task_group_context& context) {
    parallel_do(tbb::internal::first(rng), tbb::internal::last(rng), body, context);
}

template<typename Range, typename Body>
void parallel_do(const Range& rng, const Body& body, task_group_context& context) {
    parallel_do(tbb::internal::first(rng), tbb::internal::last(rng), body, context);
}

#endif // __TBB_TASK_GROUP_CONTEXT

//@}

} // namespace 

#endif /* __TBB_parallel_do_H */
