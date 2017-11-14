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

#ifndef __TBB_blocked_range_H
#define __TBB_blocked_range_H

#include "tbb_stddef.h"

namespace tbb {

/** \page range_req Requirements on range concept
    Class \c R implementing the concept of range must define:
    - \code R::R( const R& ); \endcode               Copy constructor
    - \code R::~R(); \endcode                        Destructor
    - \code bool R::is_divisible() const; \endcode   True if range can be partitioned into two subranges
    - \code bool R::empty() const; \endcode          True if range is empty
    - \code R::R( R& r, split ); \endcode            Split range \c r into two subranges.
**/

//! A range over which to iterate.
/** @ingroup algorithms */
template<typename Value>
class blocked_range {
public:
    //! Type of a value
    /** Called a const_iterator for sake of algorithms that need to treat a blocked_range
        as an STL container. */
    typedef Value const_iterator;

    //! Type for size of a range
    typedef std::size_t size_type;

    //! Construct range with default-constructed values for begin and end.
    /** Requires that Value have a default constructor. */
    blocked_range() : my_end(), my_begin() {}

    //! Construct range over half-open interval [begin,end), with the given grainsize.
    blocked_range( Value begin_, Value end_, size_type grainsize_=1 ) :
        my_end(end_), my_begin(begin_), my_grainsize(grainsize_)
    {
        __TBB_ASSERT( my_grainsize>0, "grainsize must be positive" );
    }

    //! Beginning of range.
    const_iterator begin() const {return my_begin;}

    //! One past last value in range.
    const_iterator end() const {return my_end;}

    //! Size of the range
    /** Unspecified if end()<begin(). */
    size_type size() const {
        __TBB_ASSERT( !(end()<begin()), "size() unspecified if end()<begin()" );
        return size_type(my_end-my_begin);
    }

    //! The grain size for this range.
    size_type grainsize() const {return my_grainsize;}

    //------------------------------------------------------------------------
    // Methods that implement Range concept
    //------------------------------------------------------------------------

    //! True if range is empty.
    bool empty() const {return !(my_begin<my_end);}

    //! True if range is divisible.
    /** Unspecified if end()<begin(). */
    bool is_divisible() const {return my_grainsize<size();}

    //! Split range.
    /** The new Range *this has the second part, the old range r has the first part.
        Unspecified if end()<begin() or !is_divisible(). */
    blocked_range( blocked_range& r, split ) :
        my_end(r.my_end),
        my_begin(do_split(r, split())),
        my_grainsize(r.my_grainsize)
    {
        // only comparison 'less than' is required from values of blocked_range objects
        __TBB_ASSERT( !(my_begin < r.my_end) && !(r.my_end < my_begin), "blocked_range has been split incorrectly" );
    }

#if __TBB_USE_PROPORTIONAL_SPLIT_IN_BLOCKED_RANGES
    //! Static field to support proportional split
    static const bool is_splittable_in_proportion = true;

    //! Split range.
    /** The new Range *this has the second part split according to specified proportion, the old range r has the first part.
        Unspecified if end()<begin() or !is_divisible(). */
    blocked_range( blocked_range& r, proportional_split& proportion ) :
        my_end(r.my_end),
        my_begin(do_split(r, proportion)),
        my_grainsize(r.my_grainsize)
    {
        // only comparison 'less than' is required from values of blocked_range objects
        __TBB_ASSERT( !(my_begin < r.my_end) && !(r.my_end < my_begin), "blocked_range has been split incorrectly" );
    }
#endif /* __TBB_USE_PROPORTIONAL_SPLIT_IN_BLOCKED_RANGES */

private:
    /** NOTE: my_end MUST be declared before my_begin, otherwise the forking constructor will break. */
    Value my_end;
    Value my_begin;
    size_type my_grainsize;

    //! Auxiliary function used by forking constructor.
    /** Using this function lets us not require that Value support assignment or default construction. */
    static Value do_split( blocked_range& r, split )
    {
        __TBB_ASSERT( r.is_divisible(), "cannot split blocked_range that is not divisible" );
        Value middle = r.my_begin + (r.my_end - r.my_begin) / 2u;
        r.my_end = middle;
        return middle;
    }

#if __TBB_USE_PROPORTIONAL_SPLIT_IN_BLOCKED_RANGES
    static Value do_split( blocked_range& r, proportional_split& proportion )
    {
        __TBB_ASSERT( r.is_divisible(), "cannot split blocked_range that is not divisible" );

        // usage of 32-bit floating point arithmetic is not enough to handle ranges of
        // more than 2^24 iterations accurately. However, even on ranges with 2^64
        // iterations the computational error approximately equals to 0.000001% which
        // makes small impact on uniform distribution of such range's iterations (assuming
        // all iterations take equal time to complete). See 'test_partitioner_whitebox'
        // for implementation of an exact split algorithm
        size_type right_part = size_type(float(r.size()) * float(proportion.right())
                                         / float(proportion.left() + proportion.right()) + 0.5f);
        return r.my_end = Value(r.my_end - right_part);
    }
#endif /* __TBB_USE_PROPORTIONAL_SPLIT_IN_BLOCKED_RANGES */

    template<typename RowValue, typename ColValue>
    friend class blocked_range2d;

    template<typename RowValue, typename ColValue, typename PageValue>
    friend class blocked_range3d;
};

} // namespace tbb

#endif /* __TBB_blocked_range_H */
