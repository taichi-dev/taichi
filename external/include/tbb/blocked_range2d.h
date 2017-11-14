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

#ifndef __TBB_blocked_range2d_H
#define __TBB_blocked_range2d_H

#include "tbb_stddef.h"
#include "blocked_range.h"

namespace tbb {

//! A 2-dimensional range that models the Range concept.
/** @ingroup algorithms */
template<typename RowValue, typename ColValue=RowValue>
class blocked_range2d {
public:
    //! Type for size of an iteration range
    typedef blocked_range<RowValue> row_range_type;
    typedef blocked_range<ColValue> col_range_type;

private:
    row_range_type my_rows;
    col_range_type my_cols;

public:

    blocked_range2d( RowValue row_begin, RowValue row_end, typename row_range_type::size_type row_grainsize,
                     ColValue col_begin, ColValue col_end, typename col_range_type::size_type col_grainsize ) :
        my_rows(row_begin,row_end,row_grainsize),
        my_cols(col_begin,col_end,col_grainsize)
    {
    }

    blocked_range2d( RowValue row_begin, RowValue row_end,
                     ColValue col_begin, ColValue col_end ) :
        my_rows(row_begin,row_end),
        my_cols(col_begin,col_end)
    {
    }

    //! True if range is empty
    bool empty() const {
        // Yes, it is a logical OR here, not AND.
        return my_rows.empty() || my_cols.empty();
    }

    //! True if range is divisible into two pieces.
    bool is_divisible() const {
        return my_rows.is_divisible() || my_cols.is_divisible();
    }

    blocked_range2d( blocked_range2d& r, split ) :
        my_rows(r.my_rows),
        my_cols(r.my_cols)
    {
        split split_obj;
        do_split(r, split_obj);
    }

#if __TBB_USE_PROPORTIONAL_SPLIT_IN_BLOCKED_RANGES
    //! Static field to support proportional split
    static const bool is_splittable_in_proportion = true;

    blocked_range2d( blocked_range2d& r, proportional_split& proportion ) :
        my_rows(r.my_rows),
        my_cols(r.my_cols)
    {
        do_split(r, proportion);
    }
#endif /* __TBB_USE_PROPORTIONAL_SPLIT_IN_BLOCKED_RANGES */

    template <typename Split>
    void do_split( blocked_range2d& r, Split& split_obj )
    {
        if( my_rows.size()*double(my_cols.grainsize()) < my_cols.size()*double(my_rows.grainsize()) ) {
            my_cols.my_begin = col_range_type::do_split(r.my_cols, split_obj);
        } else {
            my_rows.my_begin = row_range_type::do_split(r.my_rows, split_obj);
        }
    }

    //! The rows of the iteration space
    const row_range_type& rows() const {return my_rows;}

    //! The columns of the iteration space
    const col_range_type& cols() const {return my_cols;}
};

} // namespace tbb

#endif /* __TBB_blocked_range2d_H */
