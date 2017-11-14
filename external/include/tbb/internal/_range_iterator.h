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

#ifndef __TBB_range_iterator_H
#define __TBB_range_iterator_H

#include "../tbb_stddef.h"

#if __TBB_CPP11_STD_BEGIN_END_PRESENT && __TBB_CPP11_AUTO_PRESENT && __TBB_CPP11_DECLTYPE_PRESENT
    #include <iterator>
#endif

namespace tbb {
    // iterators to first and last elements of container
    namespace internal {

#if __TBB_CPP11_STD_BEGIN_END_PRESENT && __TBB_CPP11_AUTO_PRESENT && __TBB_CPP11_DECLTYPE_PRESENT
        using std::begin;
        using std::end;
        template<typename Container>
        auto first(Container& c)-> decltype(begin(c))  {return begin(c);}

        template<typename Container>
        auto first(const Container& c)-> decltype(begin(c))  {return begin(c);}

        template<typename Container>
        auto last(Container& c)-> decltype(begin(c))  {return end(c);}

        template<typename Container>
        auto last(const Container& c)-> decltype(begin(c)) {return end(c);}
#else
        template<typename Container>
        typename Container::iterator first(Container& c) {return c.begin();}

        template<typename Container>
        typename Container::const_iterator first(const Container& c) {return c.begin();}

        template<typename Container>
        typename Container::iterator last(Container& c) {return c.end();}

        template<typename Container>
        typename Container::const_iterator last(const Container& c) {return c.end();}
#endif

        template<typename T, size_t size>
        T* first(T (&arr) [size]) {return arr;}

        template<typename T, size_t size>
        T* last(T (&arr) [size]) {return arr + size;}
    } //namespace internal
}  //namespace tbb

#endif // __TBB_range_iterator_H
