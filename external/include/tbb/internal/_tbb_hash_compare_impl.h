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

// must be included outside namespaces.
#ifndef __TBB_tbb_hash_compare_impl_H
#define __TBB_tbb_hash_compare_impl_H

#include <string>

namespace tbb {
namespace interface5 {
namespace internal {    

// Template class for hash compare
template<typename Key, typename Hasher, typename Key_equality>
class hash_compare
{
public:
    typedef Hasher hasher;
    typedef Key_equality key_equal;

    hash_compare() {}

    hash_compare(Hasher a_hasher) : my_hash_object(a_hasher) {}

    hash_compare(Hasher a_hasher, Key_equality a_keyeq) : my_hash_object(a_hasher), my_key_compare_object(a_keyeq) {}

    size_t operator()(const Key& key) const {
        return ((size_t)my_hash_object(key));
    }

    bool operator()(const Key& key1, const Key& key2) const {
        return (!my_key_compare_object(key1, key2));
    }

    Hasher       my_hash_object;        // The hash object
    Key_equality my_key_compare_object; // The equality comparator object
};

//! Hash multiplier
static const size_t hash_multiplier = tbb::internal::select_size_t_constant<2654435769U, 11400714819323198485ULL>::value;

} // namespace internal

//! Hasher functions
template<typename T>
inline size_t tbb_hasher( const T& t ) {
    return static_cast<size_t>( t ) * internal::hash_multiplier;
}
template<typename P>
inline size_t tbb_hasher( P* ptr ) {
    size_t const h = reinterpret_cast<size_t>( ptr );
    return (h >> 3) ^ h;
}
template<typename E, typename S, typename A>
inline size_t tbb_hasher( const std::basic_string<E,S,A>& s ) {
    size_t h = 0;
    for( const E* c = s.c_str(); *c; ++c )
        h = static_cast<size_t>(*c) ^ (h * internal::hash_multiplier);
    return h;
}
template<typename F, typename S>
inline size_t tbb_hasher( const std::pair<F,S>& p ) {
    return tbb_hasher(p.first) ^ tbb_hasher(p.second);
}

} // namespace interface5
using interface5::tbb_hasher;

// Template class for hash compare
template<typename Key>
class tbb_hash
{
public:
    tbb_hash() {}

    size_t operator()(const Key& key) const
    {
        return tbb_hasher(key);
    }
};

//! hash_compare that is default argument for concurrent_hash_map
template<typename Key>
struct tbb_hash_compare {
    static size_t hash( const Key& a ) { return tbb_hasher(a); }
    static bool equal( const Key& a, const Key& b ) { return a == b; }
};

}  // namespace tbb
#endif  /*  __TBB_tbb_hash_compare_impl_H */
