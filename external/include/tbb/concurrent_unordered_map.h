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

/* Container implementations in this header are based on PPL implementations
   provided by Microsoft. */

#ifndef __TBB_concurrent_unordered_map_H
#define __TBB_concurrent_unordered_map_H

#include "internal/_concurrent_unordered_impl.h"

namespace tbb
{

namespace interface5 {

// Template class for hash map traits
template<typename Key, typename T, typename Hash_compare, typename Allocator, bool Allow_multimapping>
class concurrent_unordered_map_traits
{
protected:
    typedef std::pair<const Key, T> value_type;
    typedef Key key_type;
    typedef Hash_compare hash_compare;
    typedef typename Allocator::template rebind<value_type>::other allocator_type;
    enum { allow_multimapping = Allow_multimapping };

    concurrent_unordered_map_traits() : my_hash_compare() {}
    concurrent_unordered_map_traits(const hash_compare& hc) : my_hash_compare(hc) {}

    class value_compare : public std::binary_function<value_type, value_type, bool>
    {
        friend class concurrent_unordered_map_traits<Key, T, Hash_compare, Allocator, Allow_multimapping>;

    public:
        bool operator()(const value_type& left, const value_type& right) const
        {
            return (my_hash_compare(left.first, right.first));
        }

        value_compare(const hash_compare& comparator) : my_hash_compare(comparator) {}

    protected:
        hash_compare my_hash_compare;    // the comparator predicate for keys
    };

    template<class Type1, class Type2>
    static const Key& get_key(const std::pair<Type1, Type2>& value) {
        return (value.first);
    }

    hash_compare my_hash_compare; // the comparator predicate for keys
};

template <typename Key, typename T, typename Hasher = tbb::tbb_hash<Key>, typename Key_equality = std::equal_to<Key>,
         typename Allocator = tbb::tbb_allocator<std::pair<const Key, T> > >
class concurrent_unordered_map :
    public internal::concurrent_unordered_base< concurrent_unordered_map_traits<Key, T,
    internal::hash_compare<Key, Hasher, Key_equality>, Allocator, false> >
{
    // Base type definitions
    typedef internal::hash_compare<Key, Hasher, Key_equality> hash_compare;
    typedef concurrent_unordered_map_traits<Key, T, hash_compare, Allocator, false> traits_type;
    typedef internal::concurrent_unordered_base< traits_type > base_type;
#if __TBB_EXTRA_DEBUG
public:
#endif
    using traits_type::allow_multimapping;
public:
    using base_type::end;
    using base_type::find;
    using base_type::insert;

    // Type definitions
    typedef Key key_type;
    typedef typename base_type::value_type value_type;
    typedef T mapped_type;
    typedef Hasher hasher;
    typedef Key_equality key_equal;
    typedef hash_compare key_compare;

    typedef typename base_type::allocator_type allocator_type;
    typedef typename base_type::pointer pointer;
    typedef typename base_type::const_pointer const_pointer;
    typedef typename base_type::reference reference;
    typedef typename base_type::const_reference const_reference;

    typedef typename base_type::size_type size_type;
    typedef typename base_type::difference_type difference_type;

    typedef typename base_type::iterator iterator;
    typedef typename base_type::const_iterator const_iterator;
    typedef typename base_type::iterator local_iterator;
    typedef typename base_type::const_iterator const_local_iterator;

    // Construction/destruction/copying
    explicit concurrent_unordered_map(size_type n_of_buckets = base_type::initial_bucket_number,
        const hasher& _Hasher = hasher(), const key_equal& _Key_equality = key_equal(),
        const allocator_type& a = allocator_type())
        : base_type(n_of_buckets, key_compare(_Hasher, _Key_equality), a)
    {
    }

    concurrent_unordered_map(const Allocator& a) : base_type(base_type::initial_bucket_number, key_compare(), a)
    {
    }

    template <typename Iterator>
    concurrent_unordered_map(Iterator first, Iterator last, size_type n_of_buckets = base_type::initial_bucket_number,
        const hasher& _Hasher = hasher(), const key_equal& _Key_equality = key_equal(),
        const allocator_type& a = allocator_type())
        : base_type(n_of_buckets, key_compare(_Hasher, _Key_equality), a)
    {
        insert(first, last);
    }

#if __TBB_INITIALIZER_LISTS_PRESENT
    //! Constructor from initializer_list
    concurrent_unordered_map(std::initializer_list<value_type> il, size_type n_of_buckets = base_type::initial_bucket_number,
        const hasher& _Hasher = hasher(), const key_equal& _Key_equality = key_equal(),
        const allocator_type& a = allocator_type())
        : base_type(n_of_buckets, key_compare(_Hasher, _Key_equality), a)
    {
        this->insert(il.begin(),il.end());
    }
#endif //# __TBB_INITIALIZER_LISTS_PRESENT

#if __TBB_CPP11_RVALUE_REF_PRESENT && __TBB_CPP11_IMPLICIT_MOVE_MEMBERS_GENERATION_FOR_DERIVED_BROKEN
    concurrent_unordered_map(const concurrent_unordered_map& table)
        : base_type(table)
    {
    }

    concurrent_unordered_map& operator=(const concurrent_unordered_map& table)
    {
        return static_cast<concurrent_unordered_map&>(base_type::operator=(table));
    }

    concurrent_unordered_map(concurrent_unordered_map&& table)
        : base_type(std::move(table))
    {
    }

    concurrent_unordered_map& operator=(concurrent_unordered_map&& table)
    {
        return static_cast<concurrent_unordered_map&>(base_type::operator=(std::move(table)));
    }
#endif //__TBB_CPP11_IMPLICIT_MOVE_MEMBERS_GENERATION_FOR_DERIVED_BROKEN

    concurrent_unordered_map(const concurrent_unordered_map& table, const Allocator& a)
        : base_type(table, a)
    {
    }

#if __TBB_CPP11_RVALUE_REF_PRESENT
    concurrent_unordered_map(concurrent_unordered_map&& table, const Allocator& a) : base_type(std::move(table), a)
    {
    }
#endif
    // Observers
    mapped_type& operator[](const key_type& key)
    {
        iterator where = find(key);

        if (where == end())
        {
            where = insert(std::pair<key_type, mapped_type>(key, mapped_type())).first;
        }

        return ((*where).second);
    }

    mapped_type& at(const key_type& key)
    {
        iterator where = find(key);

        if (where == end())
        {
            tbb::internal::throw_exception(tbb::internal::eid_invalid_key);
        }

        return ((*where).second);
    }

    const mapped_type& at(const key_type& key) const
    {
        const_iterator where = find(key);

        if (where == end())
        {
            tbb::internal::throw_exception(tbb::internal::eid_invalid_key);
        }

        return ((*where).second);
    }
};

template < typename Key, typename T, typename Hasher = tbb::tbb_hash<Key>, typename Key_equality = std::equal_to<Key>,
        typename Allocator = tbb::tbb_allocator<std::pair<const Key, T> > >
class concurrent_unordered_multimap :
    public internal::concurrent_unordered_base< concurrent_unordered_map_traits< Key, T,
    internal::hash_compare<Key, Hasher, Key_equality>, Allocator, true> >
{
    // Base type definitions
    typedef internal::hash_compare<Key, Hasher, Key_equality> hash_compare;
    typedef concurrent_unordered_map_traits<Key, T, hash_compare, Allocator, true> traits_type;
    typedef internal::concurrent_unordered_base<traits_type> base_type;
#if __TBB_EXTRA_DEBUG
public:
#endif
    using traits_type::allow_multimapping;
public:
    using base_type::insert;

    // Type definitions
    typedef Key key_type;
    typedef typename base_type::value_type value_type;
    typedef T mapped_type;
    typedef Hasher hasher;
    typedef Key_equality key_equal;
    typedef hash_compare key_compare;

    typedef typename base_type::allocator_type allocator_type;
    typedef typename base_type::pointer pointer;
    typedef typename base_type::const_pointer const_pointer;
    typedef typename base_type::reference reference;
    typedef typename base_type::const_reference const_reference;

    typedef typename base_type::size_type size_type;
    typedef typename base_type::difference_type difference_type;

    typedef typename base_type::iterator iterator;
    typedef typename base_type::const_iterator const_iterator;
    typedef typename base_type::iterator local_iterator;
    typedef typename base_type::const_iterator const_local_iterator;

    // Construction/destruction/copying
    explicit concurrent_unordered_multimap(size_type n_of_buckets = base_type::initial_bucket_number,
        const hasher& _Hasher = hasher(), const key_equal& _Key_equality = key_equal(),
        const allocator_type& a = allocator_type())
        : base_type(n_of_buckets, key_compare(_Hasher, _Key_equality), a)
    {
    }

    concurrent_unordered_multimap(const Allocator& a) : base_type(base_type::initial_bucket_number, key_compare(), a)
    {
    }

    template <typename Iterator>
    concurrent_unordered_multimap(Iterator first, Iterator last, size_type n_of_buckets = base_type::initial_bucket_number,
        const hasher& _Hasher = hasher(), const key_equal& _Key_equality = key_equal(),
        const allocator_type& a = allocator_type())
        : base_type(n_of_buckets,key_compare(_Hasher,_Key_equality), a)
    {
        insert(first, last);
    }

#if __TBB_INITIALIZER_LISTS_PRESENT
    //! Constructor from initializer_list
    concurrent_unordered_multimap(std::initializer_list<value_type> il, size_type n_of_buckets = base_type::initial_bucket_number,
        const hasher& _Hasher = hasher(), const key_equal& _Key_equality = key_equal(),
        const allocator_type& a = allocator_type())
        : base_type(n_of_buckets, key_compare(_Hasher, _Key_equality), a)
    {
        this->insert(il.begin(),il.end());
    }
#endif //# __TBB_INITIALIZER_LISTS_PRESENT

#if __TBB_CPP11_RVALUE_REF_PRESENT && __TBB_CPP11_IMPLICIT_MOVE_MEMBERS_GENERATION_FOR_DERIVED_BROKEN
    concurrent_unordered_multimap(const concurrent_unordered_multimap& table)
        : base_type(table)
    {
    }

    concurrent_unordered_multimap& operator=(const concurrent_unordered_multimap& table)
    {
        return static_cast<concurrent_unordered_multimap&>(base_type::operator=(table));
    }

    concurrent_unordered_multimap(concurrent_unordered_multimap&& table)
        : base_type(std::move(table))
    {
    }

    concurrent_unordered_multimap& operator=(concurrent_unordered_multimap&& table)
    {
        return static_cast<concurrent_unordered_multimap&>(base_type::operator=(std::move(table)));
    }
#endif //__TBB_CPP11_IMPLICIT_MOVE_MEMBERS_GENERATION_FOR_DERIVED_BROKEN

    concurrent_unordered_multimap(const concurrent_unordered_multimap& table, const Allocator& a)
        : base_type(table, a)
    {
    }

#if __TBB_CPP11_RVALUE_REF_PRESENT
    concurrent_unordered_multimap(concurrent_unordered_multimap&& table, const Allocator& a) : base_type(std::move(table), a)
    {
    }
#endif
};
} // namespace interface5

using interface5::concurrent_unordered_map;
using interface5::concurrent_unordered_multimap;

} // namespace tbb

#endif// __TBB_concurrent_unordered_map_H
