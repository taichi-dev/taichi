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

#ifndef __TBB_template_helpers_H
#define __TBB_template_helpers_H

#include <utility>

namespace tbb { namespace internal {

//! Enables one or the other code branches
template<bool Condition, typename T = void> struct enable_if {};
template<typename T> struct enable_if<true, T> { typedef T type; };

//! Strips its template type argument from cv- and ref-qualifiers
template<typename T> struct strip                     { typedef T type; };
template<typename T> struct strip<const T>            { typedef T type; };
template<typename T> struct strip<volatile T>         { typedef T type; };
template<typename T> struct strip<const volatile T>   { typedef T type; };
template<typename T> struct strip<T&>                 { typedef T type; };
template<typename T> struct strip<const T&>           { typedef T type; };
template<typename T> struct strip<volatile T&>        { typedef T type; };
template<typename T> struct strip<const volatile T&>  { typedef T type; };
//! Specialization for function pointers
template<typename T> struct strip<T(&)()>             { typedef T(*type)(); };
#if __TBB_CPP11_RVALUE_REF_PRESENT
template<typename T> struct strip<T&&>                { typedef T type; };
template<typename T> struct strip<const T&&>          { typedef T type; };
template<typename T> struct strip<volatile T&&>       { typedef T type; };
template<typename T> struct strip<const volatile T&&> { typedef T type; };
#endif
//! Specialization for arrays converts to a corresponding pointer
template<typename T, size_t N> struct strip<T(&)[N]>                { typedef T* type; };
template<typename T, size_t N> struct strip<const T(&)[N]>          { typedef const T* type; };
template<typename T, size_t N> struct strip<volatile T(&)[N]>       { typedef volatile T* type; };
template<typename T, size_t N> struct strip<const volatile T(&)[N]> { typedef const volatile T* type; };

//! Detects whether two given types are the same
template<class U, class V> struct is_same_type      { static const bool value = false; };
template<class W>          struct is_same_type<W,W> { static const bool value = true; };

template<typename T> struct is_ref { static const bool value = false; };
template<typename U> struct is_ref<U&> { static const bool value = true; };

#if __TBB_CPP11_RVALUE_REF_PRESENT && __TBB_CPP11_VARIADIC_TEMPLATES_PRESENT

//! Allows to store a function parameter pack as a variable and later pass it to another function
template< typename... Types >
struct stored_pack;

template<>
struct stored_pack<>
{
    typedef stored_pack<> pack_type;
    stored_pack() {}

    // Friend front-end functions
    template< typename F, typename Pack > friend void call( F&& f, Pack&& p );
    template< typename Ret, typename F, typename Pack > friend Ret call_and_return( F&& f, Pack&& p );

protected:
    // Ideally, ref-qualified non-static methods would be used,
    // but that would greatly reduce the set of compilers where it works.
    template< typename Ret, typename F, typename... Preceding >
    static Ret call( F&& f, const pack_type& /*pack*/, Preceding&&... params ) {
        return std::forward<F>(f)( std::forward<Preceding>(params)... );
    }
    template< typename Ret, typename F, typename... Preceding >
    static Ret call( F&& f, pack_type&& /*pack*/, Preceding&&... params ) {
        return std::forward<F>(f)( std::forward<Preceding>(params)... );
    }
};

template< typename T, typename... Types >
struct stored_pack<T, Types...> : stored_pack<Types...>
{
    typedef stored_pack<T, Types...> pack_type;
    typedef stored_pack<Types...> pack_remainder;
    // Since lifetime of original values is out of control, copies should be made.
    // Thus references should be stripped away from the deduced type.
    typename strip<T>::type leftmost_value;

    // Here rvalue references act in the same way as forwarding references,
    // as long as class template parameters were deduced via forwarding references.
    stored_pack( T&& t, Types&&... types )
    : pack_remainder(std::forward<Types>(types)...), leftmost_value(std::forward<T>(t)) {}

    // Friend front-end functions
    template< typename F, typename Pack > friend void call( F&& f, Pack&& p );
    template< typename Ret, typename F, typename Pack > friend Ret call_and_return( F&& f, Pack&& p );

protected:
    template< typename Ret, typename F, typename... Preceding >
    static Ret call( F&& f, const pack_type& pack, Preceding&&... params ) {
        return pack_remainder::template call<Ret>(
            std::forward<F>(f), static_cast<const pack_remainder&>(pack),
            std::forward<Preceding>(params)... , pack.leftmost_value
        );
    }
    template< typename Ret, typename F, typename... Preceding >
    static Ret call( F&& f, pack_type&& pack, Preceding&&... params ) {
        return pack_remainder::template call<Ret>(
            std::forward<F>(f), static_cast<pack_remainder&&>(pack),
            std::forward<Preceding>(params)... , std::move(pack.leftmost_value)
        );
    }
};

//! Calls the given function with arguments taken from a stored_pack
template< typename F, typename Pack >
void call( F&& f, Pack&& p ) {
    strip<Pack>::type::template call<void>( std::forward<F>(f), std::forward<Pack>(p) );
}

template< typename Ret, typename F, typename Pack >
Ret call_and_return( F&& f, Pack&& p ) {
    return strip<Pack>::type::template call<Ret>( std::forward<F>(f), std::forward<Pack>(p) );
}

template< typename... Types >
stored_pack<Types...> save_pack( Types&&... types ) {
    return stored_pack<Types...>( std::forward<Types>(types)... );
}

#endif /* __TBB_CPP11_RVALUE_REF_PRESENT && __TBB_CPP11_VARIADIC_TEMPLATES_PRESENT */
} } // namespace internal, namespace tbb

#endif /* __TBB_template_helpers_H */
