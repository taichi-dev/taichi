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

#ifndef __TBB_cache_aligned_allocator_H
#define __TBB_cache_aligned_allocator_H

#include <new>
#include "tbb_stddef.h"
#if __TBB_ALLOCATOR_CONSTRUCT_VARIADIC
 #include <utility> // std::forward
#endif

namespace tbb {

//! @cond INTERNAL
namespace internal {
    //! Cache/sector line size.
    /** @ingroup memory_allocation */
    size_t __TBB_EXPORTED_FUNC NFS_GetLineSize();

    //! Allocate memory on cache/sector line boundary.
    /** @ingroup memory_allocation */
    void* __TBB_EXPORTED_FUNC NFS_Allocate( size_t n_element, size_t element_size, void* hint );

    //! Free memory allocated by NFS_Allocate.
    /** Freeing a NULL pointer is allowed, but has no effect.
        @ingroup memory_allocation */
    void __TBB_EXPORTED_FUNC NFS_Free( void* );
}
//! @endcond

#if _MSC_VER && !defined(__INTEL_COMPILER)
    // Workaround for erroneous "unreferenced parameter" warning in method destroy.
    #pragma warning (push)
    #pragma warning (disable: 4100)
#endif

//! Meets "allocator" requirements of ISO C++ Standard, Section 20.1.5
/** The members are ordered the same way they are in section 20.4.1
    of the ISO C++ standard.
    @ingroup memory_allocation */
template<typename T>
class cache_aligned_allocator {
public:
    typedef typename internal::allocator_type<T>::value_type value_type;
    typedef value_type* pointer;
    typedef const value_type* const_pointer;
    typedef value_type& reference;
    typedef const value_type& const_reference;
    typedef size_t size_type;
    typedef ptrdiff_t difference_type;
    template<typename U> struct rebind {
        typedef cache_aligned_allocator<U> other;
    };

    cache_aligned_allocator() throw() {}
    cache_aligned_allocator( const cache_aligned_allocator& ) throw() {}
    template<typename U> cache_aligned_allocator(const cache_aligned_allocator<U>&) throw() {}

    pointer address(reference x) const {return &x;}
    const_pointer address(const_reference x) const {return &x;}
    
    //! Allocate space for n objects, starting on a cache/sector line.
    pointer allocate( size_type n, const void* hint=0 ) {
        // The "hint" argument is always ignored in NFS_Allocate thus const_cast shouldn't hurt
        return pointer(internal::NFS_Allocate( n, sizeof(value_type), const_cast<void*>(hint) ));
    }

    //! Free block of memory that starts on a cache line
    void deallocate( pointer p, size_type ) {
        internal::NFS_Free(p);
    }

    //! Largest value for which method allocate might succeed.
    size_type max_size() const throw() {
        return (~size_t(0)-internal::NFS_MaxLineSize)/sizeof(value_type);
    }

    //! Copy-construct value at location pointed to by p.
#if __TBB_ALLOCATOR_CONSTRUCT_VARIADIC
    template<typename U, typename... Args>
    void construct(U *p, Args&&... args)
        { ::new((void *)p) U(std::forward<Args>(args)...); }
#else // __TBB_ALLOCATOR_CONSTRUCT_VARIADIC
#if __TBB_CPP11_RVALUE_REF_PRESENT
    void construct( pointer p, value_type&& value ) {::new((void*)(p)) value_type(std::move(value));}
#endif
    void construct( pointer p, const value_type& value ) {::new((void*)(p)) value_type(value);}
#endif // __TBB_ALLOCATOR_CONSTRUCT_VARIADIC

    //! Destroy value at location pointed to by p.
    void destroy( pointer p ) {p->~value_type();}
};

#if _MSC_VER && !defined(__INTEL_COMPILER)
    #pragma warning (pop)
#endif // warning 4100 is back

//! Analogous to std::allocator<void>, as defined in ISO C++ Standard, Section 20.4.1
/** @ingroup memory_allocation */
template<> 
class cache_aligned_allocator<void> {
public:
    typedef void* pointer;
    typedef const void* const_pointer;
    typedef void value_type;
    template<typename U> struct rebind {
        typedef cache_aligned_allocator<U> other;
    };
};

template<typename T, typename U>
inline bool operator==( const cache_aligned_allocator<T>&, const cache_aligned_allocator<U>& ) {return true;}

template<typename T, typename U>
inline bool operator!=( const cache_aligned_allocator<T>&, const cache_aligned_allocator<U>& ) {return false;}

} // namespace tbb

#endif /* __TBB_cache_aligned_allocator_H */
