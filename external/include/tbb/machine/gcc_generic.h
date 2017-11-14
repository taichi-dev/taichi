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

#if !defined(__TBB_machine_H) || defined(__TBB_machine_gcc_generic_H)
#error Do not #include this internal file directly; use public TBB headers instead.
#endif

#define __TBB_machine_gcc_generic_H

#include <stdint.h>
#include <unistd.h>

#define __TBB_WORDSIZE      __SIZEOF_POINTER__

#if __TBB_GCC_64BIT_ATOMIC_BUILTINS_BROKEN
    #define __TBB_64BIT_ATOMICS 0
#endif

/** FPU control setting not available for non-Intel architectures on Android **/
#if __ANDROID__ && __TBB_generic_arch 
    #define __TBB_CPU_CTL_ENV_PRESENT 0
#endif

// __BYTE_ORDER__ is used in accordance with http://gcc.gnu.org/onlinedocs/cpp/Common-Predefined-Macros.html,
// but __BIG_ENDIAN__ or __LITTLE_ENDIAN__ may be more commonly found instead.
#if __BIG_ENDIAN__ || (defined(__BYTE_ORDER__) && __BYTE_ORDER__==__ORDER_BIG_ENDIAN__)
    #define __TBB_ENDIANNESS __TBB_ENDIAN_BIG
#elif __LITTLE_ENDIAN__ || (defined(__BYTE_ORDER__) && __BYTE_ORDER__==__ORDER_LITTLE_ENDIAN__)
    #define __TBB_ENDIANNESS __TBB_ENDIAN_LITTLE
#elif defined(__BYTE_ORDER__)
    #define __TBB_ENDIANNESS __TBB_ENDIAN_UNSUPPORTED
#else
    #define __TBB_ENDIANNESS __TBB_ENDIAN_DETECT
#endif

#if __TBB_GCC_VERSION < 40700
// Use __sync_* builtins

/** As this generic implementation has absolutely no information about underlying
    hardware, its performance most likely will be sub-optimal because of full memory
    fence usages where a more lightweight synchronization means (or none at all)
    could suffice. Thus if you use this header to enable TBB on a new platform,
    consider forking it and relaxing below helpers as appropriate. **/
#define __TBB_acquire_consistency_helper()  __atomic_thread_fence(__ATOMIC_SEQ_CST)
#define __TBB_release_consistency_helper()  __atomic_thread_fence(__ATOMIC_SEQ_CST)
#define __TBB_full_memory_fence()           __atomic_thread_fence(__ATOMIC_SEQ_CST)
#define __TBB_control_consistency_helper()  __atomic_thread_fence(__ATOMIC_SEQ_CST)

#define __TBB_MACHINE_DEFINE_ATOMICS(S,T)                                                         \
inline T __TBB_machine_cmpswp##S( volatile void *ptr, T value, T comparand ) {                    \
    T retval = comparand;                                                                         \
    if (__atomic_compare_exchange_n(reinterpret_cast<volatile T *>(ptr), &retval, value,          \
                                        false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST))               \
        return retval;                                                                            \
   return retval;                                                                                 \
}                                                                                                 \
inline T __TBB_machine_fetchadd##S( volatile void *ptr, T value ) {                               \
    return __sync_fetch_and_add(reinterpret_cast<volatile T *>(ptr),value);                       \
}

#define __TBB_USE_GENERIC_FETCH_STORE 1

#else
// __TBB_GCC_VERSION >= 40700; use __atomic_* builtins available since gcc 4.7

#define __TBB_compiler_fence()              __asm__ __volatile__("": : :"memory")
// Acquire and release fence intrinsics in GCC might miss compiler fence.
// Adding it at both sides of an intrinsic, as we do not know what reordering can be made.
#define __TBB_acquire_consistency_helper()  __TBB_compiler_fence(); __atomic_thread_fence(__ATOMIC_ACQUIRE); __TBB_compiler_fence()
#define __TBB_release_consistency_helper()  __TBB_compiler_fence(); __atomic_thread_fence(__ATOMIC_RELEASE); __TBB_compiler_fence()
#define __TBB_full_memory_fence()           __atomic_thread_fence(__ATOMIC_SEQ_CST)
#define __TBB_control_consistency_helper()  __TBB_acquire_consistency_helper()

#define __TBB_MACHINE_DEFINE_ATOMICS(S,T)                                                         \
inline T __TBB_machine_cmpswp##S( volatile void *ptr, T value, T comparand ) {                    \
    (void)__atomic_compare_exchange_n(reinterpret_cast<volatile T *>(ptr), &comparand, value,     \
                                      false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);                 \
    return comparand;                                                                             \
}                                                                                                 \
inline T __TBB_machine_fetchadd##S( volatile void *ptr, T value ) {                               \
    return __atomic_fetch_add(reinterpret_cast<volatile T *>(ptr), value, __ATOMIC_SEQ_CST);      \
}                                                                                                 \
inline T __TBB_machine_fetchstore##S( volatile void *ptr, T value ) {                             \
    return __atomic_exchange_n(reinterpret_cast<volatile T *>(ptr), value, __ATOMIC_SEQ_CST);     \
}

#endif // __TBB_GCC_VERSION < 40700

__TBB_MACHINE_DEFINE_ATOMICS(1,int8_t)
__TBB_MACHINE_DEFINE_ATOMICS(2,int16_t)
__TBB_MACHINE_DEFINE_ATOMICS(4,int32_t)
__TBB_MACHINE_DEFINE_ATOMICS(8,int64_t)

#undef __TBB_MACHINE_DEFINE_ATOMICS

namespace tbb{ namespace internal { namespace gcc_builtins {
    inline int clz(unsigned int x){ return __builtin_clz(x);};
    inline int clz(unsigned long int x){ return __builtin_clzl(x);};
    inline int clz(unsigned long long int x){ return __builtin_clzll(x);};
}}}
//gcc __builtin_clz builtin count _number_ of leading zeroes
static inline intptr_t __TBB_machine_lg( uintptr_t x ) {
    return sizeof(x)*8 - tbb::internal::gcc_builtins::clz(x) -1 ;
}


typedef unsigned char __TBB_Flag;
typedef __TBB_atomic __TBB_Flag __TBB_atomic_flag;

#if __TBB_GCC_VERSION < 40700
// Use __sync_* builtins

static inline void __TBB_machine_or( volatile void *ptr, uintptr_t addend ) {
    __sync_fetch_and_or(reinterpret_cast<volatile uintptr_t *>(ptr),addend);
}

static inline void __TBB_machine_and( volatile void *ptr, uintptr_t addend ) {
    __sync_fetch_and_and(reinterpret_cast<volatile uintptr_t *>(ptr),addend);
}

inline bool __TBB_machine_try_lock_byte( __TBB_atomic_flag &flag ) {
    return __sync_lock_test_and_set(&flag,1)==0;
}

inline void __TBB_machine_unlock_byte( __TBB_atomic_flag &flag ) {
    __sync_lock_release(&flag);
}

#else
// __TBB_GCC_VERSION >= 40700; use __atomic_* builtins available since gcc 4.7

static inline void __TBB_machine_or( volatile void *ptr, uintptr_t addend ) {
    __atomic_fetch_or(reinterpret_cast<volatile uintptr_t *>(ptr),addend,__ATOMIC_SEQ_CST);
}

static inline void __TBB_machine_and( volatile void *ptr, uintptr_t addend ) {
    __atomic_fetch_and(reinterpret_cast<volatile uintptr_t *>(ptr),addend,__ATOMIC_SEQ_CST);
}

inline bool __TBB_machine_try_lock_byte( __TBB_atomic_flag &flag ) {
    return !__atomic_test_and_set(&flag,__ATOMIC_ACQUIRE);
}

inline void __TBB_machine_unlock_byte( __TBB_atomic_flag &flag ) {
    __atomic_clear(&flag,__ATOMIC_RELEASE);
}

#endif // __TBB_GCC_VERSION < 40700

// Machine specific atomic operations
#define __TBB_AtomicOR(P,V)     __TBB_machine_or(P,V)
#define __TBB_AtomicAND(P,V)    __TBB_machine_and(P,V)

#define __TBB_TryLockByte   __TBB_machine_try_lock_byte
#define __TBB_UnlockByte    __TBB_machine_unlock_byte

// Definition of other functions
#define __TBB_Log2(V)           __TBB_machine_lg(V)

// TODO: implement with __atomic_* builtins where available
#define __TBB_USE_GENERIC_HALF_FENCED_LOAD_STORE            1
#define __TBB_USE_GENERIC_RELAXED_LOAD_STORE                1
#define __TBB_USE_GENERIC_SEQUENTIAL_CONSISTENCY_LOAD_STORE 1

#if __TBB_WORDSIZE==4
    #define __TBB_USE_GENERIC_DWORD_LOAD_STORE              1
#endif

#if __TBB_x86_32 || __TBB_x86_64
#include "gcc_itsx.h"
#endif
