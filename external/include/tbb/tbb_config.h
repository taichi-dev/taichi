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

#ifndef __TBB_tbb_config_H
#define __TBB_tbb_config_H

/** This header is supposed to contain macro definitions and C style comments only.
    The macros defined here are intended to control such aspects of TBB build as
    - presence of compiler features
    - compilation modes
    - feature sets
    - known compiler/platform issues
**/

/* This macro marks incomplete code or comments describing ideas which are considered for the future.
 * See also for plain comment with TODO and FIXME marks for small improvement opportunities.
 */
#define __TBB_TODO 0

/*Check which standard library we use on OS X.*/
/*__TBB_SYMBOL is defined only while processing exported symbols list where C++ is not allowed.*/
#if !defined(__TBB_SYMBOL) && __APPLE__
    #include <cstddef>
#endif

// note that when ICC is in use __TBB_GCC_VERSION might not closely match GCC version on the machine
#define __TBB_GCC_VERSION (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)

#if __clang__
    /**according to clang documentation version can be vendor specific **/
    #define __TBB_CLANG_VERSION (__clang_major__ * 10000 + __clang_minor__ * 100 + __clang_patchlevel__)
#endif

/** Preprocessor symbols to determine HW architecture **/

#if _WIN32||_WIN64
#   if defined(_M_X64)||defined(__x86_64__)  // the latter for MinGW support
#       define __TBB_x86_64 1
#   elif defined(_M_IA64)
#       define __TBB_ipf 1
#   elif defined(_M_IX86)||defined(__i386__) // the latter for MinGW support
#       define __TBB_x86_32 1
#   else
#       define __TBB_generic_arch 1
#   endif
#else /* Assume generic Unix */
#   if !__linux__ && !__APPLE__
#       define __TBB_generic_os 1
#   endif
#   if __x86_64__
#       define __TBB_x86_64 1
#   elif __ia64__
#       define __TBB_ipf 1
#   elif __i386__||__i386  // __i386 is for Sun OS
#       define __TBB_x86_32 1
#   else
#       define __TBB_generic_arch 1
#   endif
#endif

#if __MIC__ || __MIC2__
#define __TBB_DEFINE_MIC 1
#endif

#define __TBB_TSX_AVAILABLE  (__TBB_x86_32 || __TBB_x86_64) && !__TBB_DEFINE_MIC

/** Presence of compiler features **/

#if __INTEL_COMPILER == 9999 && __INTEL_COMPILER_BUILD_DATE == 20110811
/* Intel(R) Composer XE 2011 Update 6 incorrectly sets __INTEL_COMPILER. Fix it. */
    #undef __INTEL_COMPILER
    #define __INTEL_COMPILER 1210
#endif

#if __TBB_GCC_VERSION >= 40400 && !defined(__INTEL_COMPILER)
    /** warning suppression pragmas available in GCC since 4.4 **/
    #define __TBB_GCC_WARNING_SUPPRESSION_PRESENT 1
#endif

/* Select particular features of C++11 based on compiler version.
   ICC 12.1 (Linux), GCC 4.3 and higher, clang 2.9 and higher
   set __GXX_EXPERIMENTAL_CXX0X__ in c++11 mode.

   Compilers that mimics other compilers (ICC, clang) must be processed before
   compilers they mimic (GCC, MSVC).

   TODO: The following conditions should be extended when new compilers/runtimes
   support added.
 */

#if __INTEL_COMPILER
    /** C++11 mode detection macros for Intel C++ compiler (enabled by -std=c++0x option):
          __INTEL_CXX11_MODE__ for version >=13.0
          __STDC_HOSTED__ for version >=12.0 on Windows,
          __GXX_EXPERIMENTAL_CXX0X__ for version >=12.0 on Linux and OS X. **/
    //  On Windows, C++11 features supported by Visual Studio 2010 and higher are enabled by default
    #ifndef __INTEL_CXX11_MODE__
        #define __INTEL_CXX11_MODE__ ((_MSC_VER && __STDC_HOSTED__) || __GXX_EXPERIMENTAL_CXX0X__)
        // TODO: check if more conditions can be simplified with the above macro
    #endif
    #define __TBB_CPP11_VARIADIC_TEMPLATES_PRESENT    (__INTEL_CXX11_MODE__ && __VARIADIC_TEMPLATES)
    // Both r-value reference support in compiler and std::move/std::forward
    // presence in C++ standard library is checked.
    #define __TBB_CPP11_RVALUE_REF_PRESENT            ((__GXX_EXPERIMENTAL_CXX0X__ && (__TBB_GCC_VERSION >= 40300 || _LIBCPP_VERSION) || _MSC_VER >= 1600) && __INTEL_COMPILER >= 1200)
    #if  _MSC_VER >= 1600
        #define __TBB_EXCEPTION_PTR_PRESENT           ( __INTEL_COMPILER > 1300                                                \
                                                      /*ICC 12.1 Upd 10 and 13 beta Upd 2 fixed exception_ptr linking  issue*/ \
                                                      || (__INTEL_COMPILER == 1300 && __INTEL_COMPILER_BUILD_DATE >= 20120530) \
                                                      || (__INTEL_COMPILER == 1210 && __INTEL_COMPILER_BUILD_DATE >= 20120410) )
    /** libstdc++ that comes with GCC 4.6 use C++11 features not supported by ICC 12.1.
     *  Because of that ICC 12.1 does not support C++11 mode with with gcc 4.6 (or higher),
     *  and therefore does not  define __GXX_EXPERIMENTAL_CXX0X__ macro **/
    #elif __TBB_GCC_VERSION >= 40404 && __TBB_GCC_VERSION < 40600
        #define __TBB_EXCEPTION_PTR_PRESENT           (__GXX_EXPERIMENTAL_CXX0X__ && __INTEL_COMPILER >= 1200)
    #elif __TBB_GCC_VERSION >= 40600
        #define __TBB_EXCEPTION_PTR_PRESENT           (__GXX_EXPERIMENTAL_CXX0X__ && __INTEL_COMPILER >= 1300)
    #elif _LIBCPP_VERSION
        #define __TBB_EXCEPTION_PTR_PRESENT           __GXX_EXPERIMENTAL_CXX0X__
    #else
        #define __TBB_EXCEPTION_PTR_PRESENT           0
    #endif
    #define __TBB_STATIC_ASSERT_PRESENT               (__INTEL_CXX11_MODE__ || _MSC_VER >= 1600)
    #define __TBB_CPP11_TUPLE_PRESENT                 (_MSC_VER >= 1600 || (__GXX_EXPERIMENTAL_CXX0X__ && __TBB_GCC_VERSION >= 40300))
    #if (__clang__ && __INTEL_COMPILER > 1400)
        /* Older versions of Intel Compiler do not have __has_include */
        #if (__has_feature(__cxx_generalized_initializers__) && __has_include(<initializer_list>))
            #define __TBB_INITIALIZER_LISTS_PRESENT   1
        #endif
    #else
        #define __TBB_INITIALIZER_LISTS_PRESENT       __INTEL_CXX11_MODE__ && __INTEL_COMPILER >= 1400 && (_MSC_VER >= 1800 || __TBB_GCC_VERSION >= 40400 || _LIBCPP_VERSION)
    #endif
    
    #define __TBB_CONSTEXPR_PRESENT                   __INTEL_CXX11_MODE__ && __INTEL_COMPILER >= 1400
    #define __TBB_DEFAULTED_AND_DELETED_FUNC_PRESENT  __INTEL_CXX11_MODE__ && __INTEL_COMPILER >= 1200
    /** ICC seems to disable support of noexcept event in c++11 when compiling in compatibility mode for gcc <4.6 **/
    #define __TBB_NOEXCEPT_PRESENT                    __INTEL_CXX11_MODE__ && __INTEL_COMPILER >= 1300 && (__TBB_GCC_VERSION >= 40600 || _LIBCPP_VERSION || _MSC_VER)
    #define __TBB_CPP11_STD_BEGIN_END_PRESENT         (_MSC_VER >= 1700 || __GXX_EXPERIMENTAL_CXX0X__ && __INTEL_COMPILER >= 1310 && (__TBB_GCC_VERSION >= 40600 || _LIBCPP_VERSION))
    #define __TBB_CPP11_AUTO_PRESENT                  (_MSC_VER >= 1600 || __GXX_EXPERIMENTAL_CXX0X__ && __INTEL_COMPILER >= 1210)
    #define __TBB_CPP11_DECLTYPE_PRESENT              (_MSC_VER >= 1600 || __GXX_EXPERIMENTAL_CXX0X__ && __INTEL_COMPILER >= 1210)
    #define __TBB_CPP11_LAMBDAS_PRESENT               (__INTEL_CXX11_MODE__ && __INTEL_COMPILER >= 1200)
#elif __clang__
//TODO: these options need to be rechecked
/** on OS X* the only way to get C++11 is to use clang. For library features (e.g. exception_ptr) libc++ is also
 *  required. So there is no need to check GCC version for clang**/
    #define __TBB_CPP11_VARIADIC_TEMPLATES_PRESENT    (__has_feature(__cxx_variadic_templates__))
    #define __TBB_CPP11_RVALUE_REF_PRESENT            (__has_feature(__cxx_rvalue_references__) && (__TBB_GCC_VERSION >= 40300 || _LIBCPP_VERSION))
/** TODO: extend exception_ptr related conditions to cover libstdc++ **/
    #define __TBB_EXCEPTION_PTR_PRESENT               (__cplusplus >= 201103L && _LIBCPP_VERSION)
    #define __TBB_STATIC_ASSERT_PRESENT               __has_feature(__cxx_static_assert__)
    /**Clang (preprocessor) has problems with dealing with expression having __has_include in #ifs
     * used inside C++ code. (At least version that comes with OS X 10.8 : Apple LLVM version 4.2 (clang-425.0.28) (based on LLVM 3.2svn)) **/
    #if (__GXX_EXPERIMENTAL_CXX0X__ && __has_include(<tuple>))
        #define __TBB_CPP11_TUPLE_PRESENT             1
    #endif
    #if (__has_feature(__cxx_generalized_initializers__) && __has_include(<initializer_list>))
        #define __TBB_INITIALIZER_LISTS_PRESENT       1
    #endif
    #define __TBB_CONSTEXPR_PRESENT                   __has_feature(__cxx_constexpr__)
    #define __TBB_DEFAULTED_AND_DELETED_FUNC_PRESENT  (__has_feature(__cxx_defaulted_functions__) && __has_feature(__cxx_deleted_functions__))
    /**For some unknown reason  __has_feature(__cxx_noexcept) does not yield true for all cases. Compiler bug ? **/
    #define __TBB_NOEXCEPT_PRESENT                    (__cplusplus >= 201103L)
    #define __TBB_CPP11_STD_BEGIN_END_PRESENT         (__has_feature(__cxx_range_for__) && _LIBCPP_VERSION)
    #define __TBB_CPP11_AUTO_PRESENT                  __has_feature(__cxx_auto_type__)
    #define __TBB_CPP11_DECLTYPE_PRESENT              __has_feature(__cxx_decltype__)
    #define __TBB_CPP11_LAMBDAS_PRESENT               __has_feature(cxx_lambdas)
#elif __GNUC__
    #define __TBB_CPP11_VARIADIC_TEMPLATES_PRESENT    __GXX_EXPERIMENTAL_CXX0X__
    #define __TBB_CPP11_RVALUE_REF_PRESENT            __GXX_EXPERIMENTAL_CXX0X__
    /** __GCC_HAVE_SYNC_COMPARE_AND_SWAP_4 here is a substitution for _GLIBCXX_ATOMIC_BUILTINS_4, which is a prerequisite 
        for exception_ptr but cannot be used in this file because it is defined in a header, not by the compiler. 
        If the compiler has no atomic intrinsics, the C++ library should not expect those as well. **/
    #define __TBB_EXCEPTION_PTR_PRESENT               (__GXX_EXPERIMENTAL_CXX0X__ && __TBB_GCC_VERSION >= 40404 && __GCC_HAVE_SYNC_COMPARE_AND_SWAP_4)
    #define __TBB_STATIC_ASSERT_PRESENT               (__GXX_EXPERIMENTAL_CXX0X__ && __TBB_GCC_VERSION >= 40300)
    #define __TBB_CPP11_TUPLE_PRESENT                 (__GXX_EXPERIMENTAL_CXX0X__ && __TBB_GCC_VERSION >= 40300)
    #define __TBB_INITIALIZER_LISTS_PRESENT           (__GXX_EXPERIMENTAL_CXX0X__ && __TBB_GCC_VERSION >= 40400)
    /** gcc seems have to support constexpr from 4.4 but tests in (test_atomic) seeming reasonable fail to compile prior 4.6**/
    #define __TBB_CONSTEXPR_PRESENT                   (__GXX_EXPERIMENTAL_CXX0X__ && __TBB_GCC_VERSION >= 40400)
    #define __TBB_DEFAULTED_AND_DELETED_FUNC_PRESENT  (__GXX_EXPERIMENTAL_CXX0X__ && __TBB_GCC_VERSION >= 40400)
    #define __TBB_NOEXCEPT_PRESENT                    (__GXX_EXPERIMENTAL_CXX0X__ && __TBB_GCC_VERSION >= 40600)
    #define __TBB_CPP11_STD_BEGIN_END_PRESENT         (__GXX_EXPERIMENTAL_CXX0X__ && __TBB_GCC_VERSION >= 40600)
    #define __TBB_CPP11_AUTO_PRESENT                  (__GXX_EXPERIMENTAL_CXX0X__ && __TBB_GCC_VERSION >= 40400)
    #define __TBB_CPP11_DECLTYPE_PRESENT              (__GXX_EXPERIMENTAL_CXX0X__ && __TBB_GCC_VERSION >= 40400)
    #define __TBB_CPP11_LAMBDAS_PRESENT               (__GXX_EXPERIMENTAL_CXX0X__ && __TBB_GCC_VERSION >= 40500)
#elif _MSC_VER
    #define __TBB_CPP11_VARIADIC_TEMPLATES_PRESENT    (_MSC_VER >= 1800)
    #define __TBB_CPP11_RVALUE_REF_PRESENT            (_MSC_VER >= 1600)
    #define __TBB_EXCEPTION_PTR_PRESENT               (_MSC_VER >= 1600)
    #define __TBB_STATIC_ASSERT_PRESENT               (_MSC_VER >= 1600)
    #define __TBB_CPP11_TUPLE_PRESENT                 (_MSC_VER >= 1600)
    #define __TBB_INITIALIZER_LISTS_PRESENT           (_MSC_VER >= 1800)
    #define __TBB_CONSTEXPR_PRESENT                   (_MSC_VER >= 1900)
    #define __TBB_DEFAULTED_AND_DELETED_FUNC_PRESENT  (_MSC_VER >= 1800)
    #define __TBB_NOEXCEPT_PRESENT                    (_MSC_VER >= 1900)
    #define __TBB_CPP11_STD_BEGIN_END_PRESENT         (_MSC_VER >= 1700)
    #define __TBB_CPP11_AUTO_PRESENT                  (_MSC_VER >= 1600)
    #define __TBB_CPP11_DECLTYPE_PRESENT              (_MSC_VER >= 1600)
    #define __TBB_CPP11_LAMBDAS_PRESENT               (_MSC_VER >= 1600)
#else
    #define __TBB_CPP11_VARIADIC_TEMPLATES_PRESENT    0
    #define __TBB_CPP11_RVALUE_REF_PRESENT            0
    #define __TBB_EXCEPTION_PTR_PRESENT               0
    #define __TBB_STATIC_ASSERT_PRESENT               0
    #define __TBB_CPP11_TUPLE_PRESENT                 0
    #define __TBB_INITIALIZER_LISTS_PRESENT           0
    #define __TBB_CONSTEXPR_PRESENT                   0
    #define __TBB_DEFAULTED_AND_DELETED_FUNC_PRESENT  0
    #define __TBB_NOEXCEPT_PRESENT                    0
    #define __TBB_CPP11_STD_BEGIN_END_PRESENT         0
    #define __TBB_CPP11_AUTO_PRESENT                  0
    #define __TBB_CPP11_DECLTYPE_PRESENT              0
    #define __TBB_CPP11_LAMBDAS_PRESENT               0
#endif

// C++11 standard library features

#define __TBB_CPP11_VARIADIC_TUPLE_PRESENT          (!_MSC_VER || _MSC_VER >=1800)
#define __TBB_CPP11_TYPE_PROPERTIES_PRESENT         (_LIBCPP_VERSION || _MSC_VER >= 1700 || (__TBB_GCC_VERSION >= 50000 && __GXX_EXPERIMENTAL_CXX0X__))
#define __TBB_TR1_TYPE_PROPERTIES_IN_STD_PRESENT    (__GXX_EXPERIMENTAL_CXX0X__ && __TBB_GCC_VERSION >= 40300 || _MSC_VER >= 1600)
// GCC supported some of type properties since 4.7
#define __TBB_CPP11_IS_COPY_CONSTRUCTIBLE_PRESENT   (__GXX_EXPERIMENTAL_CXX0X__ && __TBB_GCC_VERSION >= 40700 || __TBB_CPP11_TYPE_PROPERTIES_PRESENT)

// In GCC and MSVC, implementation of std::move_if_noexcept is not aligned with noexcept
#define __TBB_MOVE_IF_NOEXCEPT_PRESENT           (__GXX_EXPERIMENTAL_CXX0X__ && __TBB_GCC_VERSION >= 40700 || _MSC_VER >= 1900 || __clang__ && _LIBCPP_VERSION && __TBB_NOEXCEPT_PRESENT)
//TODO: Probably more accurate way is to analyze version of stdlibc++ via__GLIBCXX__ instead of __TBB_GCC_VERSION
#define __TBB_ALLOCATOR_TRAITS_PRESENT              (__cplusplus >= 201103L && _LIBCPP_VERSION  || _MSC_VER >= 1700 ||                                             \
                                                     __GXX_EXPERIMENTAL_CXX0X__ && __TBB_GCC_VERSION >= 40700 && !(__TBB_GCC_VERSION == 40700 && __TBB_DEFINE_MIC) \
                                                    )
#define __TBB_MAKE_EXCEPTION_PTR_PRESENT         (__TBB_EXCEPTION_PTR_PRESENT && (_MSC_VER >= 1700 || __TBB_GCC_VERSION >= 40600 || _LIBCPP_VERSION))

//TODO: not clear how exactly this macro affects exception_ptr - investigate
// On linux ICC fails to find existing std::exception_ptr in libstdc++ without this define
#if __INTEL_COMPILER && __GNUC__ && __TBB_EXCEPTION_PTR_PRESENT && !defined(__GCC_HAVE_SYNC_COMPARE_AND_SWAP_4)
    #define __GCC_HAVE_SYNC_COMPARE_AND_SWAP_4 1
#endif

// Work around a bug in MinGW32
#if __MINGW32__ && __TBB_EXCEPTION_PTR_PRESENT && !defined(_GLIBCXX_ATOMIC_BUILTINS_4)
    #define _GLIBCXX_ATOMIC_BUILTINS_4
#endif

#if __GNUC__ || __SUNPRO_CC || __IBMCPP__
    /* ICC defines __GNUC__ and so is covered */
    #define __TBB_ATTRIBUTE_ALIGNED_PRESENT 1
#elif _MSC_VER && (_MSC_VER >= 1300 || __INTEL_COMPILER)
    #define __TBB_DECLSPEC_ALIGN_PRESENT 1
#endif

/* Actually ICC supports gcc __sync_* intrinsics starting 11.1,
 * but 64 bit support for 32 bit target comes in later ones*/
/* TODO: change the version back to 4.1.2 once macro __TBB_WORD_SIZE become optional */
/* Assumed that all clang versions have these gcc compatible intrinsics. */
#if __TBB_GCC_VERSION >= 40306 || __INTEL_COMPILER >= 1200 || __clang__
    /** built-in atomics available in GCC since 4.1.2 **/
    #define __TBB_GCC_BUILTIN_ATOMICS_PRESENT 1
#endif

#if __INTEL_COMPILER >= 1200
    /** built-in C++11 style atomics available in ICC since 12.0 **/
    #define __TBB_ICC_BUILTIN_ATOMICS_PRESENT 1
#endif

#define __TBB_TSX_INTRINSICS_PRESENT ((__RTM__ || _MSC_VER>=1700 || __INTEL_COMPILER>=1300) && !__TBB_DEFINE_MIC && !__ANDROID__)

/** User controlled TBB features & modes **/

#ifndef TBB_USE_DEBUG
#ifdef _DEBUG
#define TBB_USE_DEBUG _DEBUG
#else
#define TBB_USE_DEBUG 0
#endif
#endif /* TBB_USE_DEBUG */

#ifndef TBB_USE_ASSERT
#define TBB_USE_ASSERT TBB_USE_DEBUG
#endif /* TBB_USE_ASSERT */

#ifndef TBB_USE_THREADING_TOOLS
#define TBB_USE_THREADING_TOOLS TBB_USE_DEBUG
#endif /* TBB_USE_THREADING_TOOLS */

#ifndef TBB_USE_PERFORMANCE_WARNINGS
#ifdef TBB_PERFORMANCE_WARNINGS
#define TBB_USE_PERFORMANCE_WARNINGS TBB_PERFORMANCE_WARNINGS
#else
#define TBB_USE_PERFORMANCE_WARNINGS TBB_USE_DEBUG
#endif /* TBB_PEFORMANCE_WARNINGS */
#endif /* TBB_USE_PERFORMANCE_WARNINGS */

#if !defined(__EXCEPTIONS) && !defined(_CPPUNWIND) && !defined(__SUNPRO_CC) || defined(_XBOX)
    #if TBB_USE_EXCEPTIONS
        #error Compilation settings do not support exception handling. Please do not set TBB_USE_EXCEPTIONS macro or set it to 0.
    #elif !defined(TBB_USE_EXCEPTIONS)
        #define TBB_USE_EXCEPTIONS 0
    #endif
#elif !defined(TBB_USE_EXCEPTIONS)
    #if __TBB_DEFINE_MIC
    #define TBB_USE_EXCEPTIONS 0
    #else
    #define TBB_USE_EXCEPTIONS 1
    #endif
#elif TBB_USE_EXCEPTIONS && __TBB_DEFINE_MIC
    #error Please do not set TBB_USE_EXCEPTIONS macro or set it to 0.
#endif

#ifndef TBB_IMPLEMENT_CPP0X
/** By default, use C++11 classes if available **/
    #if __clang__
        /* Old versions of Intel Compiler do not have __has_include */
        #if (__INTEL_COMPILER && __INTEL_COMPILER <= 1400) 
            #define TBB_IMPLEMENT_CPP0X !(_LIBCPP_VERSION && (__cplusplus >= 201103L))
        #else
            #define TBB_IMPLEMENT_CPP0X (__cplusplus < 201103L || (!__has_include(<thread>) && !__has_include(<condition_variable>)))
        #endif
    #elif __GNUC__
        #define TBB_IMPLEMENT_CPP0X (__TBB_GCC_VERSION < 40400 || !__GXX_EXPERIMENTAL_CXX0X__)
    #elif _MSC_VER
        #define TBB_IMPLEMENT_CPP0X (_MSC_VER < 1700)
    #else
        // TODO: Reconsider general approach to be more reliable, e.g. (!(__cplusplus >= 201103L && __ STDC_HOSTED__))
        #define TBB_IMPLEMENT_CPP0X (!__STDCPP_THREADS__)
    #endif
#endif /* TBB_IMPLEMENT_CPP0X */

/* TBB_USE_CAPTURED_EXCEPTION should be explicitly set to either 0 or 1, as it is used as C++ const */
#ifndef TBB_USE_CAPTURED_EXCEPTION
    /** IA-64 architecture pre-built TBB binaries do not support exception_ptr. **/
    #if __TBB_EXCEPTION_PTR_PRESENT && !defined(__ia64__)
        #define TBB_USE_CAPTURED_EXCEPTION 0
    #else
        #define TBB_USE_CAPTURED_EXCEPTION 1
    #endif
#else /* defined TBB_USE_CAPTURED_EXCEPTION */
    #if !TBB_USE_CAPTURED_EXCEPTION && !__TBB_EXCEPTION_PTR_PRESENT
        #error Current runtime does not support std::exception_ptr. Set TBB_USE_CAPTURED_EXCEPTION and make sure that your code is ready to catch tbb::captured_exception.
    #endif
#endif /* defined TBB_USE_CAPTURED_EXCEPTION */

/** Check whether the request to use GCC atomics can be satisfied **/
#if TBB_USE_GCC_BUILTINS && !__TBB_GCC_BUILTIN_ATOMICS_PRESENT
    #error "GCC atomic built-ins are not supported."
#endif

/** Internal TBB features & modes **/

/** __TBB_WEAK_SYMBOLS_PRESENT denotes that the system supports the weak symbol mechanism **/
#ifndef __TBB_WEAK_SYMBOLS_PRESENT
#define __TBB_WEAK_SYMBOLS_PRESENT ( !_WIN32 && !__APPLE__ && !__sun && (__TBB_GCC_VERSION >= 40000 || __INTEL_COMPILER ) )
#endif

/** __TBB_DYNAMIC_LOAD_ENABLED describes the system possibility to load shared libraries at run time **/
#ifndef __TBB_DYNAMIC_LOAD_ENABLED
    #define __TBB_DYNAMIC_LOAD_ENABLED 1
#endif

/** __TBB_SOURCE_DIRECTLY_INCLUDED is a mode used in whitebox testing when
    it's necessary to test internal functions not exported from TBB DLLs
**/
#if (_WIN32||_WIN64) && (__TBB_SOURCE_DIRECTLY_INCLUDED || TBB_USE_PREVIEW_BINARY)
    #define __TBB_NO_IMPLICIT_LINKAGE 1
    #define __TBBMALLOC_NO_IMPLICIT_LINKAGE 1
#endif

#ifndef __TBB_COUNT_TASK_NODES
    #define __TBB_COUNT_TASK_NODES TBB_USE_ASSERT
#endif

#ifndef __TBB_TASK_GROUP_CONTEXT
    #define __TBB_TASK_GROUP_CONTEXT 1
#endif /* __TBB_TASK_GROUP_CONTEXT */

#ifndef __TBB_SCHEDULER_OBSERVER
    #define __TBB_SCHEDULER_OBSERVER 1
#endif /* __TBB_SCHEDULER_OBSERVER */

#ifndef __TBB_FP_CONTEXT
    #define __TBB_FP_CONTEXT __TBB_TASK_GROUP_CONTEXT
#endif /* __TBB_FP_CONTEXT */

#if __TBB_FP_CONTEXT && !__TBB_TASK_GROUP_CONTEXT
    #error __TBB_FP_CONTEXT requires __TBB_TASK_GROUP_CONTEXT to be enabled
#endif

#ifndef __TBB_TASK_ARENA
    #define __TBB_TASK_ARENA 1
#endif /* __TBB_TASK_ARENA  */
#if __TBB_TASK_ARENA
    #define __TBB_RECYCLE_TO_ENQUEUE __TBB_BUILD // keep non-official
    #if !__TBB_SCHEDULER_OBSERVER
        #error __TBB_TASK_ARENA requires __TBB_SCHEDULER_OBSERVER to be enabled
    #endif
#endif /* __TBB_TASK_ARENA */

#ifndef __TBB_ARENA_OBSERVER
    #define __TBB_ARENA_OBSERVER ((__TBB_BUILD||TBB_PREVIEW_LOCAL_OBSERVER)&& __TBB_SCHEDULER_OBSERVER)
#endif /* __TBB_ARENA_OBSERVER */

#ifndef __TBB_SLEEP_PERMISSION
    #define __TBB_SLEEP_PERMISSION ((__TBB_CPF_BUILD||TBB_PREVIEW_LOCAL_OBSERVER)&& __TBB_SCHEDULER_OBSERVER)
#endif /* __TBB_SLEEP_PERMISSION */

#if TBB_PREVIEW_FLOW_GRAPH_TRACE
// Users of flow-graph trace need to explicitly link against the preview library.  This
// prevents the linker from implicitly linking an application with a preview version of
// TBB and unexpectedly bringing in other community preview features, which might change
// the behavior of the application.
#define __TBB_NO_IMPLICIT_LINKAGE 1
#endif /* TBB_PREVIEW_FLOW_GRAPH_TRACE */

#ifndef __TBB_ITT_STRUCTURE_API
#define __TBB_ITT_STRUCTURE_API ( !__TBB_DEFINE_MIC && (__TBB_CPF_BUILD || TBB_PREVIEW_FLOW_GRAPH_TRACE) )
#endif

#if TBB_USE_EXCEPTIONS && !__TBB_TASK_GROUP_CONTEXT
    #error TBB_USE_EXCEPTIONS requires __TBB_TASK_GROUP_CONTEXT to be enabled
#endif

#ifndef __TBB_TASK_PRIORITY
    #define __TBB_TASK_PRIORITY (__TBB_TASK_GROUP_CONTEXT)
#endif /* __TBB_TASK_PRIORITY */

#if __TBB_TASK_PRIORITY && !__TBB_TASK_GROUP_CONTEXT
    #error __TBB_TASK_PRIORITY requires __TBB_TASK_GROUP_CONTEXT to be enabled
#endif

#if TBB_PREVIEW_WAITING_FOR_WORKERS || __TBB_BUILD
    #define __TBB_SUPPORTS_WORKERS_WAITING_IN_TERMINATE 1
#endif

#if !defined(__TBB_SURVIVE_THREAD_SWITCH) && \
          (_WIN32 || _WIN64 || __APPLE__ || (__linux__ && !__ANDROID__))
    #define __TBB_SURVIVE_THREAD_SWITCH 1
#endif /* __TBB_SURVIVE_THREAD_SWITCH */

#ifndef __TBB_DEFAULT_PARTITIONER
#if TBB_DEPRECATED
/** Default partitioner for parallel loop templates in TBB 1.0-2.1 */
#define __TBB_DEFAULT_PARTITIONER tbb::simple_partitioner
#else
/** Default partitioner for parallel loop templates since TBB 2.2 */
#define __TBB_DEFAULT_PARTITIONER tbb::auto_partitioner
#endif /* TBB_DEPRECATED */
#endif /* !defined(__TBB_DEFAULT_PARTITIONER */

#ifndef __TBB_USE_PROPORTIONAL_SPLIT_IN_BLOCKED_RANGES
#define __TBB_USE_PROPORTIONAL_SPLIT_IN_BLOCKED_RANGES 1
#endif

#ifndef __TBB_ENABLE_RANGE_FEEDBACK
#define __TBB_ENABLE_RANGE_FEEDBACK 0
#endif

#ifdef _VARIADIC_MAX
    #define __TBB_VARIADIC_MAX _VARIADIC_MAX
#else
    #if _MSC_VER == 1700
        #define __TBB_VARIADIC_MAX 5 // VS11 setting, issue resolved in VS12
    #elif _MSC_VER == 1600
        #define __TBB_VARIADIC_MAX 10 // VS10 setting
    #else
        #define __TBB_VARIADIC_MAX 15 
    #endif
#endif

/** __TBB_WIN8UI_SUPPORT enables support of New Windows*8 Store Apps and limit a possibility to load
    shared libraries at run time only from application container **/
#if defined(WINAPI_FAMILY) && WINAPI_FAMILY == WINAPI_FAMILY_APP
    #define __TBB_WIN8UI_SUPPORT 1
#else
    #define __TBB_WIN8UI_SUPPORT 0
#endif

/** Macros of the form __TBB_XXX_BROKEN denote known issues that are caused by
    the bugs in compilers, standard or OS specific libraries. They should be
    removed as soon as the corresponding bugs are fixed or the buggy OS/compiler
    versions go out of the support list.
**/

#if __ANDROID__ && __TBB_GCC_VERSION <= 40403 && !__GCC_HAVE_SYNC_COMPARE_AND_SWAP_8
    /** Necessary because on Android 8-byte CAS and F&A are not available for some processor architectures,
        but no mandatory warning message appears from GCC 4.4.3. Instead, only a linkage error occurs when
        these atomic operations are used (such as in unit test test_atomic.exe). **/
    #define __TBB_GCC_64BIT_ATOMIC_BUILTINS_BROKEN 1
#elif __TBB_x86_32 && __TBB_GCC_VERSION == 40102 && ! __GNUC_RH_RELEASE__
    /** GCC 4.1.2 erroneously emit call to external function for 64 bit sync_ intrinsics.
        However these functions are not defined anywhere. It seems that this problem was fixed later on
        and RHEL got an updated version of gcc 4.1.2. **/
    #define __TBB_GCC_64BIT_ATOMIC_BUILTINS_BROKEN 1
#endif

#if __GNUC__ && __TBB_x86_64 && __INTEL_COMPILER == 1200
    #define __TBB_ICC_12_0_INL_ASM_FSTCW_BROKEN 1
#endif

#if _MSC_VER && __INTEL_COMPILER && (__INTEL_COMPILER<1110 || __INTEL_COMPILER==1110 && __INTEL_COMPILER_BUILD_DATE < 20091012)
    /** Necessary to avoid ICL error (or warning in non-strict mode):
        "exception specification for implicitly declared virtual destructor is
        incompatible with that of overridden one". **/
    #define __TBB_DEFAULT_DTOR_THROW_SPEC_BROKEN 1
#endif

#if !__INTEL_COMPILER && (_MSC_VER && _MSC_VER < 1500 || __TBB_GCC_VERSION && __TBB_GCC_VERSION < 40102)
    /** gcc 3.4.6 (and earlier) and VS2005 (and earlier) do not allow declaring template class as a friend
        of classes defined in other namespaces. **/
    #define __TBB_TEMPLATE_FRIENDS_BROKEN 1
#endif

//TODO: recheck for different clang versions 
#if __GLIBC__==2 && __GLIBC_MINOR__==3 ||  (__APPLE__ && ( __INTEL_COMPILER==1200 && !TBB_USE_DEBUG))
    /** Macro controlling EH usages in TBB tests.
        Some older versions of glibc crash when exception handling happens concurrently. **/
    #define __TBB_THROW_ACROSS_MODULE_BOUNDARY_BROKEN 1
#else
    #define __TBB_THROW_ACROSS_MODULE_BOUNDARY_BROKEN 0
#endif

#if (_WIN32||_WIN64) && __INTEL_COMPILER == 1110
    /** That's a bug in Intel compiler 11.1.044/IA-32/Windows, that leads to a worker thread crash on the thread's startup. **/
    #define __TBB_ICL_11_1_CODE_GEN_BROKEN 1
#endif

#if __clang__ || (__GNUC__==3 && __GNUC_MINOR__==3 && !defined(__INTEL_COMPILER))
    /** Bugs with access to nested classes declared in protected area */
    #define __TBB_PROTECTED_NESTED_CLASS_BROKEN 1
#endif

#if __MINGW32__ && __TBB_GCC_VERSION < 40200
    /** MinGW has a bug with stack alignment for routines invoked from MS RTLs.
        Since GCC 4.2, the bug can be worked around via a special attribute. **/
    #define __TBB_SSE_STACK_ALIGNMENT_BROKEN 1
#else
    #define __TBB_SSE_STACK_ALIGNMENT_BROKEN 0
#endif

#if __TBB_GCC_VERSION==40300 && !__INTEL_COMPILER && !__clang__
    /* GCC of this version may rashly ignore control dependencies */
    #define __TBB_GCC_OPTIMIZER_ORDERING_BROKEN 1
#endif

#if __FreeBSD__
    /** A bug in FreeBSD 8.0 results in kernel panic when there is contention
        on a mutex created with this attribute. **/
    #define __TBB_PRIO_INHERIT_BROKEN 1

    /** A bug in FreeBSD 8.0 results in test hanging when an exception occurs
        during (concurrent?) object construction by means of placement new operator. **/
    #define __TBB_PLACEMENT_NEW_EXCEPTION_SAFETY_BROKEN 1
#endif /* __FreeBSD__ */

#if (__linux__ || __APPLE__) && __i386__ && defined(__INTEL_COMPILER)
    /** The Intel compiler for IA-32 (Linux|OS X) crashes or generates
        incorrect code when __asm__ arguments have a cast to volatile. **/
    #define __TBB_ICC_ASM_VOLATILE_BROKEN 1
#endif

#if !__INTEL_COMPILER && (_MSC_VER || __GNUC__==3 && __GNUC_MINOR__<=2)
    /** Bug in GCC 3.2 and MSVC compilers that sometimes return 0 for __alignof(T)
        when T has not yet been instantiated. **/
    #define __TBB_ALIGNOF_NOT_INSTANTIATED_TYPES_BROKEN 1
#endif

#if __TBB_DEFINE_MIC
    /** Main thread and user's thread have different default thread affinity masks. **/
    #define __TBB_MAIN_THREAD_AFFINITY_BROKEN 1
#endif

#if __GXX_EXPERIMENTAL_CXX0X__ && !defined(__EXCEPTIONS) && \
    ((!__INTEL_COMPILER && !__clang__ && (__TBB_GCC_VERSION>=40400 && __TBB_GCC_VERSION<40600)) || \
     (__INTEL_COMPILER<=1400 && (__TBB_GCC_VERSION>=40400 && __TBB_GCC_VERSION<=40801)))
/* There is an issue for specific GCC toolchain when C++11 is enabled
   and exceptions are disabled:
   exceprion_ptr.h/nested_exception.h use throw unconditionally.
   GCC can ignore 'throw' since 4.6; but with ICC the issue still exists.
 */
    #define __TBB_LIBSTDCPP_EXCEPTION_HEADERS_BROKEN 1
#else
    #define __TBB_LIBSTDCPP_EXCEPTION_HEADERS_BROKEN 0
#endif

#if __INTEL_COMPILER==1300 && __TBB_GCC_VERSION>=40700 && defined(__GXX_EXPERIMENTAL_CXX0X__)
/* Some C++11 features used inside libstdc++ are not supported by Intel compiler.
 * Checking version of gcc instead of libstdc++ because
 *  - they are directly connected,
 *  - for now it is not possible to check version of any standard library in this file
 */
    #define __TBB_ICC_13_0_CPP11_STDLIB_SUPPORT_BROKEN 1
#else
    #define __TBB_ICC_13_0_CPP11_STDLIB_SUPPORT_BROKEN 0
#endif

#if (__GNUC__==4 && __GNUC_MINOR__==4 ) && !defined(__INTEL_COMPILER) && !defined(__clang__)
    /** excessive warnings related to strict aliasing rules in GCC 4.4 **/
    #define __TBB_GCC_STRICT_ALIASING_BROKEN 1
    /* topical remedy: #pragma GCC diagnostic ignored "-Wstrict-aliasing" */
    #if !__TBB_GCC_WARNING_SUPPRESSION_PRESENT
        #error Warning suppression is not supported, while should.
    #endif
#endif

/*In a PIC mode some versions of GCC 4.1.2 generate incorrect inlined code for 8 byte __sync_val_compare_and_swap intrinsic */
#if __TBB_GCC_VERSION == 40102 && __PIC__ && !defined(__INTEL_COMPILER) && !defined(__clang__)
    #define __TBB_GCC_CAS8_BUILTIN_INLINING_BROKEN 1
#endif

#if __TBB_x86_32 && (__linux__ || __APPLE__ || _WIN32 || __sun || __ANDROID__) &&  (__INTEL_COMPILER || (__GNUC__==3 && __GNUC_MINOR__==3 )||(__MINGW32__ ) && (__GNUC__==4 && __GNUC_MINOR__==5 ) || __SUNPRO_CC)
    // Some compilers for IA-32 fail to provide 8-byte alignment of objects on the stack,
    // even if the object specifies 8-byte alignment.  On such platforms, the IA-32 implementation
    // of 64 bit atomics (e.g. atomic<long long>) use different tactics depending upon
    // whether the object is properly aligned or not.
    #define __TBB_FORCE_64BIT_ALIGNMENT_BROKEN 1
#else
    #define __TBB_FORCE_64BIT_ALIGNMENT_BROKEN 0
#endif

#if __GNUC__ && !__INTEL_COMPILER && !__clang__ && __TBB_DEFAULTED_AND_DELETED_FUNC_PRESENT && __TBB_GCC_VERSION < 40700
    #define __TBB_ZERO_INIT_WITH_DEFAULTED_CTOR_BROKEN 1
#endif

#if _MSC_VER && _MSC_VER <= 1800 && !__INTEL_COMPILER
    // With MSVC, when an array is passed by const reference to a template function,
    // constness from the function parameter may get propagated to the template parameter.
    #define __TBB_CONST_REF_TO_ARRAY_TEMPLATE_PARAM_BROKEN 1
#endif

// A compiler bug: a disabled copy constructor prevents use of the moving constructor
#define __TBB_IF_NO_COPY_CTOR_MOVE_SEMANTICS_BROKEN (_MSC_VER && (__INTEL_COMPILER >= 1300 && __INTEL_COMPILER <= 1310) && !__INTEL_CXX11_MODE__)

// MSVC 2013 and ICC 15 seems do not generate implicit move constructor for empty derived class while should
#define __TBB_CPP11_IMPLICIT_MOVE_MEMBERS_GENERATION_FOR_DERIVED_BROKEN  (__TBB_CPP11_RVALUE_REF_PRESENT &&  \
      ( !__INTEL_COMPILER && _MSC_VER && _MSC_VER <= 1800 || __INTEL_COMPILER && __INTEL_COMPILER <= 1500 ))

#define __TBB_CPP11_DECLVAL_BROKEN (_MSC_VER == 1600 || (__GNUC__ && __TBB_GCC_VERSION < 40500) )

// Intel C++ compiler has difficulties with copying std::pair with VC11 std::reference_wrapper being a const member
#define __TBB_COPY_FROM_NON_CONST_REF_BROKEN (_MSC_VER == 1700 && __INTEL_COMPILER && __INTEL_COMPILER < 1600)
//The implicit upcasting of the tuple of a reference of a derived class to a base class fails on icc 13.X 
//if the system's gcc environment is 4.8
#if (__INTEL_COMPILER >=1300 && __INTEL_COMPILER <=1310) && __TBB_GCC_VERSION>=40700 && __GXX_EXPERIMENTAL_CXX0X__
    #define __TBB_UPCAST_OF_TUPLE_OF_REF_BROKEN 1 
#endif

/** End of __TBB_XXX_BROKEN macro section **/

#if defined(_MSC_VER) && _MSC_VER>=1500 && !defined(__INTEL_COMPILER)
    // A macro to suppress erroneous or benign "unreachable code" MSVC warning (4702)
    #define __TBB_MSVC_UNREACHABLE_CODE_IGNORED 1
#endif

#define __TBB_ATOMIC_CTORS     (__TBB_CONSTEXPR_PRESENT && __TBB_DEFAULTED_AND_DELETED_FUNC_PRESENT && (!__TBB_ZERO_INIT_WITH_DEFAULTED_CTOR_BROKEN))

#define __TBB_ALLOCATOR_CONSTRUCT_VARIADIC      (__TBB_CPP11_VARIADIC_TEMPLATES_PRESENT && __TBB_CPP11_RVALUE_REF_PRESENT)

#define __TBB_VARIADIC_PARALLEL_INVOKE          (TBB_PREVIEW_VARIADIC_PARALLEL_INVOKE && __TBB_CPP11_VARIADIC_TEMPLATES_PRESENT && __TBB_CPP11_RVALUE_REF_PRESENT)
#define __TBB_FLOW_GRAPH_CPP11_FEATURES         (__TBB_CPP11_VARIADIC_TEMPLATES_PRESENT \
                                                && __TBB_CPP11_RVALUE_REF_PRESENT && __TBB_CPP11_AUTO_PRESENT) \
                                                && __TBB_CPP11_VARIADIC_TUPLE_PRESENT && !__TBB_UPCAST_OF_TUPLE_OF_REF_BROKEN
#define __TBB_PREVIEW_ASYNC_NODE (__TBB_FLOW_GRAPH_CPP11_FEATURES && TBB_PREVIEW_FLOW_GRAPH_NODES)
#define __TBB_PREVIEW_OPENCL_NODE               (__TBB_FLOW_GRAPH_CPP11_FEATURES && TBB_PREVIEW_FLOW_GRAPH_NODES && !TBB_IMPLEMENT_CPP0X)
#define __TBB_PREVIEW_MESSAGE_BASED_KEY_MATCHING (TBB_PREVIEW_FLOW_GRAPH_FEATURES || __TBB_PREVIEW_OPENCL_NODE)
#endif /* __TBB_tbb_config_H */
