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

#ifndef __TBB_machine_windows_api_H
#define __TBB_machine_windows_api_H

#if _WIN32 || _WIN64

#if _XBOX

#define NONET
#define NOD3D
#include <xtl.h>

#else // Assume "usual" Windows

#include <windows.h>

#endif // _XBOX

#if _WIN32_WINNT < 0x0600
// The following Windows API function is declared explicitly;
// otherwise it fails to compile by VS2005.
#if !defined(WINBASEAPI) || (_WIN32_WINNT < 0x0501 && _MSC_VER == 1400)
#define __TBB_WINBASEAPI extern "C"
#else
#define __TBB_WINBASEAPI WINBASEAPI
#endif
__TBB_WINBASEAPI BOOL WINAPI TryEnterCriticalSection( LPCRITICAL_SECTION );
__TBB_WINBASEAPI BOOL WINAPI InitializeCriticalSectionAndSpinCount( LPCRITICAL_SECTION, DWORD );
// Overloading WINBASEAPI macro and using local functions missing in Windows XP/2003
#define InitializeCriticalSectionEx inlineInitializeCriticalSectionEx
#define CreateSemaphoreEx inlineCreateSemaphoreEx
#define CreateEventEx inlineCreateEventEx
inline BOOL WINAPI inlineInitializeCriticalSectionEx( LPCRITICAL_SECTION lpCriticalSection, DWORD dwSpinCount, DWORD )
{
    return InitializeCriticalSectionAndSpinCount( lpCriticalSection, dwSpinCount );
}
inline HANDLE WINAPI inlineCreateSemaphoreEx( LPSECURITY_ATTRIBUTES lpSemaphoreAttributes, LONG lInitialCount, LONG lMaximumCount, LPCTSTR lpName, DWORD, DWORD )
{
    return CreateSemaphore( lpSemaphoreAttributes, lInitialCount, lMaximumCount, lpName );
}
inline HANDLE WINAPI inlineCreateEventEx( LPSECURITY_ATTRIBUTES lpEventAttributes, LPCTSTR lpName, DWORD dwFlags, DWORD )
{
    BOOL manual_reset = dwFlags&0x00000001 ? TRUE : FALSE; // CREATE_EVENT_MANUAL_RESET
    BOOL initial_set  = dwFlags&0x00000002 ? TRUE : FALSE; // CREATE_EVENT_INITIAL_SET
    return CreateEvent( lpEventAttributes, manual_reset, initial_set, lpName );
}
#endif

#if defined(RTL_SRWLOCK_INIT)
#ifndef __TBB_USE_SRWLOCK
// TODO: turn it on when bug 1952 will be fixed
#define __TBB_USE_SRWLOCK 0
#endif
#endif

#else
#error tbb/machine/windows_api.h should only be used for Windows based platforms
#endif // _WIN32 || _WIN64

#endif // __TBB_machine_windows_api_H
