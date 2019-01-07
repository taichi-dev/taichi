/*
PARTIO SOFTWARE
Copyright 2010 Disney Enterprises, Inc. All rights reserved

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in
the documentation and/or other materials provided with the
distribution.

* The names "Disney", "Walt Disney Pictures", "Walt Disney Animation
Studios" or the names of its contributors may NOT be used to
endorse or promote products derived from this software without
specific prior written permission from Walt Disney Pictures.

Disclaimer: THIS SOFTWARE IS PROVIDED BY WALT DISNEY PICTURES AND
CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING,
BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE, NONINFRINGEMENT AND TITLE ARE DISCLAIMED.
IN NO EVENT SHALL WALT DISNEY PICTURES, THE COPYRIGHT HOLDER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND BASED ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.
*/

#ifndef _Mutex_
#define _Mutex_

#ifndef PARTIO_WIN32

#include <pthread.h>

namespace Partio
{

#ifndef PARTIO_USE_SPINLOCK

    class PartioMutex
    {
        pthread_mutex_t CacheLock;
    
    public:
        inline PartioMutex()
        {
            pthread_mutex_init(&CacheLock,0);
        }
    
        inline ~PartioMutex()
        {
            pthread_mutex_destroy(&CacheLock);
        }
    
        inline void lock()
        {
            pthread_mutex_lock(&CacheLock);
        }
    
        inline void unlock()
        {
            pthread_mutex_unlock(&CacheLock);
        }
    };
    
#else

    class PartioMutex
    {
        pthread_spinlock_t CacheLock;
    
    public:
        inline PartioMutex()
        {
            pthread_spinlock_init(&CacheLock,PTHREAD_PROCESS_PRIVATE);
        }
    
        inline ~PartioMutex()
        {
            pthread_spinlock_destroy(&CacheLock);
        }
    
        inline void lock()
        {
            pthread_spinlock_lock(&CacheLock);
        }
    
        inline void unlock()
        {
            pthread_spinlock_unlock(&CacheLock);
        }
    };
    
#endif // USE_PTHREAD_SPINLOCK
}

#else
#include <windows.h>
    namespace Partio{

   class PartioMutex
    {
        HANDLE CacheLock;
    
    public:
        inline PartioMutex()
        {
            CacheLock=CreateMutex(0,FALSE,"partiocache");
        }
    
        inline ~PartioMutex()
        {
            CloseHandle(CacheLock);
        }
    
        inline void lock()
        {
            WaitForSingleObject(CacheLock,INFINITE);
        }
    
        inline void unlock()
        {
            ReleaseMutex(CacheLock);
        }
    };
    }
#endif // USE_PTHREADS
#endif // Header guard
