///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012-2018 DreamWorks Animation LLC
//
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
//
// Redistributions of source code must retain the above copyright
// and license notice and the following restrictions and disclaimer.
//
// *     Neither the name of DreamWorks Animation nor the names of
// its contributors may be used to endorse or promote products derived
// from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// IN NO EVENT SHALL THE COPYRIGHT HOLDERS' AND CONTRIBUTORS' AGGREGATE
// LIABILITY FOR ALL CLAIMS REGARDLESS OF THEIR BASIS EXCEED US$250.00.
//
///////////////////////////////////////////////////////////////////////////
//
/// @file NullInterrupter.h

#ifndef OPENVDB_UTIL_NULL_INTERRUPTER_HAS_BEEN_INCLUDED
#define OPENVDB_UTIL_NULL_INTERRUPTER_HAS_BEEN_INCLUDED

#include <openvdb/version.h>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace util {

/// @brief Dummy NOOP interrupter class defining interface
///
/// This shows the required interface for the @c InterrupterType template argument
/// using by several threaded applications (e.g. tools/PointAdvect.h). The host
/// application calls start() at the beginning of an interruptible operation, end()
/// at the end of the operation, and wasInterrupted() periodically during the operation.
/// If any call to wasInterrupted() returns @c true, the operation will be aborted.
/// @note This Dummy interrupter will NEVER interrupt since wasInterrupted() always
/// returns false!
struct NullInterrupter
{
    /// Default constructor
    NullInterrupter () {}
    /// Signal the start of an interruptible operation.
    /// @param name  an optional descriptive name for the operation
    void start(const char* name = NULL) { (void)name; }
    /// Signal the end of an interruptible operation.
    void end() {}
    /// Check if an interruptible operation should be aborted.
    /// @param percent  an optional (when >= 0) percentage indicating
    ///     the fraction of the operation that has been completed
    /// @note this method is assumed to be thread-safe. The current
    /// implementation is clearly a NOOP and should compile out during
    /// optimization!
    inline bool wasInterrupted(int percent = -1) { (void)percent; return false; }
};

/// This method allows NullInterrupter::wasInterrupted to be compiled
/// out when client code only has a pointer (vs reference) to the interrupter.
///
/// @note This is a free-standing function since C++ doesn't allow for
/// partial template specialization (in client code of the interrupter).
template <typename T>
inline bool wasInterrupted(T* i, int percent = -1) { return i && i->wasInterrupted(percent); }

/// Specialization for NullInterrupter
template<>
inline bool wasInterrupted<util::NullInterrupter>(util::NullInterrupter*, int) { return false; }

} // namespace util
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_UTIL_NULL_INTERRUPTER_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
