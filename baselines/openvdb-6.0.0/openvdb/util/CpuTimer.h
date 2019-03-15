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

#ifndef OPENVDB_UTIL_CPUTIMER_HAS_BEEN_INCLUDED
#define OPENVDB_UTIL_CPUTIMER_HAS_BEEN_INCLUDED

#include <openvdb/version.h>
#include <string>
#include <tbb/tick_count.h>
#include <iostream>// for std::cerr
#include <sstream>// for ostringstream
#include <iomanip>//for setprecision

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace util {

/// @brief Simple timer for basic profiling.
///
/// @code
///    CpuTimer timer;
///    // code here will not be timed!
///    timer.start("algorithm");
///    // code to be timed goes here
///    timer.stop();
/// @endcode
///
/// or to time multiple blocks of code
///
/// @code
///    CpuTimer timer("algorithm 1");
///    // code to be timed goes here
///    timer.restart("algorithm 2");
///    // code to be timed goes here
///    timer.stop();
/// @endcode
class CpuTimer
{
public:

    /// @brief Initiate timer
    CpuTimer() : mT0(tbb::tick_count::now()) {}

    /// @brief Prints message and re-start timer.
    ///
    /// @note Should normally be followed by a call to stop()
    CpuTimer(const std::string& msg) { this->start(msg); }

    /// @brief Start timer.
    ///
    /// @note Should normally be followed by a call to time()
    inline void start() { mT0 = tbb::tick_count::now(); }

    /// @brief Print message and re-start timer.
    ///
    /// @note Should normally be followed by a call to stop()
    inline void start(const std::string& msg)
    {
        std::cerr << msg << " ... ";
        this->start();
    }

    /// @brief Stop previous timer, print message and re-start timer.
    ///
    /// @note Should normally be followed by a call to stop()
    inline void restart(const std::string& msg)
    {
        this->stop();
        this->start(msg);
    }

    /// Return Time diference in milliseconds since construction or start was called.
    inline double delta() const
    {
        tbb::tick_count::interval_t dt = tbb::tick_count::now() - mT0;
        return 1000.0*dt.seconds();
    }

    /// @brief Print time in milliseconds since construction or start was called.
    inline void stop() const
    {
        const double t = this->delta();
        std::ostringstream ostr;
        ostr << "completed in " << std::setprecision(3) << t << " ms\n";
        std::cerr << ostr.str();
    }

private:

    tbb::tick_count mT0;
};// CpuTimer

} // namespace util
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb


#endif // OPENVDB_UTIL_CPUTIMER_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
