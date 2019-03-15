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

#include "Formats.h"

#include <openvdb/Platform.h>
#include <iostream>
#include <iomanip>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace util {

int
printBytes(std::ostream& os, uint64_t bytes,
    const std::string& head, const std::string& tail,
    bool exact, int width, int precision)
{
    const uint64_t one = 1;
    int group = 0;

    // Write to a string stream so that I/O manipulators like
    // std::setprecision() don't alter the output stream.
    std::ostringstream ostr;
    ostr << head;
    ostr << std::setprecision(precision) << std::setiosflags(std::ios::fixed);
    if (bytes >> 40) {
        ostr << std::setw(width) << (double(bytes) / double(one << 40)) << " TB";
        group = 4;
    } else if (bytes >> 30) {
        ostr << std::setw(width) << (double(bytes) / double(one << 30)) << " GB";
        group = 3;
    } else if (bytes >> 20) {
        ostr << std::setw(width) << (double(bytes) / double(one << 20)) << " MB";
        group = 2;
    } else if (bytes >> 10) {
        ostr << std::setw(width) << (double(bytes) / double(one << 10)) << " KB";
        group = 1;
    } else {
        ostr << std::setw(width) << bytes << " Bytes";
    }
    if (exact && group) ostr << " (" << bytes << " Bytes)";
    ostr << tail;

    os << ostr.str();

    return group;
}


int
printNumber(std::ostream& os, uint64_t number,
    const std::string& head, const std::string& tail,
    bool exact, int width, int precision)
{
    int group = 0;

    // Write to a string stream so that I/O manipulators like
    // std::setprecision() don't alter the output stream.
    std::ostringstream ostr;
    ostr << head;
    ostr << std::setprecision(precision) << std::setiosflags(std::ios::fixed);
    if (number / UINT64_C(1000000000000)) {
        ostr << std::setw(width) << (double(number) / 1000000000000.0) << " trillion";
        group = 4;
    } else if (number / UINT64_C(1000000000)) {
        ostr << std::setw(width) << (double(number) / 1000000000.0) << " billion";
        group = 3;
    } else if (number / UINT64_C(1000000)) {
        ostr << std::setw(width) << (double(number) / 1000000.0) << " million";
        group = 2;
    } else if (number / UINT64_C(1000)) {
        ostr << std::setw(width) << (double(number) / 1000.0) << " thousand";
        group = 1;
    } else {
        ostr << std::setw(width) << number;
    }
    if (exact && group) ostr << " (" << number << ")";
    ostr << tail;

    os << ostr.str();

    return group;
}

} // namespace util
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
