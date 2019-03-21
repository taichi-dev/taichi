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
/// @author Ken Museth
///
/// @file Formats.h
///
/// @brief Utility routines to output nicely-formatted numeric values


#ifndef OPENVDB_UTIL_FORMATS_HAS_BEEN_INCLUDED
#define OPENVDB_UTIL_FORMATS_HAS_BEEN_INCLUDED

#include <iosfwd>
#include <sstream>
#include <string>
#include <openvdb/version.h>
#include <openvdb/Platform.h>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace util {

/// Output a byte count with the correct binary suffix (KB, MB, GB or TB).
/// @param os         the output stream
/// @param bytes      the byte count to be output
/// @param head       a string to be output before the numeric text
/// @param tail       a string to be output after the numeric text
/// @param exact      if true, also output the unmodified count, e.g., "4.6 KB (4620 Bytes)"
/// @param width      a fixed width for the numeric text
/// @param precision  the number of digits after the decimal point
/// @return 0, 1, 2, 3 or 4, denoting the order of magnitude of the count.
OPENVDB_API int
printBytes(std::ostream& os, uint64_t bytes,
    const std::string& head = "",
    const std::string& tail = "\n",
    bool exact = false, int width = 8, int precision = 3);

/// Output a number with the correct SI suffix (thousand, million, billion or trillion)
/// @param os         the output stream
/// @param number     the number to be output
/// @param head       a string to be output before the numeric text
/// @param tail       a string to be output after the numeric text
/// @param exact      if true, also output the unmodified count, e.g., "4.6 Thousand (4620)"
/// @param width      a fixed width for the numeric text
/// @param precision  the number of digits after the decimal point
/// @return 0, 1, 2, 3 or 4, denoting the order of magnitude of the number.
OPENVDB_API int
printNumber(std::ostream& os, uint64_t number,
    const std::string& head = "",
    const std::string& tail = "\n",
    bool exact = true, int width = 8, int precision = 3);


////////////////////////////////////////


/// @brief I/O manipulator that formats integer values with thousands separators
template<typename IntT>
class FormattedInt
{
public:
    static char sep() { return ','; }

    FormattedInt(IntT n): mInt(n) {}

    std::ostream& put(std::ostream& os) const
    {
        // Convert the integer to a string.
        std::ostringstream ostr;
        ostr << mInt;
        std::string s = ostr.str();
        // Prefix the string with spaces if its length is not a multiple of three.
        size_t padding = (s.size() % 3) ? 3 - (s.size() % 3) : 0;
        s = std::string(padding, ' ') + s;
        // Construct a new string in which groups of three digits are followed
        // by a separator character.
        ostr.str("");
        for (size_t i = 0, N = s.size(); i < N; ) {
            ostr << s[i];
            ++i;
            if (i >= padding && i % 3 == 0 && i < s.size()) {
                ostr << sep();
            }
        }
        // Remove any padding that was added and output the string.
        s = ostr.str();
        os << s.substr(padding, s.size());
        return os;
    }

private:
    IntT mInt;
};

template<typename IntT>
std::ostream& operator<<(std::ostream& os, const FormattedInt<IntT>& n) { return n.put(os); }

/// @return an I/O manipulator that formats the given integer value for output to a stream.
template<typename IntT>
FormattedInt<IntT> formattedInt(IntT n) { return FormattedInt<IntT>(n); }

} // namespace util
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_UTIL_FORMATS_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
