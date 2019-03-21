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

#ifndef OPENVDB_EXCEPTIONS_HAS_BEEN_INCLUDED
#define OPENVDB_EXCEPTIONS_HAS_BEEN_INCLUDED

#include <openvdb/version.h>
#include <exception>
#include <sstream>
#include <string>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {

class OPENVDB_API Exception: public std::exception
{
public:
    Exception(const Exception&) = default;
    Exception(Exception&&) = default;
    Exception& operator=(const Exception&) = default;
    Exception& operator=(Exception&&) = default;
    ~Exception() override = default;

    const char* what() const noexcept override
    {
        try { return mMessage.c_str(); } catch (...) {}
        return nullptr;
    }

protected:
    Exception() noexcept {}
    explicit Exception(const char* eType, const std::string* const msg = nullptr) noexcept
    {
        try {
            if (eType) mMessage = eType;
            if (msg) mMessage += ": " + (*msg);
        } catch (...) {}
    }

private:
    std::string mMessage;
};


#define OPENVDB_EXCEPTION(_classname) \
class OPENVDB_API _classname: public Exception \
{ \
public: \
    _classname() noexcept: Exception( #_classname ) {} \
    explicit _classname(const std::string& msg) noexcept: Exception( #_classname , &msg) {} \
}


OPENVDB_EXCEPTION(ArithmeticError);
OPENVDB_EXCEPTION(IndexError);
OPENVDB_EXCEPTION(IoError);
OPENVDB_EXCEPTION(KeyError);
OPENVDB_EXCEPTION(LookupError);
OPENVDB_EXCEPTION(NotImplementedError);
OPENVDB_EXCEPTION(ReferenceError);
OPENVDB_EXCEPTION(RuntimeError);
OPENVDB_EXCEPTION(TypeError);
OPENVDB_EXCEPTION(ValueError);

#undef OPENVDB_EXCEPTION


/// @deprecated Use ValueError instead.
class OPENVDB_API IllegalValueException: public Exception {
public:
    OPENVDB_DEPRECATED IllegalValueException() noexcept: Exception("IllegalValueException") {}
    OPENVDB_DEPRECATED explicit IllegalValueException(const std::string& msg) noexcept:
        Exception("IllegalValueException", &msg) {}
};

} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb


#define OPENVDB_THROW(exception, message) \
{ \
    std::string _openvdb_throw_msg; \
    try { \
        std::ostringstream _openvdb_throw_os; \
        _openvdb_throw_os << message; \
        _openvdb_throw_msg = _openvdb_throw_os.str(); \
    } catch (...) {} \
    throw exception(_openvdb_throw_msg); \
} // OPENVDB_THROW

#endif // OPENVDB_EXCEPTIONS_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
