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

#include <cppunit/extensions/HelperMacros.h>
#include <openvdb/Exceptions.h>


class TestExceptions : public CppUnit::TestCase
{
public:
    CPPUNIT_TEST_SUITE(TestExceptions);
    CPPUNIT_TEST(testArithmeticError);
    CPPUNIT_TEST(testIndexError);
    CPPUNIT_TEST(testIoError);
    CPPUNIT_TEST(testKeyError);
    CPPUNIT_TEST(testLookupError);
    CPPUNIT_TEST(testNotImplementedError);
    CPPUNIT_TEST(testReferenceError);
    CPPUNIT_TEST(testRuntimeError);
    CPPUNIT_TEST(testTypeError);
    CPPUNIT_TEST(testValueError);
    CPPUNIT_TEST_SUITE_END();

    void testArithmeticError() { testException<openvdb::ArithmeticError>(); }
    void testIndexError() { testException<openvdb::IndexError>(); }
    void testIoError() { testException<openvdb::IoError>(); }
    void testKeyError() { testException<openvdb::KeyError>(); }
    void testLookupError() { testException<openvdb::LookupError>(); }
    void testNotImplementedError() { testException<openvdb::NotImplementedError>(); }
    void testReferenceError() { testException<openvdb::ReferenceError>(); }
    void testRuntimeError() { testException<openvdb::RuntimeError>(); }
    void testTypeError() { testException<openvdb::TypeError>(); }
    void testValueError() { testException<openvdb::ValueError>(); }

private:
    template<typename ExceptionT> void testException();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestExceptions);


template<typename ExceptionT> struct ExceptionTraits
{ static std::string name() { return ""; } };
template<> struct ExceptionTraits<openvdb::ArithmeticError>
{ static std::string name() { return "ArithmeticError"; } };
template<> struct ExceptionTraits<openvdb::IndexError>
{ static std::string name() { return "IndexError"; } };
template<> struct ExceptionTraits<openvdb::IoError>
{ static std::string name() { return "IoError"; } };
template<> struct ExceptionTraits<openvdb::KeyError>
{ static std::string name() { return "KeyError"; } };
template<> struct ExceptionTraits<openvdb::LookupError>
{ static std::string name() { return "LookupError"; } };
template<> struct ExceptionTraits<openvdb::NotImplementedError>
{ static std::string name() { return "NotImplementedError"; } };
template<> struct ExceptionTraits<openvdb::ReferenceError>
{ static std::string name() { return "ReferenceError"; } };
template<> struct ExceptionTraits<openvdb::RuntimeError>
{ static std::string name() { return "RuntimeError"; } };
template<> struct ExceptionTraits<openvdb::TypeError>
{ static std::string name() { return "TypeError"; } };
template<> struct ExceptionTraits<openvdb::ValueError>
{ static std::string name() { return "ValueError"; } };


template<typename ExceptionT>
void
TestExceptions::testException()
{
    std::string ErrorMsg("Error message");

    CPPUNIT_ASSERT_THROW(OPENVDB_THROW(ExceptionT, ErrorMsg), ExceptionT);

    try {
        OPENVDB_THROW(ExceptionT, ErrorMsg);
    } catch (openvdb::Exception& e) {
        const std::string expectedMsg = ExceptionTraits<ExceptionT>::name() + ": " + ErrorMsg;
        CPPUNIT_ASSERT_EQUAL(expectedMsg, std::string(e.what()));
    }
}

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
