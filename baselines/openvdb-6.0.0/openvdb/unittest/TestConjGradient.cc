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
#include <openvdb/openvdb.h>
#include <openvdb/version.h>
#include <openvdb/math/ConjGradient.h>


class TestConjGradient: public CppUnit::TestCase
{
public:
    CPPUNIT_TEST_SUITE(TestConjGradient);
    CPPUNIT_TEST(testJacobi);
    CPPUNIT_TEST(testIncompleteCholesky);
    CPPUNIT_TEST(testVectorDotProduct);
    CPPUNIT_TEST_SUITE_END();

    void testJacobi();
    void testIncompleteCholesky();
    void testVectorDotProduct();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestConjGradient);


////////////////////////////////////////


void
TestConjGradient::testJacobi()
{
    using namespace openvdb;

    typedef math::pcg::SparseStencilMatrix<double, 7> MatrixType;

    const math::pcg::SizeType rows = 5;

    MatrixType A(rows);
    A.setValue(0, 0, 24.0);
    A.setValue(0, 2,  6.0);
    A.setValue(1, 1,  8.0);
    A.setValue(1, 2,  2.0);
    A.setValue(2, 0,  6.0);
    A.setValue(2, 1,  2.0);
    A.setValue(2, 2,  8.0);
    A.setValue(2, 3, -6.0);
    A.setValue(2, 4,  2.0);
    A.setValue(3, 2, -6.0);
    A.setValue(3, 3, 24.0);
    A.setValue(4, 2,  2.0);
    A.setValue(4, 4,  8.0);

    CPPUNIT_ASSERT(A.isFinite());

    MatrixType::VectorType
        x(rows, 0.0),
        b(rows, 1.0),
        expected(rows);

    expected[0] = 0.0104167;
    expected[1] = 0.09375;
    expected[2] = 0.125;
    expected[3] = 0.0729167;
    expected[4] = 0.09375;

    math::pcg::JacobiPreconditioner<MatrixType> precond(A);

    // Solve A * x = b for x.
    math::pcg::State result = math::pcg::solve(
        A, b, x, precond, math::pcg::terminationDefaults<double>());

    CPPUNIT_ASSERT(result.success);
    CPPUNIT_ASSERT(result.iterations <= 20);
    CPPUNIT_ASSERT(x.eq(expected, 1.0e-5));
}


void
TestConjGradient::testIncompleteCholesky()
{
    using namespace openvdb;

    typedef math::pcg::SparseStencilMatrix<double, 7> MatrixType;
    typedef math::pcg::IncompleteCholeskyPreconditioner<MatrixType> CholeskyPrecond;

    const math::pcg::SizeType rows = 5;

    MatrixType A(5);
    A.setValue(0, 0, 24.0);
    A.setValue(0, 2,  6.0);
    A.setValue(1, 1,  8.0);
    A.setValue(1, 2,  2.0);
    A.setValue(2, 0,  6.0);
    A.setValue(2, 1,  2.0);
    A.setValue(2, 2,  8.0);
    A.setValue(2, 3, -6.0);
    A.setValue(2, 4,  2.0);
    A.setValue(3, 2, -6.0);
    A.setValue(3, 3, 24.0);
    A.setValue(4, 2,  2.0);
    A.setValue(4, 4,  8.0);

    CPPUNIT_ASSERT(A.isFinite());

    CholeskyPrecond precond(A);
    {
        const CholeskyPrecond::TriangularMatrix lower = precond.lowerMatrix();

        CholeskyPrecond::TriangularMatrix expected(5);
        expected.setValue(0, 0,  4.89898);
        expected.setValue(1, 1,  2.82843);
        expected.setValue(2, 0,  1.22474);
        expected.setValue(2, 1,  0.707107);
        expected.setValue(2, 2,  2.44949);
        expected.setValue(3, 2, -2.44949);
        expected.setValue(3, 3,  4.24264);
        expected.setValue(4, 2,  0.816497);
        expected.setValue(4, 4,  2.70801);

#if 0
        std::cout << "Expected:\n";
        for (int i = 0; i < 5; ++i) {
            std::cout << "    " << expected.getConstRow(i).str() << std::endl;
        }
        std::cout << "Actual:\n";
        for (int i = 0; i < 5; ++i) {
            std::cout << "    " << lower.getConstRow(i).str() << std::endl;
        }
#endif

        CPPUNIT_ASSERT(lower.eq(expected, 1.0e-5));
    }
    {
        const CholeskyPrecond::TriangularMatrix upper = precond.upperMatrix();

        CholeskyPrecond::TriangularMatrix expected(5);
        {
            expected.setValue(0, 0,  4.89898);
            expected.setValue(0, 2,  1.22474);
            expected.setValue(1, 1,  2.82843);
            expected.setValue(1, 2,  0.707107);
            expected.setValue(2, 2,  2.44949);
            expected.setValue(2, 3, -2.44949);
            expected.setValue(2, 4,  0.816497);
            expected.setValue(3, 3,  4.24264);
            expected.setValue(4, 4,  2.70801);
        }

#if 0
        std::cout << "Expected:\n";
        for (int i = 0; i < 5; ++i) {
            std::cout << "    " << expected.getConstRow(i).str() << std::endl;
        }
        std::cout << "Actual:\n";
        for (int i = 0; i < 5; ++i) {
            std::cout << "    " << upper.getConstRow(i).str() << std::endl;
        }
#endif

        CPPUNIT_ASSERT(upper.eq(expected, 1.0e-5));
    }

    MatrixType::VectorType
        x(rows, 0.0),
        b(rows, 1.0),
        expected(rows);

    expected[0] = 0.0104167;
    expected[1] = 0.09375;
    expected[2] = 0.125;
    expected[3] = 0.0729167;
    expected[4] = 0.09375;

    // Solve A * x = b for x.
    math::pcg::State result = math::pcg::solve(
        A, b, x, precond, math::pcg::terminationDefaults<double>());

    CPPUNIT_ASSERT(result.success);
    CPPUNIT_ASSERT(result.iterations <= 20);
    CPPUNIT_ASSERT(x.eq(expected, 1.0e-5));
}

void
TestConjGradient::testVectorDotProduct()
{
    using namespace openvdb;

    typedef math::pcg::Vector<double>  VectorType;

    // Test small vector - runs in series
    {
        const size_t length = 1000;
        VectorType aVec(length, 2.0);
        VectorType bVec(length, 3.0);

        VectorType::ValueType result = aVec.dot(bVec);

        CPPUNIT_ASSERT_DOUBLES_EQUAL(result, 6.0 * length, 1.0e-7);
    }
    // Test long vector  - runs in parallel
    {
        const size_t length = 10034502;
        VectorType aVec(length, 2.0);
        VectorType bVec(length, 3.0);

        VectorType::ValueType result = aVec.dot(bVec);

        CPPUNIT_ASSERT_DOUBLES_EQUAL(result, 6.0 * length, 1.0e-7);
    }
}

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
