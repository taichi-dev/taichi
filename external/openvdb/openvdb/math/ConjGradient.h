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

/// @file    ConjGradient.h
/// @authors D.J. Hill, Peter Cucka
/// @brief   Preconditioned conjugate gradient solver (solves @e Ax = @e b using
///          the conjugate gradient method with one of a selection of preconditioners)

#ifndef OPENVDB_MATH_CONJGRADIENT_HAS_BEEN_INCLUDED
#define OPENVDB_MATH_CONJGRADIENT_HAS_BEEN_INCLUDED

#include <openvdb/Exceptions.h>
#include <openvdb/Types.h>
#include <openvdb/util/logging.h>
#include <openvdb/util/NullInterrupter.h>
#include "Math.h" // for Abs(), isZero(), Max(), Sqrt()
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <algorithm> // for std::lower_bound()
#include <cassert>
#include <cmath> // for std::isfinite()
#include <limits>
#include <sstream>
#include <string>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace math {
namespace pcg {

using SizeType = Index32;

using SizeRange = tbb::blocked_range<SizeType>;

template<typename ValueType> class Vector;

template<typename ValueType, SizeType STENCIL_SIZE> class SparseStencilMatrix;

template<typename ValueType> class Preconditioner;
template<typename MatrixType> class JacobiPreconditioner;
template<typename MatrixType> class IncompleteCholeskyPreconditioner;

/// Information about the state of a conjugate gradient solution
struct State {
    bool    success;
    int     iterations;
    double  relativeError;
    double  absoluteError;
};


/// Return default termination conditions for a conjugate gradient solver.
template<typename ValueType>
inline State
terminationDefaults()
{
    State s;
    s.success = false;
    s.iterations = 50;
    s.relativeError = 1.0e-6;
    s.absoluteError = std::numeric_limits<ValueType>::epsilon() * 100.0;
    return s;
}


////////////////////////////////////////


/// @brief Solve @e Ax = @e b via the preconditioned conjugate gradient method.
///
/// @param A  a symmetric, positive-definite, @e N x @e N matrix
/// @param b  a vector of size @e N
/// @param x  a vector of size @e N
/// @param preconditioner  a Preconditioner matrix
/// @param termination  termination conditions given as a State object with the following fields:
///     <dl>
///     <dt><i>success</i>
///     <dd>ignored
///     <dt><i>iterations</i>
///     <dd>the maximum number of iterations, with or without convergence
///     <dt><i>relativeError</i>
///     <dd>the relative error ||<i>b</i> &minus; <i>Ax</i>|| / ||<i>b</i>||
///         that denotes convergence
///     <dt><i>absoluteError</i>
///     <dd>the absolute error ||<i>b</i> &minus; <i>Ax</i>|| that denotes convergence
///
/// @throw ArithmeticError if either @a x or @a b is not of the appropriate size.
template<typename PositiveDefMatrix>
inline State
solve(
    const PositiveDefMatrix& A,
    const Vector<typename PositiveDefMatrix::ValueType>& b,
    Vector<typename PositiveDefMatrix::ValueType>& x,
    Preconditioner<typename PositiveDefMatrix::ValueType>& preconditioner,
    const State& termination = terminationDefaults<typename PositiveDefMatrix::ValueType>());


/// @brief Solve @e Ax = @e b via the preconditioned conjugate gradient method.
///
/// @param A  a symmetric, positive-definite, @e N x @e N matrix
/// @param b  a vector of size @e N
/// @param x  a vector of size @e N
/// @param preconditioner  a Preconditioner matrix
/// @param termination  termination conditions given as a State object with the following fields:
///     <dl>
///     <dt><i>success</i>
///     <dd>ignored
///     <dt><i>iterations</i>
///     <dd>the maximum number of iterations, with or without convergence
///     <dt><i>relativeError</i>
///     <dd>the relative error ||<i>b</i> &minus; <i>Ax</i>|| / ||<i>b</i>||
///         that denotes convergence
///     <dt><i>absoluteError</i>
///     <dd>the absolute error ||<i>b</i> &minus; <i>Ax</i>|| that denotes convergence
/// @param interrupter  an object adhering to the util::NullInterrupter interface
///     with which computation can be interrupted
///
/// @throw ArithmeticError if either @a x or @a b is not of the appropriate size.
/// @throw RuntimeError if the computation is interrupted.
template<typename PositiveDefMatrix, typename Interrupter>
inline State
solve(
    const PositiveDefMatrix& A,
    const Vector<typename PositiveDefMatrix::ValueType>& b,
    Vector<typename PositiveDefMatrix::ValueType>& x,
    Preconditioner<typename PositiveDefMatrix::ValueType>& preconditioner,
    Interrupter& interrupter,
    const State& termination = terminationDefaults<typename PositiveDefMatrix::ValueType>());


////////////////////////////////////////


/// Lightweight, variable-length vector
template<typename T>
class Vector
{
public:
    using ValueType = T;
    using Ptr = SharedPtr<Vector>;

    /// Construct an empty vector.
    Vector(): mData(nullptr), mSize(0) {}
    /// Construct a vector of @a n elements, with uninitialized values.
    Vector(SizeType n): mData(new T[n]), mSize(n) {}
    /// Construct a vector of @a n elements and initialize each element to the given value.
    Vector(SizeType n, const ValueType& val): mData(new T[n]), mSize(n) { this->fill(val); }

    ~Vector() { mSize = 0; delete[] mData; mData = nullptr; }

    /// Deep copy the given vector.
    Vector(const Vector&);
    /// Deep copy the given vector.
    Vector& operator=(const Vector&);

    /// Return the number of elements in this vector.
    SizeType size() const { return mSize; }
    /// Return @c true if this vector has no elements.
    bool empty() const { return (mSize == 0); }

    /// @brief Reset this vector to have @a n elements, with uninitialized values.
    /// @warning All of this vector's existing values will be lost.
    void resize(SizeType n);

    /// Swap internal storage with another vector, which need not be the same size.
    void swap(Vector& other) { std::swap(mData, other.mData); std::swap(mSize, other.mSize); }

    /// Set all elements of this vector to @a value.
    void fill(const ValueType& value);

    //@{
    /// @brief Multiply each element of this vector by @a s.
    template<typename Scalar> void scale(const Scalar& s);
    template<typename Scalar> Vector& operator*=(const Scalar& s) { this->scale(s); return *this; }
    //@}

    /// Return the dot product of this vector with the given vector, which must be the same size.
    ValueType dot(const Vector&) const;

    /// Return the infinity norm of this vector.
    ValueType infNorm() const;
    /// Return the L2 norm of this vector.
    ValueType l2Norm() const { return Sqrt(this->dot(*this)); }

    /// Return @c true if every element of this vector has a finite value.
    bool isFinite() const;

    /// @brief Return @c true if this vector is equivalent to the given vector
    /// to within the specified tolerance.
    template<typename OtherValueType>
    bool eq(const Vector<OtherValueType>& other,
        ValueType eps = Tolerance<ValueType>::value()) const;

    /// Return a string representation of this vector.
    std::string str() const;

    //@{
    /// @brief Return the value of this vector's ith element.
    inline T& at(SizeType i) { return mData[i]; }
    inline const T& at(SizeType i) const { return mData[i]; }
    inline T& operator[](SizeType i) { return this->at(i); }
    inline const T& operator[](SizeType i) const { return this->at(i); }
    //@}

    //@{
    /// @brief Return a pointer to this vector's elements.
    inline T* data() { return mData; }
    inline const T* data() const { return mData; }
    inline const T* constData() const { return mData; }
    //@}

private:
    // Functor for use with tbb::parallel_for()
    template<typename Scalar> struct ScaleOp;
    struct DeterministicDotProductOp;
    // Functors for use with tbb::parallel_reduce()
    template<typename OtherValueType> struct EqOp;
    struct InfNormOp;
    struct IsFiniteOp;

    T* mData;
    SizeType mSize;
};

using VectorS = Vector<float>;
using VectorD = Vector<double>;


////////////////////////////////////////


/// @brief Sparse, square matrix representing a 3D stencil operator of size @a STENCIL_SIZE
/// @details The implementation is a variation on compressed row storage (CRS).
template<typename ValueType_, SizeType STENCIL_SIZE>
class SparseStencilMatrix
{
public:
    using ValueType = ValueType_;
    using VectorType = Vector<ValueType>;
    using Ptr = SharedPtr<SparseStencilMatrix>;

    class ConstValueIter;
    class ConstRow;
    class RowEditor;

    static const ValueType sZeroValue;

    /// Construct an @a n x @a n matrix with at most @a STENCIL_SIZE nonzero elements per row.
    SparseStencilMatrix(SizeType n);

    /// Deep copy the given matrix.
    SparseStencilMatrix(const SparseStencilMatrix&);

    //@{
    /// Return the number of rows in this matrix.
    SizeType numRows() const { return mNumRows; }
    SizeType size() const { return mNumRows; }
    //@}

    /// @brief Set the value at the given coordinates.
    /// @warning It is not safe to set values in the same row simultaneously
    /// from multiple threads.
    void setValue(SizeType row, SizeType col, const ValueType&);

    //@{
    /// @brief Return the value at the given coordinates.
    /// @warning It is not safe to get values from a row while another thread
    /// is setting values in that row.
    const ValueType& getValue(SizeType row, SizeType col) const;
    const ValueType& operator()(SizeType row, SizeType col) const;
    //@}

    /// Return a read-only view onto the given row of this matrix.
    ConstRow getConstRow(SizeType row) const;

    /// Return a read/write view onto the given row of this matrix.
    RowEditor getRowEditor(SizeType row);

    //@{
    /// @brief Multiply all elements in the matrix by @a s;
    template<typename Scalar> void scale(const Scalar& s);
    template<typename Scalar>
    SparseStencilMatrix& operator*=(const Scalar& s) { this->scale(s); return *this; }
    //@}

    /// @brief Multiply this matrix by @a inVec and return the result in @a resultVec.
    /// @throw ArithmeticError if either @a inVec or @a resultVec is not of size @e N,
    /// where @e N x @e N is the size of this matrix.
    template<typename VecValueType>
    void vectorMultiply(const Vector<VecValueType>& inVec, Vector<VecValueType>& resultVec) const;

    /// @brief Multiply this matrix by the vector represented by the array @a inVec
    /// and return the result in @a resultVec.
    /// @warning Both @a inVec and @a resultVec must have at least @e N elements,
    /// where @e N x @e N is the size of this matrix.
    template<typename VecValueType>
    void vectorMultiply(const VecValueType* inVec, VecValueType* resultVec) const;

    /// @brief Return @c true if this matrix is equivalent to the given matrix
    /// to within the specified tolerance.
    template<typename OtherValueType>
    bool eq(const SparseStencilMatrix<OtherValueType, STENCIL_SIZE>& other,
        ValueType eps = Tolerance<ValueType>::value()) const;

    /// Return @c true if every element of this matrix has a finite value.
    bool isFinite() const;

    /// Return a string representation of this matrix.
    std::string str() const;

private:
    struct RowData {
        RowData(ValueType* v, SizeType* c, SizeType& s): mVals(v), mCols(c), mSize(s) {}
        ValueType* mVals; SizeType* mCols; SizeType& mSize;
    };

    struct ConstRowData {
        ConstRowData(const ValueType* v, const SizeType* c, const SizeType& s):
            mVals(v), mCols(c), mSize(s) {}
        const ValueType* mVals; const SizeType* mCols; const SizeType& mSize;
    };

    /// Base class for row accessors
    template<typename DataType_ = RowData>
    class RowBase
    {
    public:
        using DataType = DataType_;

        static SizeType capacity() { return STENCIL_SIZE; }

        RowBase(const DataType& data): mData(data) {}

        bool empty() const { return (mData.mSize == 0); }
        const SizeType& size() const { return mData.mSize; }

        const ValueType& getValue(SizeType columnIdx, bool& active) const;
        const ValueType& getValue(SizeType columnIdx) const;

        /// Return an iterator over the stored values in this row.
        ConstValueIter cbegin() const;

        /// @brief Return @c true if this row is equivalent to the given row
        /// to within the specified tolerance.
        template<typename OtherDataType>
        bool eq(const RowBase<OtherDataType>& other,
            ValueType eps = Tolerance<ValueType>::value()) const;

        /// @brief Return the dot product of this row with the first
        /// @a vecSize elements of @a inVec.
        /// @warning @a inVec must have at least @a vecSize elements.
        template<typename VecValueType>
        VecValueType dot(const VecValueType* inVec, SizeType vecSize) const;

        /// Return the dot product of this row with the given vector.
        template<typename VecValueType>
        VecValueType dot(const Vector<VecValueType>& inVec) const;

        /// Return a string representation of this row.
        std::string str() const;

    protected:
        friend class ConstValueIter;

        const ValueType& value(SizeType i) const { return mData.mVals[i]; }
        SizeType column(SizeType i) const { return mData.mCols[i]; }

        /// @brief Return the array index of the first column index that is
        /// equal to <i>or greater than</i> the given column index.
        /// @note If @a columnIdx is larger than any existing column index,
        /// the return value will point beyond the end of the array.
        SizeType find(SizeType columnIdx) const;

        DataType mData;
    };

    using ConstRowBase = RowBase<ConstRowData>;

public:
    /// Iterator over the stored values in a row of this matrix
    class ConstValueIter
    {
    public:
        const ValueType& operator*() const
        {
            if (mData.mSize == 0) return SparseStencilMatrix::sZeroValue;
            return mData.mVals[mCursor];
        }

        SizeType column() const { return mData.mCols[mCursor]; }

        void increment() { mCursor++; }
        ConstValueIter& operator++() { increment(); return *this; }
        operator bool() const { return (mCursor < mData.mSize); }

        void reset() { mCursor = 0; }

    private:
        friend class SparseStencilMatrix;
        ConstValueIter(const RowData& d): mData(d.mVals, d.mCols, d.mSize), mCursor(0) {}
        ConstValueIter(const ConstRowData& d): mData(d), mCursor(0) {}

        const ConstRowData mData;
        SizeType mCursor;
    };


    /// Read-only accessor to a row of this matrix
    class ConstRow: public ConstRowBase
    {
    public:
        ConstRow(const ValueType* valueHead, const SizeType* columnHead, const SizeType& rowSize);
    }; // class ConstRow


    /// Read/write accessor to a row of this matrix
    class RowEditor: public RowBase<>
    {
    public:
        RowEditor(ValueType* valueHead, SizeType* columnHead, SizeType& rowSize, SizeType colSize);

        /// Set the number of entries in this row to zero.
        void clear();

        /// @brief Set the value of the entry in the specified column.
        /// @return the current number of entries stored in this row.
        SizeType setValue(SizeType column, const ValueType& value);

        //@{
        /// @brief Scale all of the entries in this row.
        template<typename Scalar> void scale(const Scalar&);
        template<typename Scalar>
        RowEditor& operator*=(const Scalar& s) { this->scale(s); return *this; }
        //@}

    private:
        const SizeType mNumColumns; // used only for bounds checking
    }; // class RowEditor

private:
    // Functors for use with tbb::parallel_for()
    struct MatrixCopyOp;
    template<typename VecValueType> struct VecMultOp;
    template<typename Scalar> struct RowScaleOp;

    // Functors for use with tbb::parallel_reduce()
    struct IsFiniteOp;
    template<typename OtherValueType> struct EqOp;

    const SizeType                  mNumRows;
    std::unique_ptr<ValueType[]>    mValueArray;
    std::unique_ptr<SizeType[]>     mColumnIdxArray;
    std::unique_ptr<SizeType[]>     mRowSizeArray;
}; // class SparseStencilMatrix


////////////////////////////////////////


/// Base class for conjugate gradient preconditioners
template<typename T>
class Preconditioner
{
public:
    using ValueType = T;
    using Ptr = SharedPtr<Preconditioner>;

    template<SizeType STENCIL_SIZE> Preconditioner(const SparseStencilMatrix<T, STENCIL_SIZE>&) {}
    virtual ~Preconditioner() = default;

    virtual bool isValid() const { return true; }

    /// @brief Apply this preconditioner to a residue vector:
    ///     @e z = <i>M</i><sup><small>&minus;1</small></sup><i>r</i>
    /// @param      r  residue vector
    /// @param[out] z  preconditioned residue vector
    virtual void apply(const Vector<T>& r, Vector<T>& z) = 0;
};


////////////////////////////////////////


namespace internal {

// Functor for use with tbb::parallel_for() to copy data from one array to another
template<typename T>
struct CopyOp
{
    CopyOp(const T* from_, T* to_): from(from_), to(to_) {}

    void operator()(const SizeRange& range) const {
        for (SizeType n = range.begin(), N = range.end(); n < N; ++n) to[n] = from[n];
    }

    const T* from;
    T* to;
};


// Functor for use with tbb::parallel_for() to fill an array with a constant value
template<typename T>
struct FillOp
{
    FillOp(T* data_, const T& val_): data(data_), val(val_) {}

    void operator()(const SizeRange& range) const {
        for (SizeType n = range.begin(), N = range.end(); n < N; ++n) data[n] = val;
    }

    T* data;
    const T val;
};


// Functor for use with tbb::parallel_for() that computes a * x + y
template<typename T>
struct LinearOp
{
    LinearOp(const T& a_, const T* x_, const T* y_, T* out_): a(a_), x(x_), y(y_), out(out_) {}

    void operator()(const SizeRange& range) const {
        if (isExactlyEqual(a, T(1))) {
            for (SizeType n = range.begin(), N = range.end(); n < N; ++n) out[n] = x[n] + y[n];
        } else if (isExactlyEqual(a, T(-1))) {
            for (SizeType n = range.begin(), N = range.end(); n < N; ++n) out[n] = -x[n] + y[n];
        } else {
            for (SizeType n = range.begin(), N = range.end(); n < N; ++n) out[n] = a * x[n] + y[n];
        }
    }

    const T a, *x, *y;
    T* out;
};

} // namespace internal


////////////////////////////////////////


inline std::ostream&
operator<<(std::ostream& os, const State& state)
{
    os << (state.success ? "succeeded with " : "")
        << "rel. err. " << state.relativeError << ", abs. err. " << state.absoluteError
        << " after " << state.iterations << " iteration" << (state.iterations == 1 ? "" : "s");
    return os;
}


////////////////////////////////////////


template<typename T>
inline
Vector<T>::Vector(const Vector& other): mData(new T[other.mSize]), mSize(other.mSize)
{
    tbb::parallel_for(SizeRange(0, mSize),
        internal::CopyOp<T>(/*from=*/other.mData, /*to=*/mData));
}


template<typename T>
inline
Vector<T>& Vector<T>::operator=(const Vector<T>& other)
{
    // Update the internal storage to the correct size

    if (mSize != other.mSize) {
        mSize = other.mSize;
        delete[] mData;
        mData = new T[mSize];
    }

    // Deep copy the data
    tbb::parallel_for(SizeRange(0, mSize),
        internal::CopyOp<T>(/*from=*/other.mData, /*to=*/mData));

    return *this;
}


template<typename T>
inline void
Vector<T>::resize(SizeType n)
{
    if (n != mSize) {
        if (mData) delete[] mData;
        mData = new T[n];
        mSize = n;
    }
}


template<typename T>
inline void
Vector<T>::fill(const ValueType& value)
{
    tbb::parallel_for(SizeRange(0, mSize), internal::FillOp<T>(mData, value));
}


template<typename T>
template<typename Scalar>
struct Vector<T>::ScaleOp
{
    ScaleOp(T* data_, const Scalar& s_): data(data_), s(s_) {}

    void operator()(const SizeRange& range) const {
        for (SizeType n = range.begin(), N = range.end(); n < N; ++n) data[n] *= s;
    }

    T* data;
    const Scalar s;
};


template<typename T>
template<typename Scalar>
inline void
Vector<T>::scale(const Scalar& s)
{
    tbb::parallel_for(SizeRange(0, mSize), ScaleOp<Scalar>(mData, s));
}


template<typename T>
struct Vector<T>::DeterministicDotProductOp
{
    DeterministicDotProductOp(const T* a_, const T* b_,
            const SizeType binCount_, const SizeType arraySize_, T* reducetmp_):
        a(a_), b(b_), binCount(binCount_), arraySize(arraySize_), reducetmp(reducetmp_) {}

    void operator()(const SizeRange& range) const
    {
        const SizeType binSize = arraySize / binCount;

        // Iterate over bins (array segments)
        for (SizeType n = range.begin(), N = range.end(); n < N; ++n) {
            const SizeType begin = n * binSize;
            const SizeType end = (n == binCount-1) ? arraySize : begin + binSize;

            // Compute the partial sum for this array segment
            T sum = zeroVal<T>();
            for (SizeType i = begin; i < end; ++i) { sum += a[i] * b[i]; }
            // Store the partial sum
            reducetmp[n] = sum;
        }
    }


    const T* a;
    const T* b;
    const SizeType binCount;
    const SizeType arraySize;
    T* reducetmp;
};

template<typename T>
inline T
Vector<T>::dot(const Vector<T>& other) const
{
    assert(this->size() == other.size());

    const T* aData = this->data();
    const T* bData = other.data();

    SizeType arraySize = this->size();

    T result = zeroVal<T>();

    if (arraySize < 1024) {

        // Compute the dot product in serial for small arrays

        for (SizeType n = 0; n < arraySize; ++n) {
            result += aData[n] * bData[n];
        }

    } else {

        // Compute the dot product by segmenting the arrays into
        // a predetermined number of sub arrays in parallel and
        // accumulate the finial result in series.

        const SizeType binCount = 100;
        T partialSums[100];

        tbb::parallel_for(SizeRange(0, binCount),
            DeterministicDotProductOp(aData, bData, binCount, arraySize, partialSums));

        for (SizeType n = 0; n < binCount; ++n) {
            result += partialSums[n];
        }
    }

    return result;
}


template<typename T>
struct Vector<T>::InfNormOp
{
    InfNormOp(const T* data_): data(data_) {}

    T operator()(const SizeRange& range, T maxValue) const
    {
        for (SizeType n = range.begin(), N = range.end(); n < N; ++n) {
            maxValue = Max(maxValue, Abs(data[n]));
        }
        return maxValue;
    }

    const T* data;
};


template<typename T>
inline T
Vector<T>::infNorm() const
{
    // Parallelize over the elements of this vector.
    T result = tbb::parallel_reduce(SizeRange(0, this->size()), /*seed=*/zeroVal<T>(),
        InfNormOp(this->data()), /*join=*/[](T max1, T max2) { return Max(max1, max2); });
    return result;
}


template<typename T>
struct Vector<T>::IsFiniteOp
{
    IsFiniteOp(const T* data_): data(data_) {}

    bool operator()(const SizeRange& range, bool finite) const
    {
        if (finite) {
            for (SizeType n = range.begin(), N = range.end(); n < N; ++n) {
                if (!std::isfinite(data[n])) return false;
            }
        }
        return finite;
    }

    const T* data;
};


template<typename T>
inline bool
Vector<T>::isFinite() const
{
    // Parallelize over the elements of this vector.
    bool finite = tbb::parallel_reduce(SizeRange(0, this->size()), /*seed=*/true,
        IsFiniteOp(this->data()),
        /*join=*/[](bool finite1, bool finite2) { return (finite1 && finite2); });
    return finite;
}


template<typename T>
template<typename OtherValueType>
struct Vector<T>::EqOp
{
    EqOp(const T* a_, const OtherValueType* b_, T e): a(a_), b(b_), eps(e) {}

    bool operator()(const SizeRange& range, bool equal) const
    {
        if (equal) {
            for (SizeType n = range.begin(), N = range.end(); n < N; ++n) {
                if (!isApproxEqual(a[n], b[n], eps)) return false;
            }
        }
        return equal;
    }

    const T* a;
    const OtherValueType* b;
    const T eps;
};


template<typename T>
template<typename OtherValueType>
inline bool
Vector<T>::eq(const Vector<OtherValueType>& other, ValueType eps) const
{
    if (this->size() != other.size()) return false;
    bool equal = tbb::parallel_reduce(SizeRange(0, this->size()), /*seed=*/true,
        EqOp<OtherValueType>(this->data(), other.data(), eps),
        /*join=*/[](bool eq1, bool eq2) { return (eq1 && eq2); });
    return equal;
}


template<typename T>
inline std::string
Vector<T>::str() const
{
    std::ostringstream ostr;
    ostr << "[";
    std::string sep;
    for (SizeType n = 0, N = this->size(); n < N; ++n) {
        ostr << sep << (*this)[n];
        sep = ", ";
    }
    ostr << "]";
    return ostr.str();
}


////////////////////////////////////////


template<typename ValueType, SizeType STENCIL_SIZE>
const ValueType SparseStencilMatrix<ValueType, STENCIL_SIZE>::sZeroValue = zeroVal<ValueType>();


template<typename ValueType, SizeType STENCIL_SIZE>
inline
SparseStencilMatrix<ValueType, STENCIL_SIZE>::SparseStencilMatrix(SizeType numRows)
    : mNumRows(numRows)
    , mValueArray(new ValueType[mNumRows * STENCIL_SIZE])
    , mColumnIdxArray(new SizeType[mNumRows * STENCIL_SIZE])
    , mRowSizeArray(new SizeType[mNumRows])
{
    // Initialize the matrix to a null state by setting the size of each row to zero.
    tbb::parallel_for(SizeRange(0, mNumRows),
        internal::FillOp<SizeType>(mRowSizeArray.get(), /*value=*/0));
}


template<typename ValueType, SizeType STENCIL_SIZE>
struct SparseStencilMatrix<ValueType, STENCIL_SIZE>::MatrixCopyOp
{
    MatrixCopyOp(const SparseStencilMatrix& from_, SparseStencilMatrix& to_):
        from(&from_), to(&to_) {}

    void operator()(const SizeRange& range) const
    {
        const ValueType* fromVal = from->mValueArray.get();
        const SizeType* fromCol = from->mColumnIdxArray.get();
        ValueType* toVal = to->mValueArray.get();
        SizeType* toCol = to->mColumnIdxArray.get();
        for (SizeType n = range.begin(), N = range.end(); n < N; ++n) {
            toVal[n] = fromVal[n];
            toCol[n] = fromCol[n];
        }
    }

    const SparseStencilMatrix* from; SparseStencilMatrix* to;
};


template<typename ValueType, SizeType STENCIL_SIZE>
inline
SparseStencilMatrix<ValueType, STENCIL_SIZE>::SparseStencilMatrix(const SparseStencilMatrix& other)
    : mNumRows(other.mNumRows)
    , mValueArray(new ValueType[mNumRows * STENCIL_SIZE])
    , mColumnIdxArray(new SizeType[mNumRows * STENCIL_SIZE])
    , mRowSizeArray(new SizeType[mNumRows])
{
    SizeType size = mNumRows * STENCIL_SIZE;

    // Copy the value and column index arrays from the other matrix to this matrix.
    tbb::parallel_for(SizeRange(0, size), MatrixCopyOp(/*from=*/other, /*to=*/*this));

    // Copy the row size array from the other matrix to this matrix.
    tbb::parallel_for(SizeRange(0, mNumRows),
        internal::CopyOp<SizeType>(/*from=*/other.mRowSizeArray.get(), /*to=*/mRowSizeArray.get()));
}


template<typename ValueType, SizeType STENCIL_SIZE>
inline void
SparseStencilMatrix<ValueType, STENCIL_SIZE>::setValue(SizeType row, SizeType col,
    const ValueType& val)
{
    assert(row < mNumRows);
    this->getRowEditor(row).setValue(col, val);
}


template<typename ValueType, SizeType STENCIL_SIZE>
inline const ValueType&
SparseStencilMatrix<ValueType, STENCIL_SIZE>::getValue(SizeType row, SizeType col) const
{
    assert(row < mNumRows);
    return this->getConstRow(row).getValue(col);
}


template<typename ValueType, SizeType STENCIL_SIZE>
inline const ValueType&
SparseStencilMatrix<ValueType, STENCIL_SIZE>::operator()(SizeType row, SizeType col) const
{
    return this->getValue(row,col);
}


template<typename ValueType, SizeType STENCIL_SIZE>
template<typename Scalar>
struct SparseStencilMatrix<ValueType, STENCIL_SIZE>::RowScaleOp
{
    RowScaleOp(SparseStencilMatrix& m, const Scalar& s_): mat(&m), s(s_) {}

    void operator()(const SizeRange& range) const
    {
        for (SizeType n = range.begin(), N = range.end(); n < N; ++n) {
            RowEditor row = mat->getRowEditor(n);
            row.scale(s);
        }
    }

    SparseStencilMatrix* mat;
    const Scalar s;
};


template<typename ValueType, SizeType STENCIL_SIZE>
template<typename Scalar>
inline void
SparseStencilMatrix<ValueType, STENCIL_SIZE>::scale(const Scalar& s)
{
    // Parallelize over the rows in the matrix.
    tbb::parallel_for(SizeRange(0, mNumRows), RowScaleOp<Scalar>(*this, s));
}


template<typename ValueType, SizeType STENCIL_SIZE>
template<typename VecValueType>
struct SparseStencilMatrix<ValueType, STENCIL_SIZE>::VecMultOp
{
    VecMultOp(const SparseStencilMatrix& m, const VecValueType* i, VecValueType* o):
        mat(&m), in(i), out(o) {}

    void operator()(const SizeRange& range) const
    {
        for (SizeType n = range.begin(), N = range.end(); n < N; ++n) {
            ConstRow row = mat->getConstRow(n);
            out[n] = row.dot(in, mat->numRows());
        }
    }

    const SparseStencilMatrix* mat;
    const VecValueType* in;
    VecValueType* out;
};


template<typename ValueType, SizeType STENCIL_SIZE>
template<typename VecValueType>
inline void
SparseStencilMatrix<ValueType, STENCIL_SIZE>::vectorMultiply(
    const Vector<VecValueType>& inVec, Vector<VecValueType>& resultVec) const
{
    if (inVec.size() != mNumRows) {
        OPENVDB_THROW(ArithmeticError, "matrix and input vector have incompatible sizes ("
            << mNumRows << "x" << mNumRows << " vs. " << inVec.size() << ")");
    }
    if (resultVec.size() != mNumRows) {
        OPENVDB_THROW(ArithmeticError, "matrix and result vector have incompatible sizes ("
            << mNumRows << "x" << mNumRows << " vs. " << resultVec.size() << ")");
    }

    vectorMultiply(inVec.data(), resultVec.data());
}


template<typename ValueType, SizeType STENCIL_SIZE>
template<typename VecValueType>
inline void
SparseStencilMatrix<ValueType, STENCIL_SIZE>::vectorMultiply(
    const VecValueType* inVec, VecValueType* resultVec) const
{
    // Parallelize over the rows in the matrix.
    tbb::parallel_for(SizeRange(0, mNumRows),
        VecMultOp<VecValueType>(*this, inVec, resultVec));
}


template<typename ValueType, SizeType STENCIL_SIZE>
template<typename OtherValueType>
struct SparseStencilMatrix<ValueType, STENCIL_SIZE>::EqOp
{
    EqOp(const SparseStencilMatrix& a_,
        const SparseStencilMatrix<OtherValueType, STENCIL_SIZE>& b_, ValueType e):
        a(&a_), b(&b_), eps(e) {}

    bool operator()(const SizeRange& range, bool equal) const
    {
        if (equal) {
            for (SizeType n = range.begin(), N = range.end(); n < N; ++n) {
                if (!a->getConstRow(n).eq(b->getConstRow(n), eps)) return false;
            }
        }
        return equal;
    }

    const SparseStencilMatrix* a;
    const SparseStencilMatrix<OtherValueType, STENCIL_SIZE>* b;
    const ValueType eps;
};


template<typename ValueType, SizeType STENCIL_SIZE>
template<typename OtherValueType>
inline bool
SparseStencilMatrix<ValueType, STENCIL_SIZE>::eq(
    const SparseStencilMatrix<OtherValueType, STENCIL_SIZE>& other, ValueType eps) const
{
    if (this->numRows() != other.numRows()) return false;
    bool equal = tbb::parallel_reduce(SizeRange(0, this->numRows()), /*seed=*/true,
        EqOp<OtherValueType>(*this, other, eps),
        /*join=*/[](bool eq1, bool eq2) { return (eq1 && eq2); });
    return equal;
}


template<typename ValueType, SizeType STENCIL_SIZE>
struct SparseStencilMatrix<ValueType, STENCIL_SIZE>::IsFiniteOp
{
    IsFiniteOp(const SparseStencilMatrix& m): mat(&m) {}

    bool operator()(const SizeRange& range, bool finite) const
    {
        if (finite) {
            for (SizeType n = range.begin(), N = range.end(); n < N; ++n) {
                const ConstRow row = mat->getConstRow(n);
                for (ConstValueIter it = row.cbegin(); it; ++it) {
                    if (!std::isfinite(*it)) return false;
                }
            }
        }
        return finite;
    }

    const SparseStencilMatrix* mat;
};


template<typename ValueType, SizeType STENCIL_SIZE>
inline bool
SparseStencilMatrix<ValueType, STENCIL_SIZE>::isFinite() const
{
    // Parallelize over the rows of this matrix.
    bool finite = tbb::parallel_reduce(SizeRange(0, this->numRows()), /*seed=*/true,
        IsFiniteOp(*this), /*join=*/[](bool finite1, bool finite2) { return (finite1&&finite2); });
    return finite;
}


template<typename ValueType, SizeType STENCIL_SIZE>
inline std::string
SparseStencilMatrix<ValueType, STENCIL_SIZE>::str() const
{
    std::ostringstream ostr;
    for (SizeType n = 0, N = this->size(); n < N; ++n) {
        ostr << n << ": " << this->getConstRow(n).str() << "\n";
    }
    return ostr.str();
}


template<typename ValueType, SizeType STENCIL_SIZE>
inline typename SparseStencilMatrix<ValueType, STENCIL_SIZE>::RowEditor
SparseStencilMatrix<ValueType, STENCIL_SIZE>::getRowEditor(SizeType i)
{
    assert(i < mNumRows);
    const SizeType head = i * STENCIL_SIZE;
    return RowEditor(&mValueArray[head], &mColumnIdxArray[head], mRowSizeArray[i], mNumRows);
}


template<typename ValueType, SizeType STENCIL_SIZE>
inline typename SparseStencilMatrix<ValueType, STENCIL_SIZE>::ConstRow
SparseStencilMatrix<ValueType, STENCIL_SIZE>::getConstRow(SizeType i) const
{
    assert(i < mNumRows);
    const SizeType head = i * STENCIL_SIZE; // index for this row into main storage
    return ConstRow(&mValueArray[head], &mColumnIdxArray[head], mRowSizeArray[i]);
}


template<typename ValueType, SizeType STENCIL_SIZE>
template<typename DataType>
inline SizeType
SparseStencilMatrix<ValueType, STENCIL_SIZE>::RowBase<DataType>::find(SizeType columnIdx) const
{
    if (this->empty()) return mData.mSize;

    // Get a pointer to the first column index that is equal to or greater than the given index.
    // (This assumes that the data is sorted by column.)
    const SizeType* colPtr = std::lower_bound(mData.mCols, mData.mCols + mData.mSize, columnIdx);
    // Return the offset of the pointer from the beginning of the array.
    return static_cast<SizeType>(colPtr - mData.mCols);
}


template<typename ValueType, SizeType STENCIL_SIZE>
template<typename DataType>
inline const ValueType&
SparseStencilMatrix<ValueType, STENCIL_SIZE>::RowBase<DataType>::getValue(
    SizeType columnIdx, bool& active) const
{
    active = false;
    SizeType idx = this->find(columnIdx);
    if (idx < this->size() && this->column(idx) == columnIdx) {
        active = true;
        return this->value(idx);
    }
    return SparseStencilMatrix::sZeroValue;
}

template<typename ValueType, SizeType STENCIL_SIZE>
template<typename DataType>
inline const ValueType&
SparseStencilMatrix<ValueType, STENCIL_SIZE>::RowBase<DataType>::getValue(SizeType columnIdx) const
{
    SizeType idx = this->find(columnIdx);
    if (idx < this->size() && this->column(idx) == columnIdx) {
        return this->value(idx);
    }
    return SparseStencilMatrix::sZeroValue;
}


template<typename ValueType, SizeType STENCIL_SIZE>
template<typename DataType>
inline typename SparseStencilMatrix<ValueType, STENCIL_SIZE>::ConstValueIter
SparseStencilMatrix<ValueType, STENCIL_SIZE>::RowBase<DataType>::cbegin() const
{
    return ConstValueIter(mData);
}


template<typename ValueType, SizeType STENCIL_SIZE>
template<typename DataType>
template<typename OtherDataType>
inline bool
SparseStencilMatrix<ValueType, STENCIL_SIZE>::RowBase<DataType>::eq(
    const RowBase<OtherDataType>& other, ValueType eps) const
{
    if (this->size() != other.size()) return false;
    for (ConstValueIter it = cbegin(), oit = other.cbegin(); it || oit; ++it, ++oit) {
        if (it.column() != oit.column()) return false;
        if (!isApproxEqual(*it, *oit, eps)) return false;
    }
    return true;
}


template<typename ValueType, SizeType STENCIL_SIZE>
template<typename DataType>
template<typename VecValueType>
inline VecValueType
SparseStencilMatrix<ValueType, STENCIL_SIZE>::RowBase<DataType>::dot(
    const VecValueType* inVec, SizeType vecSize) const
{
    VecValueType result = zeroVal<VecValueType>();
    for (SizeType idx = 0, N = std::min(vecSize, this->size()); idx < N; ++idx) {
        result += static_cast<VecValueType>(this->value(idx) * inVec[this->column(idx)]);
    }
    return result;
}

template<typename ValueType, SizeType STENCIL_SIZE>
template<typename DataType>
template<typename VecValueType>
inline VecValueType
SparseStencilMatrix<ValueType, STENCIL_SIZE>::RowBase<DataType>::dot(
    const Vector<VecValueType>& inVec) const
{
    return dot(inVec.data(), inVec.size());
}


template<typename ValueType, SizeType STENCIL_SIZE>
template<typename DataType>
inline std::string
SparseStencilMatrix<ValueType, STENCIL_SIZE>::RowBase<DataType>::str() const
{
    std::ostringstream ostr;
    std::string sep;
    for (SizeType n = 0, N = this->size(); n < N; ++n) {
        ostr << sep << "(" << this->column(n) << ", " << this->value(n) << ")";
        sep = ", ";
    }
    return ostr.str();
}


template<typename ValueType, SizeType STENCIL_SIZE>
inline
SparseStencilMatrix<ValueType, STENCIL_SIZE>::ConstRow::ConstRow(
    const ValueType* valueHead, const SizeType* columnHead, const SizeType& rowSize):
    ConstRowBase(ConstRowData(const_cast<ValueType*>(valueHead),
        const_cast<SizeType*>(columnHead), const_cast<SizeType&>(rowSize)))
{
}


template<typename ValueType, SizeType STENCIL_SIZE>
inline
SparseStencilMatrix<ValueType, STENCIL_SIZE>::RowEditor::RowEditor(
    ValueType* valueHead, SizeType* columnHead, SizeType& rowSize, SizeType colSize):
    RowBase<>(RowData(valueHead, columnHead, rowSize)), mNumColumns(colSize)
{
}


template<typename ValueType, SizeType STENCIL_SIZE>
inline void
SparseStencilMatrix<ValueType, STENCIL_SIZE>::RowEditor::clear()
{
    // Note: since mSize is a reference, this modifies the underlying matrix.
    RowBase<>::mData.mSize = 0;
}


template<typename ValueType, SizeType STENCIL_SIZE>
inline SizeType
SparseStencilMatrix<ValueType, STENCIL_SIZE>::RowEditor::setValue(
    SizeType column, const ValueType& value)
{
    assert(column < mNumColumns);

    RowData& data = RowBase<>::mData;

    // Get the offset of the first column index that is equal to or greater than
    // the column to be modified.
    SizeType offset = this->find(column);

    if (offset < data.mSize && data.mCols[offset] == column) {
        // If the column already exists, just update its value.
        data.mVals[offset] = value;
        return data.mSize;
    }

    // Check that it is safe to add a new column.
    assert(data.mSize < this->capacity());

    if (offset >= data.mSize) {
        // The new column's index is larger than any existing index.  Append the new column.
        data.mVals[data.mSize] = value;
        data.mCols[data.mSize] = column;
    } else {
        // Insert the new column at the computed offset after shifting subsequent columns.
        for (SizeType i = data.mSize; i > offset; --i) {
            data.mVals[i] = data.mVals[i - 1];
            data.mCols[i] = data.mCols[i - 1];
        }
        data.mVals[offset] = value;
        data.mCols[offset] = column;
    }
    ++data.mSize;

    return data.mSize;
}


template<typename ValueType, SizeType STENCIL_SIZE>
template<typename Scalar>
inline void
SparseStencilMatrix<ValueType, STENCIL_SIZE>::RowEditor::scale(const Scalar& s)
{
    for (int idx = 0, N = this->size(); idx < N; ++idx) {
        RowBase<>::mData.mVals[idx] *= s;
    }
}


////////////////////////////////////////


/// Diagonal preconditioner
template<typename MatrixType>
class JacobiPreconditioner: public Preconditioner<typename MatrixType::ValueType>
{
private:
    struct InitOp;
    struct ApplyOp;

public:
    using ValueType = typename MatrixType::ValueType;
    using BaseType = Preconditioner<ValueType>;
    using VectorType = Vector<ValueType>;
    using Ptr = SharedPtr<JacobiPreconditioner>;

    JacobiPreconditioner(const MatrixType& A): BaseType(A), mDiag(A.numRows())
    {
        // Initialize vector mDiag with the values from the matrix diagonal.
        tbb::parallel_for(SizeRange(0, A.numRows()), InitOp(A, mDiag.data()));
    }

    ~JacobiPreconditioner() override = default;

    void apply(const Vector<ValueType>& r, Vector<ValueType>& z) override
    {
        const SizeType size = mDiag.size();

        assert(r.size() == z.size());
        assert(r.size() == size);

        tbb::parallel_for(SizeRange(0, size), ApplyOp(mDiag.data(), r.data(), z.data()));
    }

    /// Return @c true if all values along the diagonal are finite.
    bool isFinite() const { return mDiag.isFinite(); }

private:
    // Functor for use with tbb::parallel_for()
    struct InitOp
    {
        InitOp(const MatrixType& m, ValueType* v): mat(&m), vec(v) {}
        void operator()(const SizeRange& range) const {
            for (SizeType n = range.begin(), N = range.end(); n < N; ++n) {
                const ValueType val = mat->getValue(n, n);
                assert(!isApproxZero(val, ValueType(0.0001)));
                vec[n] = static_cast<ValueType>(1.0 / val);
            }
        }
        const MatrixType* mat; ValueType* vec;
    };

    // Functor for use with tbb::parallel_reduce()
    struct ApplyOp
    {
        ApplyOp(const ValueType* x_, const ValueType* y_, ValueType* out_):
            x(x_), y(y_), out(out_) {}
        void operator()(const SizeRange& range) const {
            for (SizeType n = range.begin(), N = range.end(); n < N; ++n) out[n] = x[n] * y[n];
        }
        const ValueType *x, *y; ValueType* out;
    };

    // The Jacobi preconditioner is a diagonal matrix
    VectorType mDiag;
}; // class JacobiPreconditioner


/// Preconditioner using incomplete Cholesky factorization
template<typename MatrixType>
class IncompleteCholeskyPreconditioner: public Preconditioner<typename MatrixType::ValueType>
{
private:
    struct CopyToLowerOp;
    struct TransposeOp;

public:
    using ValueType = typename MatrixType::ValueType;
    using BaseType = Preconditioner<ValueType>;
    using VectorType = Vector<ValueType>;
    using Ptr = SharedPtr<IncompleteCholeskyPreconditioner>;
    using TriangularMatrix = SparseStencilMatrix<ValueType, 4>;
    using TriangleConstRow = typename TriangularMatrix::ConstRow;
    using TriangleRowEditor = typename TriangularMatrix::RowEditor;

    IncompleteCholeskyPreconditioner(const MatrixType& matrix)
        : BaseType(matrix)
        , mLowerTriangular(matrix.numRows())
        , mUpperTriangular(matrix.numRows())
        , mTempVec(matrix.numRows())
    {
        // Size of matrix
        const SizeType numRows = mLowerTriangular.numRows();

        // Copy the upper triangular part to the lower triangular part.
        tbb::parallel_for(SizeRange(0, numRows), CopyToLowerOp(matrix, mLowerTriangular));

        // Build the Incomplete Cholesky Matrix
        //
        // Algorithm:
        //
        // for (k = 0; k < size; ++k) {
        //     A(k,k) = sqrt(A(k,k));
        //     for (i = k +1, i < size; ++i) {
        //         if (A(i,k) == 0) continue;
        //         A(i,k) = A(i,k) / A(k,k);
        //     }
        //     for (j = k+1; j < size; ++j) {
        //         for (i = j; i < size; ++i) {
        //             if (A(i,j) == 0) continue;
        //             A(i,j) -= A(i,k)*A(j,k);
        //         }
        //     }
        // }

        mPassedCompatibilityCondition = true;

        for (SizeType k = 0; k < numRows; ++k) {

            TriangleConstRow crow_k = mLowerTriangular.getConstRow(k);
            ValueType diagonalValue = crow_k.getValue(k);

            // Test if the matrix build has failed.
            if (diagonalValue < 1.e-5) {
                mPassedCompatibilityCondition = false;
                break;
            }

            diagonalValue = Sqrt(diagonalValue);

            TriangleRowEditor row_k = mLowerTriangular.getRowEditor(k);
            row_k.setValue(k, diagonalValue);

            // Exploit the fact that the matrix is symmetric.
            typename MatrixType::ConstRow srcRow = matrix.getConstRow(k);
            typename MatrixType::ConstValueIter citer = srcRow.cbegin();
            for ( ; citer; ++citer) {
                SizeType ii = citer.column();
                if (ii < k+1) continue; // look above diagonal

                TriangleRowEditor row_ii = mLowerTriangular.getRowEditor(ii);

                row_ii.setValue(k, *citer / diagonalValue);
            }

            // for (j = k+1; j < size; ++j) replaced by row iter below
            citer.reset(); // k,j entries
            for ( ; citer; ++citer) {
                SizeType j = citer.column();
                if (j < k+1) continue;

                TriangleConstRow row_j = mLowerTriangular.getConstRow(j);
                ValueType a_jk = row_j.getValue(k);  // a_jk is non zero if a_kj is non zero

                // Entry (i,j) is non-zero if matrix(j,i) is nonzero

                typename MatrixType::ConstRow mask = matrix.getConstRow(j);
                typename MatrixType::ConstValueIter maskIter = mask.cbegin();
                for ( ; maskIter; ++maskIter) {
                    SizeType i = maskIter.column();
                    if (i < j) continue;

                    TriangleConstRow crow_i = mLowerTriangular.getConstRow(i);
                    ValueType a_ij = crow_i.getValue(j);
                    ValueType a_ik = crow_i.getValue(k);
                    TriangleRowEditor row_i = mLowerTriangular.getRowEditor(i);
                    a_ij -= a_ik * a_jk;

                    row_i.setValue(j, a_ij);
                }
            }
        }

        // Build the transpose of the IC matrix: mUpperTriangular
        tbb::parallel_for(SizeRange(0, numRows),
            TransposeOp(matrix, mLowerTriangular, mUpperTriangular));
    }

    ~IncompleteCholeskyPreconditioner() override = default;

    bool isValid() const override { return mPassedCompatibilityCondition; }

    void apply(const Vector<ValueType>& rVec, Vector<ValueType>& zVec) override
    {
        if (!mPassedCompatibilityCondition) {
            OPENVDB_THROW(ArithmeticError, "invalid Cholesky decomposition");
        }

        // Solve mUpperTriangular * mLowerTriangular * rVec = zVec;

        SizeType size = mLowerTriangular.numRows();

        zVec.fill(zeroVal<ValueType>());
        ValueType* zData = zVec.data();

        if (size == 0) return;

        assert(rVec.size() == size);
        assert(zVec.size() == size);

        // Allocate a temp vector
        mTempVec.fill(zeroVal<ValueType>());
        ValueType* tmpData = mTempVec.data();
        const ValueType* rData = rVec.data();

        // Solve mLowerTriangular * tmp = rVec;
        for (SizeType i = 0; i < size; ++i) {
            typename TriangularMatrix::ConstRow row = mLowerTriangular.getConstRow(i);
            ValueType diagonal = row.getValue(i);
            ValueType dot = row.dot(mTempVec);
            tmpData[i] = (rData[i] - dot) / diagonal;
            if (!std::isfinite(tmpData[i])) {
                OPENVDB_LOG_DEBUG_RUNTIME("1 diagonal was " << diagonal);
                OPENVDB_LOG_DEBUG_RUNTIME("1a diagonal " << row.getValue(i));
            }
        }

        // Solve mUpperTriangular * zVec = tmp;
        for (SizeType ii = 0; ii < size; ++ii) {
            SizeType i = size - 1 - ii;
            typename TriangularMatrix::ConstRow row = mUpperTriangular.getConstRow(i);
            ValueType diagonal = row.getValue(i);
            ValueType dot = row.dot(zVec);
            zData[i] = (tmpData[i] - dot) / diagonal;
            if (!std::isfinite(zData[i])) {
                OPENVDB_LOG_DEBUG_RUNTIME("2 diagonal was " << diagonal);
            }
        }
    }

    const TriangularMatrix& lowerMatrix() const { return mLowerTriangular; }
    const TriangularMatrix& upperMatrix() const { return mUpperTriangular; }

private:
    // Functor for use with tbb::parallel_for()
    struct CopyToLowerOp
    {
        CopyToLowerOp(const MatrixType& m, TriangularMatrix& l): mat(&m), lower(&l) {}
        void operator()(const SizeRange& range) const {
            for (SizeType n = range.begin(), N = range.end(); n < N; ++n) {
                typename TriangularMatrix::RowEditor outRow = lower->getRowEditor(n);
                outRow.clear();
                typename MatrixType::ConstRow inRow = mat->getConstRow(n);
                for (typename MatrixType::ConstValueIter it = inRow.cbegin(); it; ++it) {
                    if (it.column() > n) continue; // skip above diagonal
                    outRow.setValue(it.column(), *it);
                }
            }
        }
        const MatrixType* mat; TriangularMatrix* lower;
    };

    // Functor for use with tbb::parallel_for()
    struct TransposeOp
    {
        TransposeOp(const MatrixType& m, const TriangularMatrix& l, TriangularMatrix& u):
            mat(&m), lower(&l), upper(&u) {}
        void operator()(const SizeRange& range) const {
            for (SizeType n = range.begin(), N = range.end(); n < N; ++n) {
                typename TriangularMatrix::RowEditor outRow = upper->getRowEditor(n);
                outRow.clear();
                // Use the fact that matrix is symmetric.
                typename MatrixType::ConstRow inRow = mat->getConstRow(n);
                for (typename MatrixType::ConstValueIter it = inRow.cbegin(); it; ++it) {
                    const SizeType column = it.column();
                    if (column < n) continue; // only set upper triangle
                    outRow.setValue(column, lower->getValue(column, n));
                }
            }
        }
        const MatrixType* mat; const TriangularMatrix* lower; TriangularMatrix* upper;
    };

    TriangularMatrix  mLowerTriangular;
    TriangularMatrix  mUpperTriangular;
    Vector<ValueType> mTempVec;
    bool              mPassedCompatibilityCondition;
}; // class IncompleteCholeskyPreconditioner


////////////////////////////////////////


namespace internal {

/// Compute @e ax + @e y.
template<typename T>
inline void
axpy(const T& a, const T* xVec, const T* yVec, T* resultVec, SizeType size)
{
    tbb::parallel_for(SizeRange(0, size), LinearOp<T>(a, xVec, yVec, resultVec));
}

/// Compute @e ax + @e y.
template<typename T>
inline void
axpy(const T& a, const Vector<T>& xVec, const Vector<T>& yVec, Vector<T>& result)
{
    assert(xVec.size() == yVec.size());
    assert(xVec.size() == result.size());
    axpy(a, xVec.data(), yVec.data(), result.data(), xVec.size());
}


/// Compute @e r = @e b &minus; @e Ax.
template<typename MatrixOperator, typename VecValueType>
inline void
computeResidual(const MatrixOperator& A, const VecValueType* x,
    const VecValueType* b, VecValueType* r)
{
    // Compute r = A * x.
    A.vectorMultiply(x, r);
    // Compute r = b - r.
    tbb::parallel_for(SizeRange(0, A.numRows()), LinearOp<VecValueType>(-1.0, r, b, r));
}

/// Compute @e r = @e b &minus; @e Ax.
template<typename MatrixOperator, typename T>
inline void
computeResidual(const MatrixOperator& A, const Vector<T>& x, const Vector<T>& b, Vector<T>& r)
{
    assert(x.size() == b.size());
    assert(x.size() == r.size());
    assert(x.size() == A.numRows());

    computeResidual(A, x.data(), b.data(), r.data());
}

} // namespace internal


////////////////////////////////////////


template<typename PositiveDefMatrix>
inline State
solve(
    const PositiveDefMatrix& Amat,
    const Vector<typename PositiveDefMatrix::ValueType>& bVec,
    Vector<typename PositiveDefMatrix::ValueType>& xVec,
    Preconditioner<typename PositiveDefMatrix::ValueType>& precond,
    const State& termination)
{
    util::NullInterrupter interrupter;
    return solve(Amat, bVec, xVec, precond, interrupter, termination);
}


template<typename PositiveDefMatrix, typename Interrupter>
inline State
solve(
    const PositiveDefMatrix& Amat,
    const Vector<typename PositiveDefMatrix::ValueType>& bVec,
    Vector<typename PositiveDefMatrix::ValueType>& xVec,
    Preconditioner<typename PositiveDefMatrix::ValueType>& precond,
    Interrupter& interrupter,
    const State& termination)
{
    using ValueType = typename PositiveDefMatrix::ValueType;
    using VectorType = Vector<ValueType>;

    State result;
    result.success = false;
    result.iterations = 0;
    result.relativeError = 0.0;
    result.absoluteError = 0.0;

    const SizeType size = Amat.numRows();
    if (size == 0) {
        OPENVDB_LOG_WARN("pcg::solve(): matrix has dimension zero");
        return result;
    }
    if (size != bVec.size()) {
        OPENVDB_THROW(ArithmeticError, "A and b have incompatible sizes"
            << size << "x" << size << " vs. " << bVec.size() << ")");
    }
    if (size != xVec.size()) {
        OPENVDB_THROW(ArithmeticError, "A and x have incompatible sizes"
            << size << "x" << size << " vs. " << xVec.size() << ")");
    }

    // Temp vectors
    VectorType zVec(size); // transformed residual (M^-1 r)
    VectorType pVec(size); // search direction
    VectorType qVec(size); // A * p

    // Compute norm of B (the source)
    const ValueType tmp = bVec.infNorm();
    const ValueType infNormOfB = isZero(tmp) ? ValueType(1) : tmp;

    // Compute rVec: residual = b - Ax.
    VectorType rVec(size); // vector of residuals

    internal::computeResidual(Amat, xVec, bVec, rVec);

    assert(rVec.isFinite());

    // Normalize the residual norm with the source norm and look for early out.
    result.absoluteError = static_cast<double>(rVec.infNorm());
    result.relativeError = static_cast<double>(result.absoluteError / infNormOfB);
    if (result.relativeError <= termination.relativeError) {
        result.success = true;
        return result;
    }

    // Iterations of the CG solve

    ValueType rDotZPrev(1); // inner product of <z,r>

    // Keep track of the minimum error to monitor convergence.
    ValueType minL2Error = std::numeric_limits<ValueType>::max();
    ValueType l2Error;

    int iteration = 0;
    for ( ; iteration < termination.iterations; ++iteration) {

        if (interrupter.wasInterrupted()) {
            OPENVDB_THROW(RuntimeError, "conjugate gradient solver was interrupted");
        }

        OPENVDB_LOG_DEBUG_RUNTIME("pcg::solve() " << result);

        result.iterations = iteration + 1;

        // Apply preconditioner to residual
        // z_{k} = M^-1 r_{k}
        precond.apply(rVec, zVec);

        // <r,z>
        const ValueType rDotZ = rVec.dot(zVec);
        assert(std::isfinite(rDotZ));

        if (0 == iteration) {
            // Initialize
            pVec = zVec;
        } else {
            const ValueType beta = rDotZ / rDotZPrev;
            // p = beta * p + z
            internal::axpy(beta, pVec, zVec, /*result */pVec);
        }

        // q_{k} = A p_{k}
        Amat.vectorMultiply(pVec, qVec);

        // alpha = <r_{k-1}, z_{k-1}> / <p_{k},q_{k}>
        const ValueType pAp = pVec.dot(qVec);
        assert(std::isfinite(pAp));

        const ValueType alpha = rDotZ / pAp;
        rDotZPrev = rDotZ;

        // x_{k} = x_{k-1} + alpha * p_{k}
        internal::axpy(alpha, pVec, xVec, /*result=*/xVec);

        // r_{k} = r_{k-1} - alpha_{k-1} A p_{k}
        internal::axpy(-alpha, qVec, rVec, /*result=*/rVec);

        // update tolerances
        l2Error = rVec.l2Norm();
        minL2Error = Min(l2Error, minL2Error);

        result.absoluteError = static_cast<double>(rVec.infNorm());
        result.relativeError = static_cast<double>(result.absoluteError / infNormOfB);

        if (l2Error > 2 * minL2Error) {
            // The solution started to diverge.
            result.success = false;
            break;
        }
        if (!std::isfinite(result.absoluteError)) {
            // Total divergence of solution
            result.success = false;
            break;
        }
        if (result.absoluteError <= termination.absoluteError) {
            // Convergence
            result.success = true;
            break;
        }
        if (result.relativeError <= termination.relativeError) {
            // Convergence
            result.success = true;
            break;
        }
    }
    OPENVDB_LOG_DEBUG_RUNTIME("pcg::solve() " << result);

    return result;
}

} // namespace pcg
} // namespace math
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_MATH_CONJGRADIENT_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2018 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
