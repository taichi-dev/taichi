/**
Copyright (c) 2016 Theodore Gast, Chuyuan Fu, Chenfanfu Jiang, Joseph Teran

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

If the code is used in an article, the following paper shall be cited:
@techreport{qrsvd:2016,
  title={Implicit-shifted Symmetric QR Singular Value Decomposition of 3x3 Matrices},
  author={Gast, Theodore and Fu, Chuyuan and Jiang, Chenfanfu and Teran, Joseph},
  year={2016},
  institution={University of California Los Angeles}
}

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

################################################################################
This file provides a random number generator and a timer.
Sample usage:
    RandomNumber<float> rand;
    float x = randReal(-0.5, 0.8);

    Timer timer;
    timer.start();
    SOME CODE A
    std::cout<<"CODE A took "<<timer.click()<<" seconds"<<std::endl;
    SOME CODE B
    std::cout<<"CODE B took "<<timer.click()<<" seconds"<<std::endl;
################################################################################
*/

#ifndef JIXIE_SVD_TOOLS_H
#define JIXIE_SVD_TOOLS_H

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/SVD>
#pragma GCC diagnostic pop

#include <mmintrin.h>
#include <xmmintrin.h>
#include <cmath>
#include <random>
#include <chrono>
#include <iostream>
#include <iomanip>

namespace JIXIE {

/**
    Random number generator.
*/
template <class T>
class RandomNumber {
public:
    std::mt19937 generator;

    RandomNumber(unsigned s = 123)
        : generator(s)
    {
    }

    ~RandomNumber()
    {
    }

    /**
       Random real number from an interval
    */
    T randReal(T a, T b)
    {
        std::uniform_real_distribution<T> distribution(a, b);
        return distribution(generator);
    }

    /**
      Fill with uniform random numbers
    */
    template <class Derived>
    void fill(Eigen::DenseBase<Derived>& x, T a, T b)
    {
        for (typename Derived::Index i = 0; i < x.size(); i++)
            x(i) = randReal(a, b);
    }
};

namespace MATH_TOOLS {

/**
\brief Approximate inverse square root

A fast approximation to the inverse sqrt
The relative error is less than  1.5*2^-12
*/
inline float approx_rsqrt(float a)
{
return _mm_cvtss_f32(_mm_rsqrt_ss(_mm_set_ss(a)));
}

/**
\brief Inverse square root
computed from approx_rsqrt and one newton step
*/
inline float rsqrt(float a)
{
float b = approx_rsqrt(a);
// Newton step with f(x) = a - 1/x^2
b = 0.5f * b * (3.0f - a * (b * b));
return b;
}

/**
\brief Inverse square root
computed from 1/std::sqrt
*/
inline double rsqrt(double a)
{
using std::sqrt;
return 1 / sqrt(a);
}
}

/**
    Timer. We can use either system timer or stready timer
*/
class Timer {
public:
    Timer() {}

    ~Timer() {}

    /**
 	  \brief Start timing
	*/
    void start()
    {
        start_time = std::chrono::steady_clock::now();
    }

    /**
  	 \return time elapsed since last click in seconds
*/
    double click()
    {
        to_time = std::chrono::steady_clock::now();
        elapsed_seconds = to_time - start_time;
        start_time = to_time;
        return elapsed_seconds.count();
    }

private:
    std::chrono::time_point<std::chrono::steady_clock> start_time;
    std::chrono::time_point<std::chrono::steady_clock> to_time;
    std::chrono::duration<double> elapsed_seconds;
};

namespace INTERNAL {
using namespace std;
template <class T, class Enable = void>
struct ScalarTypeHelper {
    using type = typename T::Scalar;
};
template <class T>
struct ScalarTypeHelper<T, enable_if_t<is_arithmetic<T>::value> > {
    using type = T;
};
}

template <class T>
using ScalarType = typename INTERNAL::ScalarTypeHelper<T>::type;

template <class MatrixType>
constexpr bool isSize(int m, int n)
{
    return MatrixType::RowsAtCompileTime == m && MatrixType::ColsAtCompileTime == n;
}

}
#endif
