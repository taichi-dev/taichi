
#include <iostream>
#include <Eigen/Core>
#include <bench/BenchUtil.h>
using namespace Eigen;

#ifndef REPEAT
#define REPEAT 100000
#endif

#ifndef TRIES
#define TRIES 20
#endif

typedef double Scalar;

template <typename MatrixType>
__attribute__ ((noinline)) void bench_reverse(const MatrixType& m)
{
  int rows = m.rows();
  int cols = m.cols();
  int size = m.size();

  int repeats = (REPEAT*1000)/size;
  MatrixType a = MatrixType::Random(rows,cols);
  MatrixType b = MatrixType::Random(rows,cols);

  BenchTimer timerB, timerH, timerV;

  Scalar acc = 0;
  int r = internal::random<int>(0,rows-1);
  int c = internal::random<int>(0,cols-1);
  for (int t=0; t<TRIES; ++t)
  {
    timerB.start();
    for (int k=0; k<repeats; ++k)
    {
      asm("#begin foo");
      b = a.reverse();
      asm("#end foo");
      acc += b.coeff(r,c);
    }
    timerB.stop();
  }

  if (MatrixType::RowsAtCompileTime==Dynamic)
    std::cout << "dyn   ";
  else
    std::cout << "fixed ";
  std::cout << rows << " x " << cols << " \t"
            << (timerB.value() * REPEAT) / repeats << "s "
            << "(" << 1e-6 * size*repeats/timerB.value() << " MFLOPS)\t";

  std::cout << "\n";
  // make sure the compiler does not optimize too much
  if (acc==123)
    std::cout << acc;
}

int main(int argc, char* argv[])
{
  const int dynsizes[] = {4,6,8,16,24,32,49,64,128,256,512,900,0};
  std::cout << "size            no sqrt                           standard";
//   #ifdef BENCH_GSL
//   std::cout << "       GSL (standard + double + ATLAS)  ";
//   #endif
  std::cout << "\n";
  for (uint i=0; dynsizes[i]>0; ++i)
  {
    bench_reverse(Matrix<Scalar,Dynamic,Dynamic>(dynsizes[i],dynsizes[i]));
    bench_reverse(Matrix<Scalar,Dynamic,1>(dynsizes[i]*dynsizes[i]));
  }
//   bench_reverse(Matrix<Scalar,2,2>());
//   bench_reverse(Matrix<Scalar,3,3>());
//   bench_reverse(Matrix<Scalar,4,4>());
//   bench_reverse(Matrix<Scalar,5,5>());
//   bench_reverse(Matrix<Scalar,6,6>());
//   bench_reverse(Matrix<Scalar,7,7>());
//   bench_reverse(Matrix<Scalar,8,8>());
//   bench_reverse(Matrix<Scalar,12,12>());
//   bench_reverse(Matrix<Scalar,16,16>());
  return 0;
}

