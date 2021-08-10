
#ifndef EIGEN_BENCH_BASICBENCH_H
#define EIGEN_BENCH_BASICBENCH_H

enum {LazyEval, EarlyEval, OmpEval};

template<int Mode, typename MatrixType>
void benchBasic_loop(const MatrixType& I, MatrixType& m, int iterations) __attribute__((noinline));

template<int Mode, typename MatrixType>
void benchBasic_loop(const MatrixType& I, MatrixType& m, int iterations)
{
  for(int a = 0; a < iterations; a++)
  {
    if (Mode==LazyEval)
    {
      asm("#begin_bench_loop LazyEval");
      if (MatrixType::SizeAtCompileTime!=Eigen::Dynamic) asm("#fixedsize");
      m = (I + 0.00005 * (m + m.lazy() * m)).eval();
    }
    else if (Mode==OmpEval)
    {
      asm("#begin_bench_loop OmpEval");
      if (MatrixType::SizeAtCompileTime!=Eigen::Dynamic) asm("#fixedsize");
      m = (I + 0.00005 * (m + m.lazy() * m)).evalOMP();
    }
    else
    {
      asm("#begin_bench_loop EarlyEval");
      if (MatrixType::SizeAtCompileTime!=Eigen::Dynamic) asm("#fixedsize");
      m = I + 0.00005 * (m + m * m);
    }
    asm("#end_bench_loop");
  }
}

template<int Mode, typename MatrixType>
double benchBasic(const MatrixType& mat, int size, int tries) __attribute__((noinline));

template<int Mode, typename MatrixType>
double benchBasic(const MatrixType& mat, int iterations, int tries)
{
  const int rows = mat.rows();
  const int cols = mat.cols();

  MatrixType I(rows,cols);
  MatrixType m(rows,cols);

  initMatrix_identity(I);

  Eigen::BenchTimer timer;
  for(uint t=0; t<tries; ++t)
  {
    initMatrix_random(m);
    timer.start();
    benchBasic_loop<Mode>(I, m, iterations);
    timer.stop();
    cerr << m;
  }
  return timer.value();
};

#endif // EIGEN_BENCH_BASICBENCH_H
