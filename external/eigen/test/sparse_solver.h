// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2011 Gael Guennebaud <g.gael@free.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "sparse.h"
#include <Eigen/SparseCore>
#include <sstream>

template<typename Solver, typename Rhs, typename Guess,typename Result>
void solve_with_guess(IterativeSolverBase<Solver>& solver, const MatrixBase<Rhs>& b, const Guess& g, Result &x) {
  if(internal::random<bool>())
  {
    // With a temporary through evaluator<SolveWithGuess>
    x = solver.derived().solveWithGuess(b,g) + Result::Zero(x.rows(), x.cols());
  }
  else
  {
    // direct evaluation within x through Assignment<Result,SolveWithGuess>
    x = solver.derived().solveWithGuess(b.derived(),g);
  }
}

template<typename Solver, typename Rhs, typename Guess,typename Result>
void solve_with_guess(SparseSolverBase<Solver>& solver, const MatrixBase<Rhs>& b, const Guess& , Result& x) {
  if(internal::random<bool>())
    x = solver.derived().solve(b) + Result::Zero(x.rows(), x.cols());
  else
    x = solver.derived().solve(b);
}

template<typename Solver, typename Rhs, typename Guess,typename Result>
void solve_with_guess(SparseSolverBase<Solver>& solver, const SparseMatrixBase<Rhs>& b, const Guess& , Result& x) {
  x = solver.derived().solve(b);
}

template<typename Solver, typename Rhs, typename DenseMat, typename DenseRhs>
void check_sparse_solving(Solver& solver, const typename Solver::MatrixType& A, const Rhs& b, const DenseMat& dA, const DenseRhs& db)
{
  typedef typename Solver::MatrixType Mat;
  typedef typename Mat::Scalar Scalar;
  typedef typename Mat::StorageIndex StorageIndex;

  DenseRhs refX = dA.householderQr().solve(db);
  {
    Rhs x(A.cols(), b.cols());
    Rhs oldb = b;

    solver.compute(A);
    if (solver.info() != Success)
    {
      std::cerr << "ERROR | sparse solver testing, factorization failed (" << typeid(Solver).name() << ")\n";
      VERIFY(solver.info() == Success);
    }
    x = solver.solve(b);
    if (solver.info() != Success)
    {
      std::cerr << "WARNING | sparse solver testing: solving failed (" << typeid(Solver).name() << ")\n";
      return;
    }
    VERIFY(oldb.isApprox(b) && "sparse solver testing: the rhs should not be modified!");
    VERIFY(x.isApprox(refX,test_precision<Scalar>()));

    x.setZero();
    solve_with_guess(solver, b, x, x);
    VERIFY(solver.info() == Success && "solving failed when using analyzePattern/factorize API");
    VERIFY(oldb.isApprox(b) && "sparse solver testing: the rhs should not be modified!");
    VERIFY(x.isApprox(refX,test_precision<Scalar>()));
    
    x.setZero();
    // test the analyze/factorize API
    solver.analyzePattern(A);
    solver.factorize(A);
    VERIFY(solver.info() == Success && "factorization failed when using analyzePattern/factorize API");
    x = solver.solve(b);
    VERIFY(solver.info() == Success && "solving failed when using analyzePattern/factorize API");
    VERIFY(oldb.isApprox(b) && "sparse solver testing: the rhs should not be modified!");
    VERIFY(x.isApprox(refX,test_precision<Scalar>()));
    
    x.setZero();
    // test with Map
    MappedSparseMatrix<Scalar,Mat::Options,StorageIndex> Am(A.rows(), A.cols(), A.nonZeros(), const_cast<StorageIndex*>(A.outerIndexPtr()), const_cast<StorageIndex*>(A.innerIndexPtr()), const_cast<Scalar*>(A.valuePtr()));
    solver.compute(Am);
    VERIFY(solver.info() == Success && "factorization failed when using Map");
    DenseRhs dx(refX);
    dx.setZero();
    Map<DenseRhs> xm(dx.data(), dx.rows(), dx.cols());
    Map<const DenseRhs> bm(db.data(), db.rows(), db.cols());
    xm = solver.solve(bm);
    VERIFY(solver.info() == Success && "solving failed when using Map");
    VERIFY(oldb.isApprox(bm) && "sparse solver testing: the rhs should not be modified!");
    VERIFY(xm.isApprox(refX,test_precision<Scalar>()));
  }
  
  // if not too large, do some extra check:
  if(A.rows()<2000)
  {
    // test initialization ctor
    {
      Rhs x(b.rows(), b.cols());
      Solver solver2(A);
      VERIFY(solver2.info() == Success);
      x = solver2.solve(b);
      VERIFY(x.isApprox(refX,test_precision<Scalar>()));
    }

    // test dense Block as the result and rhs:
    {
      DenseRhs x(refX.rows(), refX.cols());
      DenseRhs oldb(db);
      x.setZero();
      x.block(0,0,x.rows(),x.cols()) = solver.solve(db.block(0,0,db.rows(),db.cols()));
      VERIFY(oldb.isApprox(db) && "sparse solver testing: the rhs should not be modified!");
      VERIFY(x.isApprox(refX,test_precision<Scalar>()));
    }

    // test uncompressed inputs
    {
      Mat A2 = A;
      A2.reserve((ArrayXf::Random(A.outerSize())+2).template cast<typename Mat::StorageIndex>().eval());
      solver.compute(A2);
      Rhs x = solver.solve(b);
      VERIFY(x.isApprox(refX,test_precision<Scalar>()));
    }

    // test expression as input
    {
      solver.compute(0.5*(A+A));
      Rhs x = solver.solve(b);
      VERIFY(x.isApprox(refX,test_precision<Scalar>()));

      Solver solver2(0.5*(A+A));
      Rhs x2 = solver2.solve(b);
      VERIFY(x2.isApprox(refX,test_precision<Scalar>()));
    }
  }
}

template<typename Solver, typename Rhs>
void check_sparse_solving_real_cases(Solver& solver, const typename Solver::MatrixType& A, const Rhs& b, const typename Solver::MatrixType& fullA, const Rhs& refX)
{
  typedef typename Solver::MatrixType Mat;
  typedef typename Mat::Scalar Scalar;
  typedef typename Mat::RealScalar RealScalar;
  
  Rhs x(A.cols(), b.cols());

  solver.compute(A);
  if (solver.info() != Success)
  {
    std::cerr << "ERROR | sparse solver testing, factorization failed (" << typeid(Solver).name() << ")\n";
    VERIFY(solver.info() == Success);
  }
  x = solver.solve(b);
  
  if (solver.info() != Success)
  {
    std::cerr << "WARNING | sparse solver testing, solving failed (" << typeid(Solver).name() << ")\n";
    return;
  }
  
  RealScalar res_error = (fullA*x-b).norm()/b.norm();  
  VERIFY( (res_error <= test_precision<Scalar>() ) && "sparse solver failed without noticing it"); 

  
  if(refX.size() != 0 && (refX - x).norm()/refX.norm() > test_precision<Scalar>())
  {
    std::cerr << "WARNING | found solution is different from the provided reference one\n";
  }
  
}
template<typename Solver, typename DenseMat>
void check_sparse_determinant(Solver& solver, const typename Solver::MatrixType& A, const DenseMat& dA)
{
  typedef typename Solver::MatrixType Mat;
  typedef typename Mat::Scalar Scalar;
  
  solver.compute(A);
  if (solver.info() != Success)
  {
    std::cerr << "WARNING | sparse solver testing: factorization failed (check_sparse_determinant)\n";
    return;
  }

  Scalar refDet = dA.determinant();
  VERIFY_IS_APPROX(refDet,solver.determinant());
}
template<typename Solver, typename DenseMat>
void check_sparse_abs_determinant(Solver& solver, const typename Solver::MatrixType& A, const DenseMat& dA)
{
  using std::abs;
  typedef typename Solver::MatrixType Mat;
  typedef typename Mat::Scalar Scalar;
  
  solver.compute(A);
  if (solver.info() != Success)
  {
    std::cerr << "WARNING | sparse solver testing: factorization failed (check_sparse_abs_determinant)\n";
    return;
  }

  Scalar refDet = abs(dA.determinant());
  VERIFY_IS_APPROX(refDet,solver.absDeterminant());
}

template<typename Solver, typename DenseMat>
int generate_sparse_spd_problem(Solver& , typename Solver::MatrixType& A, typename Solver::MatrixType& halfA, DenseMat& dA, int maxSize = 300)
{
  typedef typename Solver::MatrixType Mat;
  typedef typename Mat::Scalar Scalar;
  typedef Matrix<Scalar,Dynamic,Dynamic> DenseMatrix;

  int size = internal::random<int>(1,maxSize);
  double density = (std::max)(8./(size*size), 0.01);

  Mat M(size, size);
  DenseMatrix dM(size, size);

  initSparse<Scalar>(density, dM, M, ForceNonZeroDiag);

  A = M * M.adjoint();
  dA = dM * dM.adjoint();
  
  halfA.resize(size,size);
  if(Solver::UpLo==(Lower|Upper))
    halfA = A;
  else
    halfA.template selfadjointView<Solver::UpLo>().rankUpdate(M);
  
  return size;
}


#ifdef TEST_REAL_CASES
template<typename Scalar>
inline std::string get_matrixfolder()
{
  std::string mat_folder = TEST_REAL_CASES; 
  if( internal::is_same<Scalar, std::complex<float> >::value || internal::is_same<Scalar, std::complex<double> >::value )
    mat_folder  = mat_folder + static_cast<std::string>("/complex/");
  else
    mat_folder = mat_folder + static_cast<std::string>("/real/");
  return mat_folder;
}
std::string sym_to_string(int sym)
{
  if(sym==Symmetric) return "Symmetric ";
  if(sym==SPD)       return "SPD ";
  return "";
}
template<typename Derived>
std::string solver_stats(const IterativeSolverBase<Derived> &solver)
{
  std::stringstream ss;
  ss << solver.iterations() << " iters, error: " << solver.error();
  return ss.str();
}
template<typename Derived>
std::string solver_stats(const SparseSolverBase<Derived> &/*solver*/)
{
  return "";
}
#endif

template<typename Solver> void check_sparse_spd_solving(Solver& solver, int maxSize = 300, int maxRealWorldSize = 100000)
{
  typedef typename Solver::MatrixType Mat;
  typedef typename Mat::Scalar Scalar;
  typedef typename Mat::StorageIndex StorageIndex;
  typedef SparseMatrix<Scalar,ColMajor, StorageIndex> SpMat;
  typedef SparseVector<Scalar, 0, StorageIndex> SpVec;
  typedef Matrix<Scalar,Dynamic,Dynamic> DenseMatrix;
  typedef Matrix<Scalar,Dynamic,1> DenseVector;

  // generate the problem
  Mat A, halfA;
  DenseMatrix dA;
  for (int i = 0; i < g_repeat; i++) {
    int size = generate_sparse_spd_problem(solver, A, halfA, dA, maxSize);

    // generate the right hand sides
    int rhsCols = internal::random<int>(1,16);
    double density = (std::max)(8./(size*rhsCols), 0.1);
    SpMat B(size,rhsCols);
    DenseVector b = DenseVector::Random(size);
    DenseMatrix dB(size,rhsCols);
    initSparse<Scalar>(density, dB, B, ForceNonZeroDiag);
    SpVec c = B.col(0);
    DenseVector dc = dB.col(0);
  
    CALL_SUBTEST( check_sparse_solving(solver, A,     b,  dA, b)  );
    CALL_SUBTEST( check_sparse_solving(solver, halfA, b,  dA, b)  );
    CALL_SUBTEST( check_sparse_solving(solver, A,     dB, dA, dB) );
    CALL_SUBTEST( check_sparse_solving(solver, halfA, dB, dA, dB) );
    CALL_SUBTEST( check_sparse_solving(solver, A,     B,  dA, dB) );
    CALL_SUBTEST( check_sparse_solving(solver, halfA, B,  dA, dB) );
    CALL_SUBTEST( check_sparse_solving(solver, A,     c,  dA, dc) );
    CALL_SUBTEST( check_sparse_solving(solver, halfA, c,  dA, dc) );
    
    // check only once
    if(i==0)
    {
      b = DenseVector::Zero(size);
      check_sparse_solving(solver, A, b, dA, b);
    }
  }
  
  // First, get the folder 
#ifdef TEST_REAL_CASES
  // Test real problems with double precision only
  if (internal::is_same<typename NumTraits<Scalar>::Real, double>::value)
  {
    std::string mat_folder = get_matrixfolder<Scalar>();
    MatrixMarketIterator<Scalar> it(mat_folder);
    for (; it; ++it)
    {
      if (it.sym() == SPD){
        A = it.matrix();
        if(A.diagonal().size() <= maxRealWorldSize)
        {
          DenseVector b = it.rhs();
          DenseVector refX = it.refX();
          PermutationMatrix<Dynamic, Dynamic, StorageIndex> pnull;
          halfA.resize(A.rows(), A.cols());
          if(Solver::UpLo == (Lower|Upper))
            halfA = A;
          else
            halfA.template selfadjointView<Solver::UpLo>() = A.template triangularView<Eigen::Lower>().twistedBy(pnull);
          
          std::cout << "INFO | Testing " << sym_to_string(it.sym()) << "sparse problem " << it.matname()
                  << " (" << A.rows() << "x" << A.cols() << ") using " << typeid(Solver).name() << "..." << std::endl;
          CALL_SUBTEST( check_sparse_solving_real_cases(solver, A,     b, A, refX) );
          std::string stats = solver_stats(solver);
          if(stats.size()>0)
            std::cout << "INFO |  " << stats << std::endl;
          CALL_SUBTEST( check_sparse_solving_real_cases(solver, halfA, b, A, refX) );
        }
        else
        {
          std::cout << "INFO | Skip sparse problem \"" << it.matname() << "\" (too large)" << std::endl;
        }
      }
    }
  }
#else
  EIGEN_UNUSED_VARIABLE(maxRealWorldSize);
#endif
}

template<typename Solver> void check_sparse_spd_determinant(Solver& solver)
{
  typedef typename Solver::MatrixType Mat;
  typedef typename Mat::Scalar Scalar;
  typedef Matrix<Scalar,Dynamic,Dynamic> DenseMatrix;

  // generate the problem
  Mat A, halfA;
  DenseMatrix dA;
  generate_sparse_spd_problem(solver, A, halfA, dA, 30);
  
  for (int i = 0; i < g_repeat; i++) {
    check_sparse_determinant(solver, A,     dA);
    check_sparse_determinant(solver, halfA, dA );
  }
}

template<typename Solver, typename DenseMat>
Index generate_sparse_square_problem(Solver&, typename Solver::MatrixType& A, DenseMat& dA, int maxSize = 300, int options = ForceNonZeroDiag)
{
  typedef typename Solver::MatrixType Mat;
  typedef typename Mat::Scalar Scalar;

  Index size = internal::random<int>(1,maxSize);
  double density = (std::max)(8./(size*size), 0.01);
  
  A.resize(size,size);
  dA.resize(size,size);

  initSparse<Scalar>(density, dA, A, options);
  
  return size;
}


struct prune_column {
  Index m_col;
  prune_column(Index col) : m_col(col) {}
  template<class Scalar>
  bool operator()(Index, Index col, const Scalar&) const {
    return col != m_col;
  }
};


template<typename Solver> void check_sparse_square_solving(Solver& solver, int maxSize = 300, int maxRealWorldSize = 100000, bool checkDeficient = false)
{
  typedef typename Solver::MatrixType Mat;
  typedef typename Mat::Scalar Scalar;
  typedef SparseMatrix<Scalar,ColMajor, typename Mat::StorageIndex> SpMat;
  typedef SparseVector<Scalar, 0, typename Mat::StorageIndex> SpVec;
  typedef Matrix<Scalar,Dynamic,Dynamic> DenseMatrix;
  typedef Matrix<Scalar,Dynamic,1> DenseVector;

  int rhsCols = internal::random<int>(1,16);

  Mat A;
  DenseMatrix dA;
  for (int i = 0; i < g_repeat; i++) {
    Index size = generate_sparse_square_problem(solver, A, dA, maxSize);

    A.makeCompressed();
    DenseVector b = DenseVector::Random(size);
    DenseMatrix dB(size,rhsCols);
    SpMat B(size,rhsCols);
    double density = (std::max)(8./(size*rhsCols), 0.1);
    initSparse<Scalar>(density, dB, B, ForceNonZeroDiag);
    B.makeCompressed();
    SpVec c = B.col(0);
    DenseVector dc = dB.col(0);
    CALL_SUBTEST(check_sparse_solving(solver, A, b,  dA, b));
    CALL_SUBTEST(check_sparse_solving(solver, A, dB, dA, dB));
    CALL_SUBTEST(check_sparse_solving(solver, A, B,  dA, dB));
    CALL_SUBTEST(check_sparse_solving(solver, A, c,  dA, dc));
    
    // check only once
    if(i==0)
    {
      b = DenseVector::Zero(size);
      check_sparse_solving(solver, A, b, dA, b);
    }
    // regression test for Bug 792 (structurally rank deficient matrices):
    if(checkDeficient && size>1) {
      Index col = internal::random<int>(0,int(size-1));
      A.prune(prune_column(col));
      solver.compute(A);
      VERIFY_IS_EQUAL(solver.info(), NumericalIssue);
    }
  }
  
  // First, get the folder 
#ifdef TEST_REAL_CASES
  // Test real problems with double precision only
  if (internal::is_same<typename NumTraits<Scalar>::Real, double>::value)
  {
    std::string mat_folder = get_matrixfolder<Scalar>();
    MatrixMarketIterator<Scalar> it(mat_folder);
    for (; it; ++it)
    {
      A = it.matrix();
      if(A.diagonal().size() <= maxRealWorldSize)
      {
        DenseVector b = it.rhs();
        DenseVector refX = it.refX();
        std::cout << "INFO | Testing " << sym_to_string(it.sym()) << "sparse problem " << it.matname()
                  << " (" << A.rows() << "x" << A.cols() << ") using " << typeid(Solver).name() << "..." << std::endl;
        CALL_SUBTEST(check_sparse_solving_real_cases(solver, A, b, A, refX));
        std::string stats = solver_stats(solver);
        if(stats.size()>0)
          std::cout << "INFO |  " << stats << std::endl;
      }
      else
      {
        std::cout << "INFO | SKIP sparse problem \"" << it.matname() << "\" (too large)" << std::endl;
      }
    }
  }
#else
  EIGEN_UNUSED_VARIABLE(maxRealWorldSize);
#endif

}

template<typename Solver> void check_sparse_square_determinant(Solver& solver)
{
  typedef typename Solver::MatrixType Mat;
  typedef typename Mat::Scalar Scalar;
  typedef Matrix<Scalar,Dynamic,Dynamic> DenseMatrix;
  
  for (int i = 0; i < g_repeat; i++) {
    // generate the problem
    Mat A;
    DenseMatrix dA;
    
    int size = internal::random<int>(1,30);
    dA.setRandom(size,size);
    
    dA = (dA.array().abs()<0.3).select(0,dA);
    dA.diagonal() = (dA.diagonal().array()==0).select(1,dA.diagonal());
    A = dA.sparseView();
    A.makeCompressed();
  
    check_sparse_determinant(solver, A, dA);
  }
}

template<typename Solver> void check_sparse_square_abs_determinant(Solver& solver)
{
  typedef typename Solver::MatrixType Mat;
  typedef typename Mat::Scalar Scalar;
  typedef Matrix<Scalar,Dynamic,Dynamic> DenseMatrix;

  for (int i = 0; i < g_repeat; i++) {
    // generate the problem
    Mat A;
    DenseMatrix dA;
    generate_sparse_square_problem(solver, A, dA, 30);
    A.makeCompressed();
    check_sparse_abs_determinant(solver, A, dA);
  }
}

template<typename Solver, typename DenseMat>
void generate_sparse_leastsquare_problem(Solver&, typename Solver::MatrixType& A, DenseMat& dA, int maxSize = 300, int options = ForceNonZeroDiag)
{
  typedef typename Solver::MatrixType Mat;
  typedef typename Mat::Scalar Scalar;

  int rows = internal::random<int>(1,maxSize);
  int cols = internal::random<int>(1,rows);
  double density = (std::max)(8./(rows*cols), 0.01);
  
  A.resize(rows,cols);
  dA.resize(rows,cols);

  initSparse<Scalar>(density, dA, A, options);
}

template<typename Solver> void check_sparse_leastsquare_solving(Solver& solver)
{
  typedef typename Solver::MatrixType Mat;
  typedef typename Mat::Scalar Scalar;
  typedef SparseMatrix<Scalar,ColMajor, typename Mat::StorageIndex> SpMat;
  typedef Matrix<Scalar,Dynamic,Dynamic> DenseMatrix;
  typedef Matrix<Scalar,Dynamic,1> DenseVector;

  int rhsCols = internal::random<int>(1,16);

  Mat A;
  DenseMatrix dA;
  for (int i = 0; i < g_repeat; i++) {
    generate_sparse_leastsquare_problem(solver, A, dA);

    A.makeCompressed();
    DenseVector b = DenseVector::Random(A.rows());
    DenseMatrix dB(A.rows(),rhsCols);
    SpMat B(A.rows(),rhsCols);
    double density = (std::max)(8./(A.rows()*rhsCols), 0.1);
    initSparse<Scalar>(density, dB, B, ForceNonZeroDiag);
    B.makeCompressed();
    check_sparse_solving(solver, A, b,  dA, b);
    check_sparse_solving(solver, A, dB, dA, dB);
    check_sparse_solving(solver, A, B,  dA, dB);
    
    // check only once
    if(i==0)
    {
      b = DenseVector::Zero(A.rows());
      check_sparse_solving(solver, A, b, dA, b);
    }
  }
}
