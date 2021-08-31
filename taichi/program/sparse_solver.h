#pragma once

#include "sparse_matrix.h"

#include "taichi/common/core.h"
#include "taichi/inc/constants.h"
#include "Eigen/Sparse"

namespace taichi {
namespace lang {

class SparseLUSolver{
public:
    SparseLUSolver();
    bool compute(const SparseMatrix& sm);
    void analyzePattern(const SparseMatrix& sm);
    void factorize(const SparseMatrix& sm);
    Eigen::VectorXf solve(const Eigen::Ref<const Eigen::VectorXf> &b);

private:
    using LU = Eigen::SparseLU<Eigen::SparseMatrix<float32>>;
    std::unique_ptr<LU> solver_{nullptr};
};

class SparseLDLTSolver{
public:
    SparseLDLTSolver();
    bool compute(const SparseMatrix& sm);
    void analyzePattern(const SparseMatrix& sm);
    void factorize(const SparseMatrix& sm);
    Eigen::VectorXf solve(const Eigen::Ref<const Eigen::VectorXf> &b);

private:
    using LDLT = Eigen::SimplicialLDLT<Eigen::SparseMatrix<float32>>;
    std::unique_ptr<LDLT> solver_{nullptr};
};

class SparseLLTSolver{
public:
    SparseLLTSolver();
    bool compute(const SparseMatrix& sm);
    void analyzePattern(const SparseMatrix& sm);
    void factorize(const SparseMatrix& sm);
    Eigen::VectorXf solve(const Eigen::Ref<const Eigen::VectorXf> &b);

private:
    using LLT = Eigen::SimplicialLLT<Eigen::SparseMatrix<float32>>;
    std::unique_ptr<LLT> solver_{nullptr};
};

} // namespace lang
} // namespace taichi
