// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// This C++ file compiles to binary code that can be linked to by your C program,
// thanks to the extern "C" syntax used in the declarations in binary_library.h.

#include "binary_library.h"

#include <Eigen/Core>

using namespace Eigen;

/************************* pointer conversion methods **********************************************/

////// class MatrixXd //////

inline MatrixXd& c_to_eigen(C_MatrixXd* ptr)
{
  return *reinterpret_cast<MatrixXd*>(ptr);
}

inline const MatrixXd& c_to_eigen(const C_MatrixXd* ptr)
{
  return *reinterpret_cast<const MatrixXd*>(ptr);
}

inline C_MatrixXd* eigen_to_c(MatrixXd& ref)
{
  return reinterpret_cast<C_MatrixXd*>(&ref);
}

inline const C_MatrixXd* eigen_to_c(const MatrixXd& ref)
{
  return reinterpret_cast<const C_MatrixXd*>(&ref);
}

////// class Map<MatrixXd> //////

inline Map<MatrixXd>& c_to_eigen(C_Map_MatrixXd* ptr)
{
  return *reinterpret_cast<Map<MatrixXd>*>(ptr);
}

inline const Map<MatrixXd>& c_to_eigen(const C_Map_MatrixXd* ptr)
{
  return *reinterpret_cast<const Map<MatrixXd>*>(ptr);
}

inline C_Map_MatrixXd* eigen_to_c(Map<MatrixXd>& ref)
{
  return reinterpret_cast<C_Map_MatrixXd*>(&ref);
}

inline const C_Map_MatrixXd* eigen_to_c(const Map<MatrixXd>& ref)
{
  return reinterpret_cast<const C_Map_MatrixXd*>(&ref);
}


/************************* implementation of classes **********************************************/


////// class MatrixXd //////


C_MatrixXd* MatrixXd_new(int rows, int cols)
{
  return eigen_to_c(*new MatrixXd(rows,cols));
}

void MatrixXd_delete(C_MatrixXd *m)
{
  delete &c_to_eigen(m);
}

double* MatrixXd_data(C_MatrixXd *m)
{
  return c_to_eigen(m).data();
}

void MatrixXd_set_zero(C_MatrixXd *m)
{
  c_to_eigen(m).setZero();
}

void MatrixXd_resize(C_MatrixXd *m, int rows, int cols)
{
  c_to_eigen(m).resize(rows,cols);
}

void MatrixXd_copy(C_MatrixXd *dst, const C_MatrixXd *src)
{
  c_to_eigen(dst) = c_to_eigen(src);
}

void MatrixXd_copy_map(C_MatrixXd *dst, const C_Map_MatrixXd *src)
{
  c_to_eigen(dst) = c_to_eigen(src);
}

void MatrixXd_set_coeff(C_MatrixXd *m, int i, int j, double coeff)
{
  c_to_eigen(m)(i,j) = coeff;
}

double MatrixXd_get_coeff(const C_MatrixXd *m, int i, int j)
{
  return c_to_eigen(m)(i,j);
}

void MatrixXd_print(const C_MatrixXd *m)
{
  std::cout << c_to_eigen(m) << std::endl;
}

void MatrixXd_multiply(const C_MatrixXd *m1, const C_MatrixXd *m2, C_MatrixXd *result)
{
  c_to_eigen(result) = c_to_eigen(m1) * c_to_eigen(m2);
}

void MatrixXd_add(const C_MatrixXd *m1, const C_MatrixXd *m2, C_MatrixXd *result)
{
  c_to_eigen(result) = c_to_eigen(m1) + c_to_eigen(m2);
}



////// class Map_MatrixXd //////


C_Map_MatrixXd* Map_MatrixXd_new(double *array, int rows, int cols)
{
  return eigen_to_c(*new Map<MatrixXd>(array,rows,cols));
}

void Map_MatrixXd_delete(C_Map_MatrixXd *m)
{
  delete &c_to_eigen(m);
}

void Map_MatrixXd_set_zero(C_Map_MatrixXd *m)
{
  c_to_eigen(m).setZero();
}

void Map_MatrixXd_copy(C_Map_MatrixXd *dst, const C_Map_MatrixXd *src)
{
  c_to_eigen(dst) = c_to_eigen(src);
}

void Map_MatrixXd_copy_matrix(C_Map_MatrixXd *dst, const C_MatrixXd *src)
{
  c_to_eigen(dst) = c_to_eigen(src);
}

void Map_MatrixXd_set_coeff(C_Map_MatrixXd *m, int i, int j, double coeff)
{
  c_to_eigen(m)(i,j) = coeff;
}

double Map_MatrixXd_get_coeff(const C_Map_MatrixXd *m, int i, int j)
{
  return c_to_eigen(m)(i,j);
}

void Map_MatrixXd_print(const C_Map_MatrixXd *m)
{
  std::cout << c_to_eigen(m) << std::endl;
}

void Map_MatrixXd_multiply(const C_Map_MatrixXd *m1, const C_Map_MatrixXd *m2, C_Map_MatrixXd *result)
{
  c_to_eigen(result) = c_to_eigen(m1) * c_to_eigen(m2);
}

void Map_MatrixXd_add(const C_Map_MatrixXd *m1, const C_Map_MatrixXd *m2, C_Map_MatrixXd *result)
{
  c_to_eigen(result) = c_to_eigen(m1) + c_to_eigen(m2);
}
