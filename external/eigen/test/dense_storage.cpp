// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2013 Hauke Heibel <hauke.heibel@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"

#include <Eigen/Core>

template <typename T, int Rows, int Cols>
void dense_storage_copy()
{
  static const int Size = ((Rows==Dynamic || Cols==Dynamic) ? Dynamic : Rows*Cols);
  typedef DenseStorage<T,Size, Rows,Cols, 0> DenseStorageType;
  
  const int rows = (Rows==Dynamic) ? 4 : Rows;
  const int cols = (Cols==Dynamic) ? 3 : Cols;
  const int size = rows*cols;
  DenseStorageType reference(size, rows, cols);
  T* raw_reference = reference.data();
  for (int i=0; i<size; ++i)
    raw_reference[i] = static_cast<T>(i);
    
  DenseStorageType copied_reference(reference);
  const T* raw_copied_reference = copied_reference.data();
  for (int i=0; i<size; ++i)
    VERIFY_IS_EQUAL(raw_reference[i], raw_copied_reference[i]);
}

template <typename T, int Rows, int Cols>
void dense_storage_assignment()
{
  static const int Size = ((Rows==Dynamic || Cols==Dynamic) ? Dynamic : Rows*Cols);
  typedef DenseStorage<T,Size, Rows,Cols, 0> DenseStorageType;
  
  const int rows = (Rows==Dynamic) ? 4 : Rows;
  const int cols = (Cols==Dynamic) ? 3 : Cols;
  const int size = rows*cols;
  DenseStorageType reference(size, rows, cols);
  T* raw_reference = reference.data();
  for (int i=0; i<size; ++i)
    raw_reference[i] = static_cast<T>(i);
    
  DenseStorageType copied_reference;
  copied_reference = reference;
  const T* raw_copied_reference = copied_reference.data();
  for (int i=0; i<size; ++i)
    VERIFY_IS_EQUAL(raw_reference[i], raw_copied_reference[i]);
}

void test_dense_storage()
{
  dense_storage_copy<int,Dynamic,Dynamic>();  
  dense_storage_copy<int,Dynamic,3>();
  dense_storage_copy<int,4,Dynamic>();
  dense_storage_copy<int,4,3>();

  dense_storage_copy<float,Dynamic,Dynamic>();
  dense_storage_copy<float,Dynamic,3>();
  dense_storage_copy<float,4,Dynamic>();  
  dense_storage_copy<float,4,3>();
  
  dense_storage_assignment<int,Dynamic,Dynamic>();  
  dense_storage_assignment<int,Dynamic,3>();
  dense_storage_assignment<int,4,Dynamic>();
  dense_storage_assignment<int,4,3>();

  dense_storage_assignment<float,Dynamic,Dynamic>();
  dense_storage_assignment<float,Dynamic,3>();
  dense_storage_assignment<float,4,Dynamic>();  
  dense_storage_assignment<float,4,3>();  
}
