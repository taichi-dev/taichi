/*******************************************************************************
     Copyright (c) 2020 The Taichi Authors
     Use of this software is governed by the LICENSE file.
*******************************************************************************/

// Lists of extension features
PER_EXTENSION(sparse)
PER_EXTENSION(data64)   // Metal doesn't support 64-bit data buffers yet...
PER_EXTENSION(adstack)  // For keeping the history of mutable local variables
