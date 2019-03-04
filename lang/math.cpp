//
// Created by yuanming on 3/3/19.
//
#include "math.h"
#include "ir.h"

TLANG_NAMESPACE_BEGIN

void Matrix::fill_global(DataType dt) {
  for (int i = 0; i < n * m; i++) {
    entries[i].set(global_new(dt));
  }
}

TLANG_NAMESPACE_END
