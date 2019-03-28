#include "math.h"
#include "ir.h"

TLANG_NAMESPACE_BEGIN

void Matrix::fill_global(DataType dt) {
  for (int i = 0; i < n * m; i++) {
    entries[i].set(global_new(dt));
  }
}

void Mutable(Matrix &mat, DataType dt) {
  for (int i = 0; i < mat.entries.size(); i++) {
    declare_unnamed_var(mat.entries[i], dt);
  }
}

TLANG_NAMESPACE_END
