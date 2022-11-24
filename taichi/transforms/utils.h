#pragma once

namespace taichi::lang {

// These two helper functions are targeting cases where x is assumed
// non-negative but with a signed type so no automatic transformation to bitwise
// operations can be applied in other compiler passes.
Stmt *generate_mod(VecStatement *stmts, Stmt *x, int y);
Stmt *generate_div(VecStatement *stmts, Stmt *x, int y);

}  // namespace taichi::lang
