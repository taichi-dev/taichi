#pragma once

namespace taichi::lang {

Stmt *generate_mod(VecStatement *stmts, Stmt *x, int y);
Stmt *generate_div(VecStatement *stmts, Stmt *x, int y);

}  // namespace taichi::lang
