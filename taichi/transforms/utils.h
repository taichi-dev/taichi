#pragma once

namespace taichi {
namespace lang {

Stmt *generate_mod_x_div_y(VecStatement *stmts, Stmt *num, int x, int y);

}  // namespace lang
}  // namespace taichi
