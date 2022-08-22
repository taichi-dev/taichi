#pragma once

namespace taichi {
namespace lang {

Stmt *generate_mod_x_div_y(VecStatement *stmts, Stmt *num, int x, int y);

std::string message_append_backtrace_info(const std::string &message,
                                          Stmt *stmt);

}  // namespace lang
}  // namespace taichi
