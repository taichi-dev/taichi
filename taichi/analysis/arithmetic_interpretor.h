#pragma once

#include <optional>
#include <unordered_set>
#include <unordered_map>

#include "taichi/ir/statements.h"
#include "taichi/ir/type.h"

namespace taichi {
namespace lang {

/**
 * Interprets a sequence of CHI IR statements within a block (acts like a
 * VM based on CHI).
 */
class ArithmeticInterpretor {
 public:
  /**
   * Evaluation context that maps from a Stmt to a constant value.
   */
  class EvalContext {
   public:
    /**
     * Pre-defines a value for statement @param s.
     *
     * @param s: Statement to be evaluated
     * @param c: Predefined value
     */
    EvalContext &insert(const Stmt *s, TypedConstant c) {
      map_[s] = c;
      return *this;
    }

    /**
     * Tries to get the evaluated value for statement @param s.
     *
     * @param s: Statement to get
     * @return: The evaluated value, empty if not found.
     */
    std::optional<TypedConstant> maybe_get(const Stmt *s) const {
      auto itr = map_.find(s);
      if (itr == map_.end()) {
        return std::nullopt;
      }
      return itr->second;
    }

    /**
     * Tells the interpretor to ignore statement @param s.
     *
     * This is effective only for statements that are not supported by
     * ArithmeticInterpretor.
     *
     * @param s: Statement to ignore
     */
    void ignore(const Stmt *s) {
      ignored_.insert(s);
    }

    /**
     * Checks if statement @param s is ignored.
     *
     * @return: True if ignored
     */
    bool should_ignore(const Stmt *s) {
      return ignored_.count(s) > 0;
    }

   private:
    std::unordered_map<const Stmt *, TypedConstant> map_;
    std::unordered_set<const Stmt *> ignored_;
  };

  /**
   * Defines the region of CHI statments to be evaluated.
   */
  struct CodeRegion {
    // Defines the sequence of CHI statements.
    Block *block{nullptr};
    // The beginning statement within |block| to be evaluated. If nullptr,
    // evaluates from the beginning of |block|.
    Stmt *begin{nullptr};
    // The ending statement (exclusive) within |block| to be evaluated. If
    // nullptr, evaluates to the end of |block|.
    Stmt *end{nullptr};
  };

  /**
   * Evaluates the sequence of CHI as defined in |region|.
   * @param region: A sequence of CHI statements to be evaluated
   * @param init_ctx: This context can mock the result for certain types of
   *   statements that are not supported, or cannot be evaluated statically.
   */
  std::optional<TypedConstant> evaluate(const CodeRegion &region,
                                        const EvalContext &init_ctx) const;
};

}  // namespace lang
}  // namespace taichi
