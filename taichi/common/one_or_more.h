#pragma once

#include <vector>
#include <variant>

namespace taichi {

template <typename T, class Container = std::vector<T>>
struct one_or_more {
  using value_type = T;

  std::variant<value_type, Container> var;

  // NOLINTNEXTLINE
  one_or_more(value_type const &value) : var(value) {
  }

  // NOLINTNEXTLINE
  one_or_more(value_type &value) : var(value) {
  }

  // NOLINTNEXTLINE
  one_or_more(value_type &&value) : var(std::move(value)) {
  }

  // NOLINTNEXTLINE
  one_or_more(Container const &value) : var(value) {
  }

  // NOLINTNEXTLINE
  one_or_more(Container &value) : var(value) {
  }

  // NOLINTNEXTLINE
  one_or_more(Container &&value) : var(std::move(value)) {
  }

  one_or_more(one_or_more &value) : var(value.var) {
  }

  one_or_more(one_or_more &&value) : var(std::move(value.var)) {
  }

  one_or_more &operator=(one_or_more &&) = default;

  value_type *begin() {
    if (value_type *s = std::get_if<value_type>(&var)) {
      return s;
    } else {
      return (std::get_if<std::vector<value_type>>(&var))->data();
    }
  }

  value_type *end() {
    if (value_type *s = std::get_if<value_type>(&var)) {
      if (*s) {
        return s + 1;
      } else {
        return s;
      }
    } else {
      auto *vec = std::get_if<Container>(&var);
      return vec->data() + vec->size();
    }
  }

  size_t size() const {
    if (const value_type *s = std::get_if<value_type>(&var)) {
      if (*s) {
        return 1;
      } else {
        return 0;
      }
    } else {
      return std::get_if<Container>(&var)->size();
    }
  }

  bool empty() const {
    if (const value_type *s = std::get_if<value_type>(&var)) {
      if (*s) {
        return false;
      } else {
        return true;
      }
    } else {
      return std::get_if<Container>(&var)->empty();
    }
  }
};

}  // namespace taichi
