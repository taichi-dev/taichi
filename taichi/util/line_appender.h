#pragma once

#include <string>

#include "spdlog/spdlog.h"
#include "taichi/common/core.h"

TI_NAMESPACE_BEGIN

class LineAppender {
 public:
  explicit LineAppender(int indent_size = 2)
      : single_indent_(indent_size, ' ') {
  }

  LineAppender(const LineAppender &) = default;
  LineAppender &operator=(const LineAppender &) = default;
  LineAppender(LineAppender &&) = default;
  LineAppender &operator=(LineAppender &&) = default;

  inline const std::string &lines() const {
    return lines_;
  }

  template <typename... Args>
  void append(std::string f, Args &&... args) {
    lines_ += indent_ + fmt::format(f, std::forward<Args>(args)...) + '\n';
  }

  inline void append_raw(const std::string &s) {
    lines_ += s + '\n';
  }

  inline void dump(std::string *output) {
    *output = std::move(lines_);
  }

  void clear_lines() {
    // Free up the memory as well
    std::string s;
    dump(&s);
  }

  void clear_all() {
    clear_lines();
    indent_.clear();
  }

  inline void push_indent() {
    indent_ += single_indent_;
  }

  inline void pop_indent() {
    indent_.erase(indent_.size() - single_indent_.size());
  }

 private:
  std::string single_indent_;
  std::string indent_;
  std::string lines_;
};

class ScopedIndent {
 public:
  explicit ScopedIndent(LineAppender &la) : la_(la) {
    la_.push_indent();
  }

  ~ScopedIndent() {
    la_.pop_indent();
  }

 private:
  LineAppender &la_;
};

TI_NAMESPACE_END
