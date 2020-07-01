#pragma once

#include <list>
#include <string>
#include <vector>
#include <unordered_map>

#include "spdlog/spdlog.h"
#include "taichi/common/core.h"

TI_NAMESPACE_BEGIN

class LineAppender {
 public:
  // A cursor is a fixed coordinate in the line sequences maintained by a
  // LineAppender. Example usage:
  //
  // LineAppender l;
  // l.append("a");  // "a"
  // l.append("b");  // "ab"
  // auto c = l.make_cursor();
  // l.append("e");  // "abe"
  // l.rewind_to_cursor(c);  // |c| is invalid after this call
  // l.append("c");  // "abce"
  // l.append("d");  // "abcde"
  // l.lines();      // "abcde"
  class Cursor {
   private:
    // Non-public constructable on purpose, so that users cannot pass in an
    // arbitrary cursor.
    explicit Cursor(const LineAppender *parent, int id)
        : parent_(parent), id_(id) {
    }

    friend class LineAppender;
    const LineAppender *parent_;
    int id_;
  };

  explicit LineAppender(int indent_size = 2)
      : single_indent_(indent_size, ' '), next_cursor_id_(0) {
    clear_lines();
  }

  LineAppender(const LineAppender &) = default;
  LineAppender &operator=(const LineAppender &) = default;
  LineAppender(LineAppender &&) = default;
  LineAppender &operator=(LineAppender &&) = default;

  std::string lines(const std::string &sep = "") const;

  template <typename... Args>
  void append(std::string f, Args &&... args) {
    cur_seg_iter_->push_back(
        indent_ + fmt::format(std::move(f), std::forward<Args>(args)...));
  }

  inline void append_raw(std::string s) {
    cur_seg_iter_->push_back(std::move(s));
  }

  // TODO(k-ye): remove this.
  inline void dump(std::string *output) {
    *output = lines();
    clear_all();
  }

  // TODO(k-ye): If we want to create a new LineAppender with the indentation
  // preversed, provide a dedicated method for that. Merge clear_lines() and
  // clear_all() into one clear().
  void clear_lines();

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

  inline std::string indent() const {
    return indent_;
  }

  // Returns a cursor that points at the current tail of the LineAppender.
  Cursor make_cursor();

  // Rewinds the LineAppender to where |c| points to.
  //
  // After calling this, |c| will be invalid. I.e. a cursor cannot be fed back
  // more than once.
  void rewind_to_cursor(const Cursor &rc);
  // Rewinds to the tail.
  void rewind_to_end();

 private:
  using StrSeg = std::vector<std::string>;
  using StrSegList = std::list<StrSeg>;

  struct CursorRecord {
    StrSegList::iterator seg_iter;
  };

  std::string single_indent_;
  std::string indent_;

  StrSegList str_segs_;
  StrSegList::iterator cur_seg_iter_;
  std::unordered_map<int, CursorRecord> reserved_cursors_;
  int next_cursor_id_;
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

class ScopedCursor {
 public:
  explicit ScopedCursor(LineAppender &la, const LineAppender::Cursor &c)
      : la_(la) {
    la_.rewind_to_cursor(c);
  }

  ~ScopedCursor() {
    la_.rewind_to_end();
  }

 private:
  LineAppender &la_;
};

TI_NAMESPACE_END
