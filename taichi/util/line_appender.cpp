#include "taichi/util/line_appender.h"

TI_NAMESPACE_BEGIN

std::string LineAppender::lines(const std::string &sep) const {
  std::string s;
  for (const auto &seg : str_segs_) {
    s += fmt::format("{}{}", fmt::join(seg, sep), sep);
  }
  return s;
}

void LineAppender::clear_lines() {
  {
    // Free up the memory as well
    StrSegList tmp;
    tmp.swap(str_segs_);
    str_segs_.clear();
  }
  reserved_cursors_.clear();

  str_segs_.push_back(StrSeg());
  cur_seg_iter_ = str_segs_.begin();
}

LineAppender::Cursor LineAppender::make_cursor() {
  const int id = next_cursor_id_++;
  Cursor rc(this, id);
  auto iter = cur_seg_iter_;
  reserved_cursors_[id] = {iter};
  ++iter;
  cur_seg_iter_ = str_segs_.insert(iter, StrSeg());
  return rc;
}

void LineAppender::rewind_to_cursor(const Cursor &rc) {
  TI_ASSERT(rc.parent_ == this);
  const auto record = reserved_cursors_.at(rc.id_);
  reserved_cursors_.erase(rc.id_);
  cur_seg_iter_ = record.seg_iter;
}

void LineAppender::rewind_to_end() {
  cur_seg_iter_ = std::prev(str_segs_.end());
}

TI_NAMESPACE_END
