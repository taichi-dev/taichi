#pragma once

TLANG_NAMESPACE_BEGIN

enum OffloadedTaskType : int {
  serial,
  range_for,
  struct_for,
  listgen,
  gc,
};

TLANG_NAMESPACE_END
