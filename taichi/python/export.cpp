/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#include "taichi/python/export.h"
#include "taichi/common/interface.h"
#include "taichi/util/io.h"

TI_NAMESPACE_BEGIN

void export_lang(py::module &m);

PYBIND11_MODULE(taichi_core, m) {
  m.doc() = "taichi_core";

  for (auto &kv : InterfaceHolder::get_instance()->methods) {
    kv.second(&m);
  }

  export_visual(m);
  export_math(m);
  export_misc(m);
  export_lang(m);
}

TI_NAMESPACE_END
