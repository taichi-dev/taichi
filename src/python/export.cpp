/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#include <taichi/python/export.h>
#include <taichi/common/interface.h>
#include <taichi/visualization/rgb.h>
#include <taichi/io/io.h>
#include <taichi/geometry/factory.h>

TC_NAMESPACE_BEGIN

PYBIND11_MODULE(taichi_core, m) {
  m.doc() = "taichi_core";

  for (auto &kv : InterfaceHolder::get_instance()->methods) {
    kv.second(&m);
  }

  export_math(m);
  export_dynamics(m);
  export_visual(m);
  export_io(m);
  export_misc(m);
}

TC_NAMESPACE_END
