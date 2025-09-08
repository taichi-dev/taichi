/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#include "taichi/python/export.h"
#include "taichi/common/interface.h"
#include "taichi/util/io.h"
#include <cstdio>
#include <cstdlib>

namespace taichi {

static void log_boot(const char *msg) {
  const char *temp = std::getenv("TEMP");
  std::string dir = temp ? std::string(temp) : std::string(".");
  std::string path = join_path(dir, "taichi_pyd_boot.log");
  if (FILE *f = std::fopen(path.c_str(), "a")) {
    std::fprintf(f, "%s\n", msg);
    std::fclose(f);
  }
}

PYBIND11_MODULE(taichi_python, m) {
  try {
    log_boot("PYD: enter taichi_python module init");
    m.doc() = "taichi_python";

    log_boot("PYD: before InterfaceHolder methods");
    for (auto &kv : InterfaceHolder::get_instance()->methods) {
      kv.second(&m);
    }
    log_boot("PYD: after InterfaceHolder methods");

    log_boot("PYD: before export_lang");
    export_lang(m);
    log_boot("PYD: after export_lang");

    export_math(m);
    log_boot("PYD: after export_math");

    export_misc(m);
    log_boot("PYD: after export_misc");

    export_visual(m);
    log_boot("PYD: after export_visual");

    export_ggui(m);
    log_boot("PYD: after export_ggui");

    log_boot("PYD: module init completed");
  } catch (const std::exception &e) {
    log_boot("PYD: exception during module init");
    log_boot(e.what());
    throw;
  } catch (...) {
    log_boot("PYD: unknown exception during module init");
    throw;
  }
}

}  // namespace taichi
