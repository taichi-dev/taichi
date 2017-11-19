/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#include <taichi/python/exception.h>

TC_NAMESPACE_BEGIN

void raise_assertion_failure_in_python(const std::string &msg) {
  //throw ExceptionForPython(msg);
}

TC_NAMESPACE_END

TC_EXPORT void taichi_raise_assertion_failure_in_python(const char *msg) {
  taichi::raise_assertion_failure_in_python(std::string(msg));
}
