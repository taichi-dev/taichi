/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#ifndef MEMORY_ALLOCATOR_H
#define MEMORY_ALLOCATOR_H

#include <cassert>

template <typename T, unsigned int size>
class MemoryAllocator {
 private:
  void *stack[size];
  char pool[size * sizeof(T)];
  int stackTop;

 public:
  MemoryAllocator() {
    for (stackTop = 0; stackTop < size; stackTop++)
      stack[stackTop] = pool + sizeof(T) * stackTop;
  }
  T *Allocate() {
    assert(stackTop != 0);
    return (T *)stack[--stackTop];
  }
  void Dispose(void *p) { stack[stackTop++] = p; }
};

#endif
