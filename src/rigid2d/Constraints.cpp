/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#include "Constraints.h"

MemoryAllocator<Contact, 8196> contactMemoryAllocator;

void *Contact::operator new(size_t _) {
  return contactMemoryAllocator.Allocate();
}

void Contact::operator delete(void *p, size_t _) {
  if (p != NULL)
    contactMemoryAllocator.Dispose(p);
}
