/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#include <taichi/util.h>
#include <taichi/testing.h>
#include <taichi/system/virtual_memory.h>

TC_NAMESPACE_BEGIN

TC_TEST("Virtual Memory") {
  for (int i = 0; i < 3; i++) {
    // Allocate 1 TB of virtual memory
    std::size_t size = 1LL << 40;
    VirtualMemoryAllocator vm(size);
    // Touch 512 MB (1 << 29 B)
    for (int j = 0; j < (1 << 29) / (int)VirtualMemoryAllocator::page_size; j++) {
      uint8 val = *((uint8 *)vm.ptr + rand_int64() % size);
      CHECK(val == 0);
    }
  }

}

TC_NAMESPACE_END
