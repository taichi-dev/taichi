#include "gtest/gtest.h"

#include "taichi/rhi/common/host_memory_pool.h"

namespace taichi::lang {

class HostMemoryPoolTestHelper {
 public:
  static void setDefaultAllocatorSize(std::size_t size) {
    UnifiedAllocator::default_allocator_size = size;
  }
  static size_t getDefaultAllocatorSize() {
    return UnifiedAllocator::default_allocator_size;
  }
};

TEST(HostMemoryPool, AllocateMemory) {
  auto oldAllocatorSize = HostMemoryPoolTestHelper::getDefaultAllocatorSize();
  HostMemoryPoolTestHelper::setDefaultAllocatorSize(102400);  // 100KB

  HostMemoryPool pool;

  void *ptr1 = pool.allocate(1024, 16);
  void *ptr2 = pool.allocate(1024, 16);
  void *ptr3 = pool.allocate(1024, 16);

  EXPECT_NE(ptr1, ptr2);
  EXPECT_NE(ptr1, ptr3);
  EXPECT_NE(ptr2, ptr3);

  EXPECT_EQ((std::size_t)ptr2, (std::size_t)ptr1 + 1024);
  EXPECT_EQ((std::size_t)ptr3, (std::size_t)ptr2 + 1024);

  HostMemoryPoolTestHelper::setDefaultAllocatorSize(oldAllocatorSize);
}

}  // namespace taichi::lang
