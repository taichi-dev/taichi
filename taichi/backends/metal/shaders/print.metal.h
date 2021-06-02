#include "taichi/backends/metal/shaders/prolog.h"

#ifdef TI_INSIDE_METAL_CODEGEN

#ifndef TI_METAL_NESTED_INCLUDE
#define METAL_BEGIN_PRINT_DEF constexpr auto kMetalPrintSourceCode =
#define METAL_END_PRINT_DEF ;
#else
#define METAL_BEGIN_PRINT_DEF
#define METAL_END_PRINT_DEF
#endif  // TI_METAL_NESTED_INCLUDE

#else

#include <cstdint>

#define METAL_BEGIN_PRINT_DEF
#define METAL_END_PRINT_DEF

#endif  // TI_INSIDE_METAL_CODEGEN

// Kernel side utils to support print()
//
// Each print() call on each thread generates a PrintMsg to be stored inside the
// print buffer.

// clang-format off
METAL_BEGIN_PRINT_DEF
STR(
    // Each type takes 4 bits to encode, which means we can support a maximum
    // of 16 types. For now, we actually only need 2 bits.
    constant constexpr int kMetalNumBitsPerPrintMsgType = 4;
    constant constexpr int kMetalNumPrintMsgTypePerI32 =
        sizeof(int32_t) * 8 / kMetalNumBitsPerPrintMsgType;
    constant constexpr int kMetalPrintMsgTypeWidthMask =
        ((1 << kMetalNumBitsPerPrintMsgType) - 1);

    [[maybe_unused]] constexpr inline int mtl_compute_num_print_msg_typemasks(
        int num_entries) {
      return (num_entries + kMetalNumPrintMsgTypePerI32 - 1) /
             kMetalNumPrintMsgTypePerI32;
    }

    [[maybe_unused]] constexpr inline int mtl_compute_print_msg_bytes(
        int num_entries) {
      // See PrintMsg's layout for how this is computed.
      const int sz =
          sizeof(int32_t) *
          (1 + mtl_compute_num_print_msg_typemasks(num_entries) + num_entries);
      return sz;
    }

    class PrintMsg {
     public:
      // Data layout:
      //
      // * 1 i32 to record how many args there are
      // * Followed by M i32s, which are the type masks. Each i32 can encode
      // hold up to |kMetalNumPrintMsgTypePerI32| types.
      // * Followed by N i32s, one for each print arg. F32 are encoded to I32.
      // For strings, there is a string table on the host side, so that the
      // kernel only needs to store a I32 string ID.
      enum Type { I32 = 1, U32 = 2, F32 = 3, Str = 4 };

      PrintMsg(device int32_t *buf, int num_entries)
          : mask_buf_(buf),
            data_buf_(buf + mtl_compute_num_print_msg_typemasks(num_entries)) {
      }

      void pm_set_i32(int i, int x) {
        set_entry(i, x, Type::I32);
      }

      void pm_set_u32(int i, uint x) {
        // https://stackoverflow.com/a/21769421/12003165
        const int32_t ix = static_cast<int32_t>(x);
        set_entry(i, ix, Type::U32);
      }

      void pm_set_f32(int i, float x) {
        const int32_t ix = *reinterpret_cast<thread int32_t *>(&x);
        set_entry(i, ix, Type::F32);
      }

      void pm_set_str(int i, int str_id) {
        set_entry(i, str_id, Type::Str);
      }

      Type pm_get_type(int i) {
        const int mask_i = i / kMetalNumPrintMsgTypePerI32;
        const int i_in_mask = i % kMetalNumPrintMsgTypePerI32;
        int mask = mask_buf_[mask_i];
        mask >>= typemask_shift(i_in_mask);
        mask &= kMetalPrintMsgTypeWidthMask;
        return (Type)mask;
      }

      int32_t pm_get_data(int i) {
        return data_buf_[i];
      }

     private:
      void set_entry(int i, int32_t x, Type ty) {
        const int mask_i = i / kMetalNumPrintMsgTypePerI32;
        const int i_in_mask = i % kMetalNumPrintMsgTypePerI32;
        int mask = ((int)ty & kMetalPrintMsgTypeWidthMask);
        mask <<= typemask_shift(i_in_mask);
        mask_buf_[mask_i] |= mask;
        data_buf_[i] = x;
      }

      inline static int typemask_shift(int i_in_mask) {
        return (kMetalNumPrintMsgTypePerI32 - 1 - i_in_mask) *
               kMetalNumBitsPerPrintMsgType;
      }

      device int32_t *mask_buf_;
      device int32_t *data_buf_;
    };

    // This struct is stored in the Metal buffer.
    // The mem space immediately after this struct stores the actual PrintMsg.
    struct AssertRecorderData {
      atomic_int flag;
      int32_t num_args;
    };

    // This is just a lightweight wrapper of AssertRecorderData in each Metal
    // thread. It adds assertion functionality around the wrapped data.
    class AssertRecorder {
     public:
      explicit AssertRecorder(device byte * addr)
          : ac_(reinterpret_cast<device AssertRecorderData *>(addr)) {
      }

      // Returns true if this is the first failure
      bool mark_first_failure() {
        return atomic_exchange_explicit(&(ac_->flag), 1,
                                        metal::memory_order_relaxed) == 0;
      }

      void set_num_args(int n) {
        ac_->num_args = n;
      }

      device int32_t *msg_buf_addr() {
        return reinterpret_cast<device int32_t *>(ac_ + 1);
      }

     private:
      device AssertRecorderData *ac_;
    };

    constant constexpr int kMetalMaxNumAssertArgs = 64;
    // Buffer size of the AssertRecorderData + the actual PrintMsg size for
    // supporting assert().
    //
    // assert() will produce at most one PrintMsg. The assert PrintMsg is
    // assumed to have <= kMetalMaxNumAssertArgs args.
    constant constexpr int kMetalAssertBufferSize =
        sizeof(AssertRecorderData) +
        mtl_compute_print_msg_bytes(kMetalMaxNumAssertArgs);

    struct PrintMsgAllocator { atomic_int next; };

    // 2MB, this stores PrintMsgs for both assert() and print(), as well as
    // the tiny allocator/recorder objects.
    //
    // MetalPrintAssertBuffer memory view:
    //
    // +------------------------+ \
    // | AssertRecorderData     | |
    // +------------------------+ | -> for assert(), kMetalAssertBufferSize
    // | PrintMsg for assert()  | |
    // +------------------------+ /
    // | PrintMsgAllocator      | \
    // +------------------------+ |
    // |                        | |
    // |                        | |
    // | a queue of PrintMsgs   | | -> for print()
    // | for print()            | |
    // | ... ...                | |
    // |                        | |
    // +------------------------+ /
    constant constexpr int kMetalPrintAssertBufferSize = 2 * 1024 * 1024;
    // Space to hold the PrintMsgs. These PrintMsgs are pushed into a queue.
    constant constexpr int kMetalPrintMsgsMaxQueueSize =
        kMetalPrintAssertBufferSize - sizeof(PrintMsgAllocator) -
        kMetalAssertBufferSize;

    [[maybe_unused]] device int32_t *
    mtl_print_alloc_buf(device PrintMsgAllocator *pa, int num_entries) {
      const int sz = mtl_compute_print_msg_bytes(num_entries);
      const int cur = atomic_fetch_add_explicit(&(pa->next), sz,
                                                metal::memory_order_relaxed);
      if (cur + sz >= kMetalPrintMsgsMaxQueueSize) {
        // Avoid buffer overflow
        return (device int32_t *)0;
      }
      device byte *data_begin = reinterpret_cast<device byte *>(pa + 1);
      device int32_t *ptr =
          reinterpret_cast<device int32_t *>(data_begin + cur);
      *ptr = num_entries;
      return (ptr + 1);
    }
)
METAL_END_PRINT_DEF
// clang-format on

#undef METAL_BEGIN_PRINT_DEF
#undef METAL_END_PRINT_DEF

#include "taichi/backends/metal/shaders/epilog.h"
