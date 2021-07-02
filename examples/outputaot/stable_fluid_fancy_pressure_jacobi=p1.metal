#include <metal_stdlib>
#include <metal_compute>
using namespace metal;
namespace {
using byte = char;

template <typename T, typename G> T union_cast(G g) { static_assert(sizeof(T) == sizeof(G), "Size mismatch"); return *reinterpret_cast<thread const T *>(&g); } inline int ifloordiv(int lhs, int rhs) { const int intm = (lhs / rhs); return (((lhs < 0) != (rhs < 0) && lhs && (rhs * intm != lhs)) ? (intm - 1) : intm); } int32_t pow_i32(int32_t x, int32_t n) { int32_t tmp = x; int32_t ans = 1; while (n) { if (n & 1) ans *= tmp; tmp *= tmp; n >>= 1; } return ans; } float fatomic_fetch_add(device float *dest, const float operand) { bool ok = false; float old_val = 0.0f; while (!ok) { old_val = *dest; float new_val = (old_val + operand); ok = atomic_compare_exchange_weak_explicit( (device atomic_int *)dest, (thread int *)(&old_val), *((thread int *)(&new_val)), metal::memory_order_relaxed, metal::memory_order_relaxed); } return old_val; } float fatomic_fetch_min(device float *dest, const float operand) { bool ok = false; float old_val = 0.0f; while (!ok) { old_val = *dest; float new_val = (old_val < operand) ? old_val : operand; ok = atomic_compare_exchange_weak_explicit( (device atomic_int *)dest, (thread int *)(&old_val), *((thread int *)(&new_val)), metal::memory_order_relaxed, metal::memory_order_relaxed); } return old_val; } float fatomic_fetch_max(device float *dest, const float operand) { bool ok = false; float old_val = 0.0f; while (!ok) { old_val = *dest; float new_val = (old_val > operand) ? old_val : operand; ok = atomic_compare_exchange_weak_explicit( (device atomic_int *)dest, (thread int *)(&old_val), *((thread int *)(&new_val)), metal::memory_order_relaxed, metal::memory_order_relaxed); } return old_val; } struct RandState { uint32_t seed; }; uint32_t metal_rand_u32(device RandState * state) { device uint *sp = (device uint *)&(state->seed); bool done = false; uint32_t nxt = 0; while (!done) { uint32_t o = *sp; nxt = o * 1103515245 + 12345; done = atomic_compare_exchange_weak_explicit( (device atomic_uint *)sp, &o, nxt, metal::memory_order_relaxed, metal::memory_order_relaxed); } return nxt * 1000000007; } int32_t metal_rand_i32(device RandState * state) { return metal_rand_u32(state); } float metal_rand_f32(device RandState *state) { return metal_rand_u32(state) * (1.0f / 4294967296.0f); }

constant constexpr int kTaichiMaxNumIndices = 8; constant constexpr int kTaichiNumChunks = 1024; constant constexpr int kAlignment = 8; using PtrOffset = int32_t; struct MemoryAllocator { atomic_int next; constant constexpr static int kInitOffset = 8; static inline bool is_valid(PtrOffset v) { return v >= kInitOffset; } }; struct ListManagerData { int32_t element_stride = 0; int32_t log2_num_elems_per_chunk = 0; atomic_int next; atomic_int chunks[kTaichiNumChunks]; struct ReservedElemPtrOffset { public: ReservedElemPtrOffset() = default; explicit ReservedElemPtrOffset(PtrOffset v) : val_(v) { } inline bool is_valid() const { return is_valid(val_); } inline static bool is_valid(PtrOffset v) { return MemoryAllocator::is_valid(v); } inline PtrOffset value() const { return val_; } private: PtrOffset val_{0}; }; }; struct NodeManagerData { using ElemIndex = ListManagerData::ReservedElemPtrOffset; ListManagerData data_list; ListManagerData free_list; ListManagerData recycled_list; atomic_int free_list_used; int recycled_list_size_backup; }; struct SNodeMeta { enum Type { Root = 0, Dense = 1, Bitmasked = 2, Dynamic = 3, Pointer = 4, BitStruct = 5, }; int32_t element_stride = 0; int32_t num_slots = 0; int32_t mem_offset_in_parent = 0; int32_t type = 0; }; struct SNodeExtractors { struct Extractor { int32_t start = 0; int32_t num_bits = 0; int32_t acc_offset = 0; int32_t num_elements = 0; }; Extractor extractors[kTaichiMaxNumIndices]; }; struct ElementCoords { int32_t at[kTaichiMaxNumIndices]; }; struct ListgenElement { ElementCoords coords; int32_t mem_offset = 0; struct BelongedNodeManager { int32_t id = -1; NodeManagerData::ElemIndex elem_idx; }; BelongedNodeManager belonged_nodemgr; inline bool in_root_buffer() const { return belonged_nodemgr.id < 0; } };

struct Runtime {
  SNodeMeta snode_metas[22];
  SNodeExtractors snode_extractors[22];
  ListManagerData snode_lists[22];
  NodeManagerData snode_allocators[22];
  NodeManagerData::ElemIndex ambient_indices[22];
  uint32_t rand_seeds[65536];
};

[[maybe_unused]] PtrOffset mtl_memalloc_alloc(device MemoryAllocator *ma, int32_t size) { size = ((size + kAlignment - 1) / kAlignment) * kAlignment; return atomic_fetch_add_explicit(&ma->next, size, metal::memory_order_relaxed); } [[maybe_unused]] device char *mtl_memalloc_to_ptr(device MemoryAllocator *ma, PtrOffset offs) { return reinterpret_cast<device char *>(ma + 1) + offs; } struct ListManager { using ReservedElemPtrOffset = ListManagerData::ReservedElemPtrOffset; device ListManagerData *lm_data; device MemoryAllocator *mem_alloc; inline int num_active() { return atomic_load_explicit(&(lm_data->next), metal::memory_order_relaxed); } inline void resize(int sz) { atomic_store_explicit(&(lm_data->next), sz, metal::memory_order_relaxed); } inline void clear() { resize(0); } ReservedElemPtrOffset reserve_new_elem() { const int elem_idx = atomic_fetch_add_explicit( &lm_data->next, 1, metal::memory_order_relaxed); const int chunk_idx = get_chunk_index(elem_idx); const PtrOffset chunk_ptr_offs = ensure_chunk(chunk_idx); const auto offset = get_elem_ptr_offs_from_chunk(elem_idx, chunk_ptr_offs); return ReservedElemPtrOffset{offset}; } device char *append() { auto reserved = reserve_new_elem(); return get_ptr(reserved); } template <typename T> void append(thread const T &elem) { device char *ptr = append(); thread char *elem_ptr = (thread char *)(&elem); for (int i = 0; i < lm_data->element_stride; ++i) { *ptr = *elem_ptr; ++ptr; ++elem_ptr; } } device char *get_ptr(ReservedElemPtrOffset offs) { return mtl_memalloc_to_ptr(mem_alloc, offs.value()); } device char *get_ptr(int i) { const int chunk_idx = get_chunk_index(i); const PtrOffset chunk_ptr_offs = atomic_load_explicit( lm_data->chunks + chunk_idx, metal::memory_order_relaxed); return get_elem_from_chunk(i, chunk_ptr_offs); } template <typename T> T get(int i) { return *reinterpret_cast<device T *>(get_ptr(i)); } private: inline int get_chunk_index(int elem_idx) const { return elem_idx >> lm_data->log2_num_elems_per_chunk; } PtrOffset ensure_chunk(int chunk_idx) { PtrOffset offs = 0; const int chunk_bytes = (lm_data->element_stride << lm_data->log2_num_elems_per_chunk); while (true) { int stored = 0; const bool is_me = atomic_compare_exchange_weak_explicit( lm_data->chunks + chunk_idx, &stored, 1, metal::memory_order_relaxed, metal::memory_order_relaxed); if (is_me) { offs = mtl_memalloc_alloc(mem_alloc, chunk_bytes); atomic_store_explicit(lm_data->chunks + chunk_idx, offs, metal::memory_order_relaxed); break; } else if (stored > 1) { offs = stored; break; } } return offs; } PtrOffset get_elem_ptr_offs_from_chunk(int elem_idx, PtrOffset chunk_ptr_offs) { const uint32_t mask = ((1 << lm_data->log2_num_elems_per_chunk) - 1); return chunk_ptr_offs + ((elem_idx & mask) * lm_data->element_stride); } device char *get_elem_from_chunk(int elem_idx, PtrOffset chunk_ptr_offs) { const auto offs = get_elem_ptr_offs_from_chunk(elem_idx, chunk_ptr_offs); return mtl_memalloc_to_ptr(mem_alloc, offs); } }; struct NodeManager { using ElemIndex = NodeManagerData::ElemIndex; device NodeManagerData *nm_data; device MemoryAllocator *mem_alloc; ElemIndex allocate() { ListManager free_list; free_list.lm_data = &(nm_data->free_list); free_list.mem_alloc = mem_alloc; ListManager data_list; data_list.lm_data = &(nm_data->data_list); data_list.mem_alloc = mem_alloc; const int cur_used = atomic_fetch_add_explicit( &(nm_data->free_list_used), 1, metal::memory_order_relaxed); if (cur_used < free_list.num_active()) { return free_list.get<ElemIndex>(cur_used); } return data_list.reserve_new_elem(); } device byte *get(ElemIndex i) { ListManager data_list; data_list.lm_data = &(nm_data->data_list); data_list.mem_alloc = mem_alloc; return data_list.get_ptr(i); } void recycle(ElemIndex i) { ListManager recycled_list; recycled_list.lm_data = &(nm_data->recycled_list); recycled_list.mem_alloc = mem_alloc; recycled_list.append(i); } }; class SNodeRep_dense { public: void init(device byte * addr) { addr_ = addr; } inline device byte *addr() { return addr_; } inline bool is_active(int) { return true; } inline void activate(int) { } inline void deactivate(int) { } private: device byte *addr_ = nullptr; }; using SNodeRep_root = SNodeRep_dense; class SNodeRep_bitmasked { public: constant static constexpr int kBitsPerMask = (sizeof(uint32_t) * 8); void init(device byte * addr, int meta_offset) { addr_ = addr; meta_offset_ = meta_offset; } inline device byte *addr() { return addr_; } bool is_active(int i) { device auto *ptr = to_bitmask_ptr(i); uint32_t bits = atomic_load_explicit(ptr, metal::memory_order_relaxed); return ((bits >> (i % kBitsPerMask)) & 1); } void activate(int i) { device auto *ptr = to_bitmask_ptr(i); const uint32_t mask = (1 << (i % kBitsPerMask)); atomic_fetch_or_explicit(ptr, mask, metal::memory_order_relaxed); } void deactivate(int i) { device auto *ptr = to_bitmask_ptr(i); const uint32_t mask = ~(1 << (i % kBitsPerMask)); atomic_fetch_and_explicit(ptr, mask, metal::memory_order_relaxed); } private: inline device atomic_uint *to_bitmask_ptr(int i) { return reinterpret_cast<device atomic_uint *>(addr_ + meta_offset_) + (i / kBitsPerMask); } device byte *addr_ = nullptr; int32_t meta_offset_ = 0; }; class SNodeRep_dynamic { public: void init(device byte * addr, int meta_offset) { addr_ = addr; meta_offset_ = meta_offset; } inline device byte *addr() { return addr_; } bool is_active(int i) { const auto n = atomic_load_explicit(to_meta_ptr(), metal::memory_order_relaxed); return i < n; } void activate(int i) { device auto *ptr = to_meta_ptr(); atomic_fetch_max_explicit(ptr, (i + 1), metal::memory_order_relaxed); return; } void deactivate() { device auto *ptr = to_meta_ptr(); atomic_store_explicit(ptr, 0, metal::memory_order_relaxed); } int append(int32_t data) { device auto *ptr = to_meta_ptr(); int me = atomic_fetch_add_explicit(ptr, 1, metal::memory_order_relaxed); *(reinterpret_cast<device int32_t *>(addr_) + me) = data; return me; } int length() { return atomic_load_explicit(to_meta_ptr(), metal::memory_order_relaxed); } private: inline device atomic_int *to_meta_ptr() { return reinterpret_cast<device atomic_int *>(addr_ + meta_offset_); } device byte *addr_ = nullptr; int32_t meta_offset_ = 0; }; class SNodeRep_pointer { public: using ElemIndex = NodeManagerData::ElemIndex; void init(device byte * addr, NodeManager nm, ElemIndex ambient_idx) { addr_ = addr; nm_ = nm; ambient_idx_ = ambient_idx; } device byte *child_or_ambient_addr(int i) { auto nm_idx = to_nodemgr_idx(addr_, i); nm_idx = nm_idx.is_valid() ? nm_idx : ambient_idx_; return nm_.get(nm_idx); } inline bool is_active(int i) { return is_active(addr_, i); } void activate(int i) { device auto *nm_idx_ptr = to_nodemgr_idx_ptr(addr_, i); auto nm_idx_val = atomic_load_explicit(nm_idx_ptr, metal::memory_order_relaxed); while (!ElemIndex::is_valid(nm_idx_val)) { nm_idx_val = 0; const bool is_me = atomic_compare_exchange_weak_explicit( nm_idx_ptr, &nm_idx_val, 1, metal::memory_order_relaxed, metal::memory_order_relaxed); if (is_me) { nm_idx_val = nm_.allocate().value(); atomic_store_explicit(nm_idx_ptr, nm_idx_val, metal::memory_order_relaxed); break; } else if (ElemIndex::is_valid(nm_idx_val)) { break; } } } void deactivate(int i) { device auto *nm_idx_ptr = to_nodemgr_idx_ptr(addr_, i); const auto old_nm_idx_val = atomic_exchange_explicit( nm_idx_ptr, 0, metal::memory_order_relaxed); const auto old_nm_idx = ElemIndex(old_nm_idx_val); if (!old_nm_idx.is_valid()) { return; } nm_.recycle(old_nm_idx); } static inline device atomic_int *to_nodemgr_idx_ptr(device byte * addr, int ch_i) { return reinterpret_cast<device atomic_int *>(addr + ch_i * sizeof(ElemIndex)); } static inline ElemIndex to_nodemgr_idx(device byte * addr, int ch_i) { device auto *ptr = to_nodemgr_idx_ptr(addr, ch_i); const auto v = atomic_load_explicit(ptr, metal::memory_order_relaxed); return ElemIndex(v); } static bool is_active(device byte * addr, int ch_i) { return to_nodemgr_idx(addr, ch_i).is_valid(); } private: device byte *addr_; NodeManager nm_; ElemIndex ambient_idx_; }; [[maybe_unused]] int is_active(device byte *addr, SNodeMeta meta, int i) { if (meta.type == SNodeMeta::Root || meta.type == SNodeMeta::Dense) { return true; } else if (meta.type == SNodeMeta::Dynamic) { SNodeRep_dynamic rep; rep.init(addr, meta.num_slots * meta.element_stride); return rep.is_active(i); } else if (meta.type == SNodeMeta::Bitmasked) { SNodeRep_bitmasked rep; rep.init(addr, meta.num_slots * meta.element_stride); return rep.is_active(i); } else if (meta.type == SNodeMeta::Pointer) { return SNodeRep_pointer::is_active(addr, i); } return false; } [[maybe_unused]] void refine_coordinates( thread const ElementCoords &parent, device const SNodeExtractors &child_extrators, int l, thread ElementCoords *child) { for (int i = 0; i < kTaichiMaxNumIndices; ++i) { device const auto &ex = child_extrators.extractors[i]; const int mask = ((1 << ex.num_bits) - 1); const int addition = ((l >> ex.acc_offset) & mask); child->at[i] = ((parent.at[i] << ex.num_bits) | addition); } } [[maybe_unused]] device byte *mtl_lgen_snode_addr( thread const ListgenElement &lgen, device byte *root_addr, device Runtime *rtm, device MemoryAllocator *mem_alloc) { if (lgen.in_root_buffer()) { return root_addr + lgen.mem_offset; } NodeManager nm; nm.nm_data = (rtm->snode_allocators + lgen.belonged_nodemgr.id); nm.mem_alloc = mem_alloc; device byte *addr = nm.get(lgen.belonged_nodemgr.elem_idx); return addr + lgen.mem_offset; } [[maybe_unused]] void run_gc_compact_free_list( device NodeManagerData *nm_data, device MemoryAllocator *mem_alloc, const int tid, const int grid_size) { NodeManager nm; nm.nm_data = nm_data; nm.mem_alloc = mem_alloc; ListManager free_list; free_list.lm_data = &(nm.nm_data->free_list); free_list.mem_alloc = nm.mem_alloc; const int free_size = free_list.num_active(); const int free_used = atomic_load_explicit(&(nm.nm_data->free_list_used), metal::memory_order_relaxed); int num_to_copy = 0; if (free_used * 2 > free_size) { num_to_copy = free_size - free_used; } else { num_to_copy = free_used; } const int offs = free_size - num_to_copy; using ElemIndex = NodeManager::ElemIndex; for (int ii = tid; ii < num_to_copy; ii += grid_size) { device auto *dest = reinterpret_cast<device ElemIndex *>(free_list.get_ptr(ii)); *dest = free_list.get<ElemIndex>(ii + offs); } } [[maybe_unused]] void run_gc_reset_free_list( device NodeManagerData *nm_data, device MemoryAllocator *mem_alloc) { NodeManager nm; nm.nm_data = nm_data; nm.mem_alloc = mem_alloc; ListManager free_list; free_list.lm_data = &(nm.nm_data->free_list); free_list.mem_alloc = nm.mem_alloc; const int free_size = free_list.num_active(); const int free_used = atomic_exchange_explicit( &(nm.nm_data->free_list_used), 0, metal::memory_order_relaxed); int free_remaining = free_size - free_used; free_remaining = free_remaining > 0 ? free_remaining : 0; free_list.resize(free_remaining); nm.nm_data->recycled_list_size_backup = atomic_exchange_explicit( &(nm.nm_data->recycled_list.next), 0, metal::memory_order_relaxed); } struct GCMoveRecycledToFreeThreadParams { int thread_position_in_threadgroup; int threadgroup_position_in_grid; int threadgroups_per_grid; int threads_per_threadgroup; }; [[maybe_unused]] void run_gc_move_recycled_to_free( device NodeManagerData *nm_data, device MemoryAllocator *mem_alloc, thread const GCMoveRecycledToFreeThreadParams &thparams) { NodeManager nm; nm.nm_data = nm_data; nm.mem_alloc = mem_alloc; ListManager free_list; free_list.lm_data = &(nm.nm_data->free_list); free_list.mem_alloc = nm.mem_alloc; ListManager recycled_list; recycled_list.lm_data = &(nm.nm_data->recycled_list); recycled_list.mem_alloc = nm.mem_alloc; ListManager data_list; data_list.lm_data = &(nm.nm_data->data_list); data_list.mem_alloc = nm.mem_alloc; const int kInt32Stride = sizeof(int32_t); const int recycled_size = nm.nm_data->recycled_list_size_backup; using ElemIndex = NodeManager::ElemIndex; for (int ii = thparams.threadgroup_position_in_grid; ii < recycled_size; ii += thparams.threadgroups_per_grid) { const auto elem_idx = recycled_list.get<ElemIndex>(ii); device char *ptr = nm.get(elem_idx); device const char *ptr_end = ptr + data_list.lm_data->element_stride; const int ptr_mod = ((int64_t)(ptr) % kInt32Stride); if (ptr_mod) { device char *new_ptr = ptr + kInt32Stride - ptr_mod; if (thparams.thread_position_in_threadgroup == 0) { for (device char *p = ptr; p < new_ptr; ++p) { *p = 0; } } ptr = new_ptr; } ptr += (thparams.thread_position_in_threadgroup * kInt32Stride); while ((ptr + kInt32Stride) <= ptr_end) { *reinterpret_cast<device int32_t *>(ptr) = 0; ptr += (kInt32Stride * thparams.threads_per_threadgroup); } while (ptr < ptr_end) { *ptr = 0; ++ptr; } if (thparams.thread_position_in_threadgroup == 0) { free_list.append(elem_idx); } } }

struct SNodeBitPointer { device uint32_t *base; uint32_t offset; SNodeBitPointer(device byte * b, uint32_t o) : base((device uint32_t *)b), offset(o) { } }; template <typename C> C mtl_float_to_custom_int(float f) { const int32_t delta_bits = (union_cast<int32_t>(f) & 0x80000000) | union_cast<int32_t>(0.5f); const float delta = union_cast<float>(delta_bits); return static_cast<C>(f + delta); } void mtl_set_partial_bits(SNodeBitPointer bp, uint32_t value, uint32_t bits) { using P = uint32_t; constexpr int N = sizeof(P) * 8; const uint32_t mask = ((~(uint32_t)0U) << (N - bits)) >> (N - bp.offset - bits); device auto *atm_ptr = reinterpret_cast<device atomic_uint *>(bp.base); bool ok = false; while (!ok) { P old_val = *(bp.base); P new_val = (old_val & (~mask)) | (value << bp.offset); ok = atomic_compare_exchange_weak_explicit(atm_ptr, &old_val, new_val, metal::memory_order_relaxed, metal::memory_order_relaxed); } } void mtl_set_full_bits(SNodeBitPointer bp, uint32_t value) { device auto *atm_ptr = reinterpret_cast<device atomic_uint *>(bp.base); atomic_store_explicit(atm_ptr, value, metal::memory_order_relaxed); } uint32_t mtl_atomic_add_partial_bits(SNodeBitPointer bp, uint32_t value, uint32_t bits) { using P = uint32_t; constexpr int N = sizeof(P) * 8; const uint32_t mask = ((~(uint32_t)0U) << (N - bits)) >> (N - bp.offset - bits); device auto *atm_ptr = reinterpret_cast<device atomic_uint *>(bp.base); P old_val = 0; bool ok = false; while (!ok) { old_val = *(bp.base); P new_val = old_val + (value << bp.offset); new_val = (old_val & (~mask)) | (new_val & mask); ok = atomic_compare_exchange_weak_explicit(atm_ptr, &old_val, new_val, metal::memory_order_relaxed, metal::memory_order_relaxed); } return old_val; } uint32_t mtl_atomic_add_full_bits(SNodeBitPointer bp, uint32_t value) { device auto *atm_ptr = reinterpret_cast<device atomic_uint *>(bp.base); return atomic_fetch_add_explicit(atm_ptr, value, metal::memory_order_relaxed); } namespace detail { template <bool Signed> struct SHRSelector { using type = int32_t; }; template <> struct SHRSelector<false> { using type = uint32_t; }; } template <typename C> C mtl_get_partial_bits(SNodeBitPointer bp, uint32_t bits) { using P = uint32_t; constexpr int N = sizeof(P) * 8; const P phy_val = *(bp.base); using CSel = typename detail::SHRSelector<is_signed<C>::value>::type; const auto step1 = static_cast<CSel>(phy_val << (N - (bp.offset + bits))); return static_cast<C>(step1 >> (N - bits)); } template <typename C> C mtl_get_full_bits(SNodeBitPointer bp) { return static_cast<C>(*(bp.base)); }




struct S22 {
  // place
  constant static constexpr int stride = sizeof(float);

  S22(device byte *v, device Runtime *, device MemoryAllocator *)
    : val((device float*)v) {}

  device float *val;
};

class S21_ch {
 public:
  S21_ch(device byte *a) : addr_(a) {}
  S22 get0(device Runtime *rtm, device MemoryAllocator *ma) {
    return {addr_ + (0), rtm, ma};
  }

  device byte *addr() { return addr_; }

  constant static constexpr int stride = 0 + S22::stride;
 private:
  device byte *addr_;
};

struct S21 {
  // dense
  constant static constexpr int n = 524288;
  constant static constexpr int elem_stride = S21_ch::stride;
  constant static constexpr int stride = elem_stride * n;

  S21(device byte *addr, device Runtime *rtm, device MemoryAllocator *ma) {
    rep_.init(addr);
  }

  S21_ch children(int i) {
    return {rep_.addr() + (i * elem_stride)};
  }

  inline bool is_active(int i) {
    return rep_.is_active(i);
  }

  inline void activate(int i) {
    rep_.activate(i);
  }

  inline void deactivate(int i) {
    rep_.deactivate(i);
  }

 private:
  SNodeRep_dense rep_;
};


struct S20 {
  // place
  constant static constexpr int stride = sizeof(float);

  S20(device byte *v, device Runtime *, device MemoryAllocator *)
    : val((device float*)v) {}

  device float *val;
};

class S19_ch {
 public:
  S19_ch(device byte *a) : addr_(a) {}
  S20 get0(device Runtime *rtm, device MemoryAllocator *ma) {
    return {addr_ + (0), rtm, ma};
  }

  device byte *addr() { return addr_; }

  constant static constexpr int stride = 0 + S20::stride;
 private:
  device byte *addr_;
};

struct S19 {
  // dense
  constant static constexpr int n = 524288;
  constant static constexpr int elem_stride = S19_ch::stride;
  constant static constexpr int stride = elem_stride * n;

  S19(device byte *addr, device Runtime *rtm, device MemoryAllocator *ma) {
    rep_.init(addr);
  }

  S19_ch children(int i) {
    return {rep_.addr() + (i * elem_stride)};
  }

  inline bool is_active(int i) {
    return rep_.is_active(i);
  }

  inline void activate(int i) {
    rep_.activate(i);
  }

  inline void deactivate(int i) {
    rep_.deactivate(i);
  }

 private:
  SNodeRep_dense rep_;
};


struct S18 {
  // place
  constant static constexpr int stride = sizeof(float);

  S18(device byte *v, device Runtime *, device MemoryAllocator *)
    : val((device float*)v) {}

  device float *val;
};

class S17_ch {
 public:
  S17_ch(device byte *a) : addr_(a) {}
  S18 get0(device Runtime *rtm, device MemoryAllocator *ma) {
    return {addr_ + (0), rtm, ma};
  }

  device byte *addr() { return addr_; }

  constant static constexpr int stride = 0 + S18::stride;
 private:
  device byte *addr_;
};

struct S17 {
  // dense
  constant static constexpr int n = 524288;
  constant static constexpr int elem_stride = S17_ch::stride;
  constant static constexpr int stride = elem_stride * n;

  S17(device byte *addr, device Runtime *rtm, device MemoryAllocator *ma) {
    rep_.init(addr);
  }

  S17_ch children(int i) {
    return {rep_.addr() + (i * elem_stride)};
  }

  inline bool is_active(int i) {
    return rep_.is_active(i);
  }

  inline void activate(int i) {
    rep_.activate(i);
  }

  inline void deactivate(int i) {
    rep_.deactivate(i);
  }

 private:
  SNodeRep_dense rep_;
};


struct S16 {
  // place
  constant static constexpr int stride = sizeof(float);

  S16(device byte *v, device Runtime *, device MemoryAllocator *)
    : val((device float*)v) {}

  device float *val;
};

class S15_ch {
 public:
  S15_ch(device byte *a) : addr_(a) {}
  S16 get0(device Runtime *rtm, device MemoryAllocator *ma) {
    return {addr_ + (0), rtm, ma};
  }

  device byte *addr() { return addr_; }

  constant static constexpr int stride = 0 + S16::stride;
 private:
  device byte *addr_;
};

struct S15 {
  // dense
  constant static constexpr int n = 524288;
  constant static constexpr int elem_stride = S15_ch::stride;
  constant static constexpr int stride = elem_stride * n;

  S15(device byte *addr, device Runtime *rtm, device MemoryAllocator *ma) {
    rep_.init(addr);
  }

  S15_ch children(int i) {
    return {rep_.addr() + (i * elem_stride)};
  }

  inline bool is_active(int i) {
    return rep_.is_active(i);
  }

  inline void activate(int i) {
    rep_.activate(i);
  }

  inline void deactivate(int i) {
    rep_.deactivate(i);
  }

 private:
  SNodeRep_dense rep_;
};


struct S14 {
  // place
  constant static constexpr int stride = sizeof(float);

  S14(device byte *v, device Runtime *, device MemoryAllocator *)
    : val((device float*)v) {}

  device float *val;
};


struct S13 {
  // place
  constant static constexpr int stride = sizeof(float);

  S13(device byte *v, device Runtime *, device MemoryAllocator *)
    : val((device float*)v) {}

  device float *val;
};

class S12_ch {
 public:
  S12_ch(device byte *a) : addr_(a) {}
  S13 get0(device Runtime *rtm, device MemoryAllocator *ma) {
    return {addr_ + (0), rtm, ma};
  }

  S14 get1(device Runtime *rtm, device MemoryAllocator *ma) {
    return {addr_ + (0 + S13::stride), rtm, ma};
  }

  device byte *addr() { return addr_; }

  constant static constexpr int stride = 0 + S13::stride + S14::stride;
 private:
  device byte *addr_;
};

struct S12 {
  // dense
  constant static constexpr int n = 524288;
  constant static constexpr int elem_stride = S12_ch::stride;
  constant static constexpr int stride = elem_stride * n;

  S12(device byte *addr, device Runtime *rtm, device MemoryAllocator *ma) {
    rep_.init(addr);
  }

  S12_ch children(int i) {
    return {rep_.addr() + (i * elem_stride)};
  }

  inline bool is_active(int i) {
    return rep_.is_active(i);
  }

  inline void activate(int i) {
    rep_.activate(i);
  }

  inline void deactivate(int i) {
    rep_.deactivate(i);
  }

 private:
  SNodeRep_dense rep_;
};


struct S11 {
  // place
  constant static constexpr int stride = sizeof(float);

  S11(device byte *v, device Runtime *, device MemoryAllocator *)
    : val((device float*)v) {}

  device float *val;
};


struct S10 {
  // place
  constant static constexpr int stride = sizeof(float);

  S10(device byte *v, device Runtime *, device MemoryAllocator *)
    : val((device float*)v) {}

  device float *val;
};

class S9_ch {
 public:
  S9_ch(device byte *a) : addr_(a) {}
  S10 get0(device Runtime *rtm, device MemoryAllocator *ma) {
    return {addr_ + (0), rtm, ma};
  }

  S11 get1(device Runtime *rtm, device MemoryAllocator *ma) {
    return {addr_ + (0 + S10::stride), rtm, ma};
  }

  device byte *addr() { return addr_; }

  constant static constexpr int stride = 0 + S10::stride + S11::stride;
 private:
  device byte *addr_;
};

struct S9 {
  // dense
  constant static constexpr int n = 524288;
  constant static constexpr int elem_stride = S9_ch::stride;
  constant static constexpr int stride = elem_stride * n;

  S9(device byte *addr, device Runtime *rtm, device MemoryAllocator *ma) {
    rep_.init(addr);
  }

  S9_ch children(int i) {
    return {rep_.addr() + (i * elem_stride)};
  }

  inline bool is_active(int i) {
    return rep_.is_active(i);
  }

  inline void activate(int i) {
    rep_.activate(i);
  }

  inline void deactivate(int i) {
    rep_.deactivate(i);
  }

 private:
  SNodeRep_dense rep_;
};


struct S8 {
  // place
  constant static constexpr int stride = sizeof(float);

  S8(device byte *v, device Runtime *, device MemoryAllocator *)
    : val((device float*)v) {}

  device float *val;
};


struct S7 {
  // place
  constant static constexpr int stride = sizeof(float);

  S7(device byte *v, device Runtime *, device MemoryAllocator *)
    : val((device float*)v) {}

  device float *val;
};


struct S6 {
  // place
  constant static constexpr int stride = sizeof(float);

  S6(device byte *v, device Runtime *, device MemoryAllocator *)
    : val((device float*)v) {}

  device float *val;
};

class S5_ch {
 public:
  S5_ch(device byte *a) : addr_(a) {}
  S6 get0(device Runtime *rtm, device MemoryAllocator *ma) {
    return {addr_ + (0), rtm, ma};
  }

  S7 get1(device Runtime *rtm, device MemoryAllocator *ma) {
    return {addr_ + (0 + S6::stride), rtm, ma};
  }

  S8 get2(device Runtime *rtm, device MemoryAllocator *ma) {
    return {addr_ + (0 + S6::stride + S7::stride), rtm, ma};
  }

  device byte *addr() { return addr_; }

  constant static constexpr int stride = 0 + S6::stride + S7::stride + S8::stride;
 private:
  device byte *addr_;
};

struct S5 {
  // dense
  constant static constexpr int n = 8388608;
  constant static constexpr int elem_stride = S5_ch::stride;
  constant static constexpr int stride = elem_stride * n;

  S5(device byte *addr, device Runtime *rtm, device MemoryAllocator *ma) {
    rep_.init(addr);
  }

  S5_ch children(int i) {
    return {rep_.addr() + (i * elem_stride)};
  }

  inline bool is_active(int i) {
    return rep_.is_active(i);
  }

  inline void activate(int i) {
    rep_.activate(i);
  }

  inline void deactivate(int i) {
    rep_.deactivate(i);
  }

 private:
  SNodeRep_dense rep_;
};


struct S4 {
  // place
  constant static constexpr int stride = sizeof(float);

  S4(device byte *v, device Runtime *, device MemoryAllocator *)
    : val((device float*)v) {}

  device float *val;
};


struct S3 {
  // place
  constant static constexpr int stride = sizeof(float);

  S3(device byte *v, device Runtime *, device MemoryAllocator *)
    : val((device float*)v) {}

  device float *val;
};


struct S2 {
  // place
  constant static constexpr int stride = sizeof(float);

  S2(device byte *v, device Runtime *, device MemoryAllocator *)
    : val((device float*)v) {}

  device float *val;
};

class S1_ch {
 public:
  S1_ch(device byte *a) : addr_(a) {}
  S2 get0(device Runtime *rtm, device MemoryAllocator *ma) {
    return {addr_ + (0), rtm, ma};
  }

  S3 get1(device Runtime *rtm, device MemoryAllocator *ma) {
    return {addr_ + (0 + S2::stride), rtm, ma};
  }

  S4 get2(device Runtime *rtm, device MemoryAllocator *ma) {
    return {addr_ + (0 + S2::stride + S3::stride), rtm, ma};
  }

  device byte *addr() { return addr_; }

  constant static constexpr int stride = 0 + S2::stride + S3::stride + S4::stride;
 private:
  device byte *addr_;
};

struct S1 {
  // dense
  constant static constexpr int n = 8388608;
  constant static constexpr int elem_stride = S1_ch::stride;
  constant static constexpr int stride = elem_stride * n;

  S1(device byte *addr, device Runtime *rtm, device MemoryAllocator *ma) {
    rep_.init(addr);
  }

  S1_ch children(int i) {
    return {rep_.addr() + (i * elem_stride)};
  }

  inline bool is_active(int i) {
    return rep_.is_active(i);
  }

  inline void activate(int i) {
    rep_.activate(i);
  }

  inline void deactivate(int i) {
    rep_.deactivate(i);
  }

 private:
  SNodeRep_dense rep_;
};

class S0_ch {
 public:
  S0_ch(device byte *a) : addr_(a) {}
  S1 get0(device Runtime *rtm, device MemoryAllocator *ma) {
    return {addr_ + (0), rtm, ma};
  }

  S5 get1(device Runtime *rtm, device MemoryAllocator *ma) {
    return {addr_ + (0 + S1::stride), rtm, ma};
  }

  S9 get2(device Runtime *rtm, device MemoryAllocator *ma) {
    return {addr_ + (0 + S1::stride + S5::stride), rtm, ma};
  }

  S12 get3(device Runtime *rtm, device MemoryAllocator *ma) {
    return {addr_ + (0 + S1::stride + S5::stride + S9::stride), rtm, ma};
  }

  S15 get4(device Runtime *rtm, device MemoryAllocator *ma) {
    return {addr_ + (0 + S1::stride + S5::stride + S9::stride + S12::stride), rtm, ma};
  }

  S17 get5(device Runtime *rtm, device MemoryAllocator *ma) {
    return {addr_ + (0 + S1::stride + S5::stride + S9::stride + S12::stride + S15::stride), rtm, ma};
  }

  S19 get6(device Runtime *rtm, device MemoryAllocator *ma) {
    return {addr_ + (0 + S1::stride + S5::stride + S9::stride + S12::stride + S15::stride + S17::stride), rtm, ma};
  }

  S21 get7(device Runtime *rtm, device MemoryAllocator *ma) {
    return {addr_ + (0 + S1::stride + S5::stride + S9::stride + S12::stride + S15::stride + S17::stride + S19::stride), rtm, ma};
  }

  device byte *addr() { return addr_; }

  constant static constexpr int stride = 0 + S1::stride + S5::stride + S9::stride + S12::stride + S15::stride + S17::stride + S19::stride + S21::stride;
 private:
  device byte *addr_;
};

struct S0 {
  // root
  constant static constexpr int n = 1;
  constant static constexpr int elem_stride = S0_ch::stride;
  constant static constexpr int stride = elem_stride * n;

  S0(device byte *addr) {
    rep_.init(addr);
  }

  S0_ch children(int i) {
    return {rep_.addr() + (i * elem_stride)};
  }

  inline bool is_active(int i) {
    return rep_.is_active(i);
  }

  inline void activate(int i) {
    rep_.activate(i);
  }

  inline void deactivate(int i) {
    rep_.deactivate(i);
  }

 private:
  SNodeRep_root rep_;
};



using AdStackPtr = thread byte *; inline thread uint32_t * mtl_ad_stack_n(AdStackPtr stack) { return reinterpret_cast<thread uint32_t *>(stack); } inline AdStackPtr mtl_ad_stack_data(AdStackPtr stack) { return stack + sizeof(uint32_t); } inline void mtl_ad_stack_init(AdStackPtr stack) { *mtl_ad_stack_n(stack) = 0; } inline AdStackPtr mtl_ad_stack_top_primal(AdStackPtr stack, int element_size) { const auto n = *mtl_ad_stack_n(stack); return mtl_ad_stack_data(stack) + (n - 1) * 2 * element_size; } inline AdStackPtr mtl_ad_stack_top_adjoint(AdStackPtr stack, int element_size) { return mtl_ad_stack_top_primal(stack, element_size) + element_size; } inline void mtl_ad_stack_pop(AdStackPtr stack) { thread auto &n = *mtl_ad_stack_n(stack); --n; } void mtl_ad_stack_push(AdStackPtr stack, int element_size) { thread auto &n = *mtl_ad_stack_n(stack); ++n; AdStackPtr data = mtl_ad_stack_top_primal(stack, element_size); for (int i = 0; i < element_size * 2; ++i) { data[i] = 0; } }

constant constexpr int kMetalNumBitsPerPrintMsgType = 4; constant constexpr int kMetalNumPrintMsgTypePerI32 = sizeof(int32_t) * 8 / kMetalNumBitsPerPrintMsgType; constant constexpr int kMetalPrintMsgTypeWidthMask = ((1 << kMetalNumBitsPerPrintMsgType) - 1); [[maybe_unused]] constexpr inline int mtl_compute_num_print_msg_typemasks( int num_entries) { return (num_entries + kMetalNumPrintMsgTypePerI32 - 1) / kMetalNumPrintMsgTypePerI32; } [[maybe_unused]] constexpr inline int mtl_compute_print_msg_bytes( int num_entries) { const int sz = sizeof(int32_t) * (1 + mtl_compute_num_print_msg_typemasks(num_entries) + num_entries); return sz; } class PrintMsg { public: enum Type { I32 = 1, U32 = 2, F32 = 3, Str = 4 }; PrintMsg(device int32_t *buf, int num_entries) : mask_buf_(buf), data_buf_(buf + mtl_compute_num_print_msg_typemasks(num_entries)) { } void pm_set_i32(int i, int x) { set_entry(i, x, Type::I32); } void pm_set_u32(int i, uint x) { const int32_t ix = static_cast<int32_t>(x); set_entry(i, ix, Type::U32); } void pm_set_f32(int i, float x) { const int32_t ix = *reinterpret_cast<thread int32_t *>(&x); set_entry(i, ix, Type::F32); } void pm_set_str(int i, int str_id) { set_entry(i, str_id, Type::Str); } Type pm_get_type(int i) { const int mask_i = i / kMetalNumPrintMsgTypePerI32; const int i_in_mask = i % kMetalNumPrintMsgTypePerI32; int mask = mask_buf_[mask_i]; mask >>= typemask_shift(i_in_mask); mask &= kMetalPrintMsgTypeWidthMask; return (Type)mask; } int32_t pm_get_data(int i) { return data_buf_[i]; } private: void set_entry(int i, int32_t x, Type ty) { const int mask_i = i / kMetalNumPrintMsgTypePerI32; const int i_in_mask = i % kMetalNumPrintMsgTypePerI32; int mask = ((int)ty & kMetalPrintMsgTypeWidthMask); mask <<= typemask_shift(i_in_mask); mask_buf_[mask_i] |= mask; data_buf_[i] = x; } inline static int typemask_shift(int i_in_mask) { return (kMetalNumPrintMsgTypePerI32 - 1 - i_in_mask) * kMetalNumBitsPerPrintMsgType; } device int32_t *mask_buf_; device int32_t *data_buf_; }; struct AssertRecorderData { atomic_int flag; int32_t num_args; }; class AssertRecorder { public: explicit AssertRecorder(device byte * addr) : ac_(reinterpret_cast<device AssertRecorderData *>(addr)) { } bool mark_first_failure() { return atomic_exchange_explicit(&(ac_->flag), 1, metal::memory_order_relaxed) == 0; } void set_num_args(int n) { ac_->num_args = n; } device int32_t *msg_buf_addr() { return reinterpret_cast<device int32_t *>(ac_ + 1); } private: device AssertRecorderData *ac_; }; constant constexpr int kMetalMaxNumAssertArgs = 64; constant constexpr int kMetalAssertBufferSize = sizeof(AssertRecorderData) + mtl_compute_print_msg_bytes(kMetalMaxNumAssertArgs); struct PrintMsgAllocator { atomic_int next; }; constant constexpr int kMetalPrintAssertBufferSize = 2 * 1024 * 1024; constant constexpr int kMetalPrintMsgsMaxQueueSize = kMetalPrintAssertBufferSize - sizeof(PrintMsgAllocator) - kMetalAssertBufferSize; [[maybe_unused]] device int32_t * mtl_print_alloc_buf(device PrintMsgAllocator *pa, int num_entries) { const int sz = mtl_compute_print_msg_bytes(num_entries); const int cur = atomic_fetch_add_explicit(&(pa->next), sz, metal::memory_order_relaxed); if (cur + sz >= kMetalPrintMsgsMaxQueueSize) { return (device int32_t *)0; } device byte *data_begin = reinterpret_cast<device byte *>(pa + 1); device int32_t *ptr = reinterpret_cast<device int32_t *>(data_begin + cur); *ptr = num_entries; return (ptr + 1); }

void mtl_k0015_pressure_jacobi_c14_1_0_func(
    device byte* root_addr,
    device byte* global_tmps_addr,
    device byte* runtime_addr,
    device byte* print_assert_addr,
    const int linear_loop_idx_) {
  device auto *runtime_ = reinterpret_cast<device Runtime *>(runtime_addr);
  device auto *mem_alloc_ = reinterpret_cast<device MemoryAllocator *>(runtime_ + 1);
  device RandState* rand_state_ = reinterpret_cast<device RandState*>(runtime_->rand_seeds + (linear_loop_idx_ % 65536));
  AssertRecorder assert_rec_(print_assert_addr);
  device auto* print_alloc_ = reinterpret_cast<device PrintMsgAllocator*>(print_assert_addr + 300);
  constexpr int32_t tmp13568 = 1;
  constexpr int32_t tmp13562 = -1;
  constexpr int32_t tmp13552 = 1023;
  constexpr int32_t tmp13548 = 511;
  constexpr float tmp56 = 0.25;
  constexpr float tmp53 = -2.8057652e-06;
  constexpr int32_t tmp26 = 596;
  constexpr int32_t tmp24 = 0;
  constexpr int32_t tmp22 = 416;
  const int tmp3 = linear_loop_idx_;
  constexpr int32_t tmp13498 = 10;
  const int32_t tmp13499 = (tmp3 >> tmp13498);
  const int32_t tmp13501 = (tmp13499 & tmp13548);
  const int32_t tmp13505 = (tmp3 & tmp13552);
  constexpr int32_t tmp13 = 417;
  const int32_t tmp14 = -(tmp13501 < tmp13);
  constexpr int32_t tmp16 = 597;
  const int32_t tmp17 = -(tmp13505 < tmp16);
  const int32_t tmp18 = (tmp14 & tmp17);
  if (tmp18) {
    const int32_t tmp13555 = (tmp13501 + tmp13562);
    const int32_t tmp23 =  min(tmp22, tmp13555);
    const int32_t tmp25 =  max(tmp24, tmp23);
    const int32_t tmp27 =  min(tmp26, tmp13505);
    const int32_t tmp28 =  max(tmp24, tmp27);
    S0 tmp13428(root_addr);
    S0_ch tmp13430 = tmp13428.children(tmp24);
    S21 tmp13431 = tmp13430.get7(runtime_, mem_alloc_);
    const int32_t tmp13509 = (tmp25 & tmp13548);
    const int32_t tmp13513 = (tmp28 & tmp13552);
    const int32_t tmp13621 = (tmp13509 << tmp13498);
    const int32_t tmp13573 = (tmp13513 + tmp13621);
    S21_ch tmp13435 = tmp13431.children(tmp13573);
    device float* tmp13436 = tmp13435.get0(runtime_, mem_alloc_).val;
    const auto tmp30 = *tmp13436;
    const int32_t tmp13557 = (tmp13501 + tmp13568);
    const int32_t tmp32 =  min(tmp22, tmp13557);
    const int32_t tmp33 =  max(tmp24, tmp32);
    const int32_t tmp13517 = (tmp33 & tmp13548);
    const int32_t tmp13623 = (tmp13517 << tmp13498);
    const int32_t tmp13581 = (tmp13513 + tmp13623);
    S21_ch tmp13447 = tmp13431.children(tmp13581);
    device float* tmp13448 = tmp13447.get0(runtime_, mem_alloc_).val;
    const auto tmp35 = *tmp13448;
    const int32_t tmp13559 = (tmp13505 + tmp13562);
    const int32_t tmp37 =  min(tmp22, tmp13501);
    const int32_t tmp38 =  max(tmp24, tmp37);
    const int32_t tmp39 =  min(tmp26, tmp13559);
    const int32_t tmp40 =  max(tmp24, tmp39);
    const int32_t tmp13525 = (tmp38 & tmp13548);
    const int32_t tmp13529 = (tmp40 & tmp13552);
    const int32_t tmp13625 = (tmp13525 << tmp13498);
    const int32_t tmp13589 = (tmp13529 + tmp13625);
    S21_ch tmp13459 = tmp13431.children(tmp13589);
    device float* tmp13460 = tmp13459.get0(runtime_, mem_alloc_).val;
    const auto tmp42 = *tmp13460;
    const int32_t tmp13561 = (tmp13505 + tmp13568);
    const int32_t tmp44 =  min(tmp26, tmp13561);
    const int32_t tmp45 =  max(tmp24, tmp44);
    const int32_t tmp13537 = (tmp45 & tmp13552);
    const int32_t tmp13597 = (tmp13537 + tmp13625);
    S21_ch tmp13471 = tmp13431.children(tmp13597);
    device float* tmp13472 = tmp13471.get0(runtime_, mem_alloc_).val;
    const auto tmp47 = *tmp13472;
    S15 tmp13479 = tmp13430.get4(runtime_, mem_alloc_);
    const int32_t tmp13627 = (tmp13501 << tmp13498);
    const int32_t tmp13605 = (tmp13505 + tmp13627);
    S15_ch tmp13483 = tmp13479.children(tmp13605);
    device float* tmp13484 = tmp13483.get0(runtime_, mem_alloc_).val;
    const auto tmp49 = *tmp13484;
    const float tmp50 = (tmp30 + tmp35);
    const float tmp51 = (tmp50 + tmp42);
    const float tmp52 = (tmp51 + tmp47);
    const float tmp54 = (tmp49 * tmp53);
    const float tmp55 = (tmp52 + tmp54);
    const float tmp57 = (tmp55 * tmp56);
    S19 tmp13491 = tmp13430.get6(runtime_, mem_alloc_);
    S19_ch tmp13495 = tmp13491.children(tmp13605);
    device float* tmp13496 = tmp13495.get0(runtime_, mem_alloc_).val;
    *tmp13496 = tmp57;
  } else {
  }
}

}  // namespace
kernel void mtl_k0015_pressure_jacobi_c14_1_0(
    device byte* root_addr [[buffer(0)]],
    device byte* global_tmps_addr [[buffer(1)]],
    device byte* runtime_addr [[buffer(2)]],
    device byte* print_assert_addr [[buffer(3)]],
    const uint ugrid_size_ [[threads_per_grid]],
    const uint utid_ [[thread_position_in_grid]]) {
  // range_for, range known at compile time
  const int total_elems = 524288;
  const int begin_ = utid_ + 0;
  const int end_ = total_elems + 0;
  device auto *runtime_ = reinterpret_cast<device Runtime *>(runtime_addr);
  device auto *mem_alloc_ = reinterpret_cast<device MemoryAllocator *>(runtime_ + 1);
  for (int ii = begin_; ii < end_; ii += ugrid_size_) {
    mtl_k0015_pressure_jacobi_c14_1_0_func(root_addr, global_tmps_addr, runtime_addr, print_assert_addr, ii);
  }
}

