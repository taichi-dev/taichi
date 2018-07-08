#include <atomic>
#include <mutex>
#include <functional>
#include <hash_map>
#include <experimental/filesystem>
#include <tbb/tbb.h>
#include <taichi/util.h>
#include <taichi/system/virtual_memory.h>
#include <taichi/testing.h>
#include <taichi/system/timer.h>
#include <taichi/common/bit.h>
#include <taichi/system/profiler.h>
#include <taichi/io/optix.h>
#include <taichi/io/ply_writer.h>
#include <mpi.h>

TC_NAMESPACE_BEGIN

constexpr bool grid_debug = false;

template <typename T, int N>
TC_FORCE_INLINE constexpr T product(const std::array<T, N> arr) {
  T ret(1);
  for (int i = 0; i < N; i++) {
    ret *= arr[i];
  }
  return ret;
}

inline bool with_mpi() {
  auto c = std::getenv("OMPI_COMM_WORLD_SIZE");
  if (c == nullptr) {
    return false;
  } else {
    return true;
  }
}

constexpr std::size_t least_pot_bound(std::size_t v) {
  std::size_t ret = 1;
  while (ret < v) {
    ret *= 2;
  }
  return ret;
}

TC_FORCE_INLINE uint32 pdep(uint32 value, uint32 mask) {
  return _pdep_u32(value, mask);
}

TC_FORCE_INLINE uint32 pdep(int32 value, int32 mask) {
  return pdep((uint32)value, (uint32)mask);
}

TC_FORCE_INLINE uint64 pdep(uint64 value, uint64 mask) {
  return _pdep_u64(value, mask);
}

TC_FORCE_INLINE constexpr uint32 pot_mask(int x) {
  return (1u << x) - 1;
}

TC_FORCE_INLINE constexpr uint32 log2int(uint64 value) {
  int ret = 0;
  value >>= 1;
  while (value) {
    value >>= 1;
    ret += 1;
  }
  return ret;
}

// TODO: speed it up
// using SHR
TC_FORCE_INLINE Vector3i div_floor(Vector3i a, Vector3i b) {
  auto half = TVector<uint32, 3>(1u << 31);
  return ((a.template cast<uint32>() + half) / b.template cast<uint32>() -
          half / b.template cast<uint32>())
      .template cast<int32>();
}

// In C++17, this can be replaced with a std::array
template <int x_, int y_ = x_, int z_ = y_>
struct TSize3D {
  static constexpr int x() {
    return x_;
  }
  static constexpr int y() {
    return y_;
  }
  static constexpr int z() {
    return z_;
  }
  static Vector3i VectorI() {
    return Vector3i(x_, y_, z_);
  }
};

template <typename Node_,
          typename Particle_,
          typename block_size_ = TSize3D<8>,
          int dilation_ = 0>
struct TBlock {
  static constexpr int dim = 3;
  static constexpr int dilation = dilation_;
  using VectorI = TVector<int, 3>;
  using Vector = TVector<real, 3>;
  static constexpr uint32 coord_mask = (dim == 2) ? 0x55555555 : 0x49249249;
  using Node = Node_;
  using Particle = Particle_;
  using block_size = block_size_;

  using particle_to_grid_func = std::function<TVector<int, dim>(Particle &)>;

  static constexpr const std::array<int, dim> size = {
      block_size_::x(), block_size_::y(), block_size_::z()};

  static constexpr const std::array<int, dim> grid_size = {
      block_size_::x() + 2 * dilation, block_size_::y() + 2 * dilation,
      block_size_::z() + 2 * dilation};
  // This does NOT account for dilation
  static constexpr int num_nodes = product<int, dim>(grid_size);

  static_assert(bit::is_power_of_two(block_size::x()), "");
  static_assert(bit::is_power_of_two(block_size::y()), "");
  static_assert(bit::is_power_of_two(block_size::z()), "");

  // Meta data
  VectorI base_coord;  // smallest coordinate
  bool killed;

  // Grid data
  Node nodes[num_nodes];

  static constexpr int max_num_particles = 12 * num_nodes;
  Particle particles[max_num_particles];
  // Particle data
  std::size_t particle_count;

  TBlock() {
  }

  TBlock(VectorI base_coord) {
    initialize(base_coord);
  }

  // This is cheaper because particles/nodes will not be initialized
  void initialize(VectorI base_coord) {
    this->base_coord = base_coord;
    this->killed = false;
    this->particle_count = 0;
    std::memset(nodes, 0, sizeof(nodes));
  }

  void add_particle(const Particle &p) {
    if (grid_debug) {
      TC_ASSERT(particle_count < max_num_particles - 1);
    }
    particles[particle_count++] = p;
  }

  TC_FORCE_INLINE int linearize_local(const VectorI local_coord) {
    int ret = local_coord[0] + dilation;
    for (int i = 1; i < dim; i++) {
      // Note: Grid size = block_size * dilation * 2
      ret = ret * grid_size[i] + (local_coord[i] + dilation);
    }
    return ret;
  }

  TC_FORCE_INLINE VectorI to_local(const VectorI global_coord) {
    return global_coord - base_coord;
  }

  TC_FORCE_INLINE VectorI to_global(const VectorI local_coord) {
    return local_coord + base_coord;
  }

  TC_FORCE_INLINE int linearize_global(const VectorI global_coord) {
    return linearize_local(to_local(global_coord));
  }

  TC_FORCE_INLINE Node &node_local(const VectorI local_coord) {
    if (grid_debug) {
      TC_ASSERT(inside_dilated_local(local_coord));
    }
    return nodes[linearize_local(local_coord)];
  }

  TC_FORCE_INLINE Node &node_global(const VectorI global_coord) {
    if (grid_debug) {
      TC_ASSERT(inside_dilated_global(global_coord));
    }
    return nodes[linearize_global(global_coord)];
  }

  template <typename T>
  TC_FORCE_INLINE void for_each_node(const T &t) {
    for (int i = 0; i < num_nodes; i++) {
      t(nodes[i]);
    }
  }

  TC_FORCE_INLINE bool inside_dilated_local(const VectorI local_coord) {
    return -dilation <= local_coord[0] &&
           local_coord[0] < block_size_::x() + dilation &&
           -dilation <= local_coord[1] &&
           local_coord[1] < block_size_::y() + dilation &&
           -dilation <= local_coord[2] &&
           local_coord[2] < block_size_::z() + dilation;
  }

  TC_FORCE_INLINE bool inside_undilated_local(const VectorI local_coord) {
    return 0 <= local_coord[0] && local_coord[0] < block_size_::x() &&
           0 <= local_coord[1] && local_coord[1] < block_size_::y() &&
           0 <= local_coord[2] && local_coord[2] < block_size_::z();
  }

  TC_FORCE_INLINE bool inside_dilated_global(const VectorI global_coord) {
    return inside_dilated_local(to_local(global_coord));
  }

  TC_FORCE_INLINE bool inside_undilated_global(const VectorI global_coord) {
    return inside_undilated_local(to_local(global_coord));
  }
};

// Root buckets
template <typename Block, typename bucket_size>
struct TRootDomain {
  // each bucket has a VM allocator
  static constexpr int dim = 3;
  static constexpr const std::array<int, dim> &block_size = Block::size;

  using VectorI = VectorND<dim, int>;
  static constexpr auto coord_mask = Block::coord_mask;

  std::unique_ptr<VirtualMemoryAllocator> allocator;
  Block *data;
  std::vector<uint64> bitmap;
  VectorI base_coord;

  static constexpr int num_blocks = pow<3>(128 / 8);

  TRootDomain(VectorI base_coord) : base_coord(base_coord) {
    TC_ASSERT(num_blocks ==
              (bucket_size::x() / block_size[0]) *
                  (bucket_size::y() / block_size[1]) *
                  (bucket_size::z() / block_size[2]));
    // TC_P(num_blocks);
    bitmap.resize((num_blocks + 63) / 64);
    allocator =
        std::make_unique<VirtualMemoryAllocator>(num_blocks * sizeof(Block));
    data = (Block *)allocator->ptr;
    reset();
  }

  TC_FORCE_INLINE VectorI to_local(VectorI global_coord) {
    return global_coord - base_coord;
  }

  TC_FORCE_INLINE VectorI to_global(VectorI local_coord) {
    return local_coord + base_coord;
  }

  TC_FORCE_INLINE uint global_coord_to_block_id(VectorI global_coord) {
    return local_coord_to_block_id(to_local(global_coord));
  }

  // Input: local node coord
  // Output: block id
  TC_FORCE_INLINE uint32 local_coord_to_block_id(const VectorI &coord) {
    uint32 ret = pdep((uint32)coord[0] >> log2int(block_size[0]), coord_mask);
    ret |= pdep((uint32)coord[1] >> log2int(block_size[1]), coord_mask << 1);
    if (dim == 3)
      ret |= pdep((uint32)coord[2] >> log2int(block_size[2]), coord_mask << 2);
    static_assert((coord_mask & (coord_mask << 1)) == 0);
    static_assert((coord_mask & (coord_mask >> 1)) == 0);
    bool ok = 0 <= ret && ret < num_blocks;
    if (grid_debug && !ok) {
      TC_P(ret);
      TC_P(base_coord);
      TC_P(coord);
      TC_ASSERT(0 <= ret && ret < num_blocks);
    }
    return ret;
  }

  void touch(VectorI global_coord) {
    auto local_coord = global_coord - base_coord;
    auto bid = local_coord_to_block_id(local_coord);
    if (get_block_activity(bid)) {
      return;
    } else {
      auto block_base_coord = to_global(
          div_floor(local_coord, VectorI(block_size)) * VectorI(block_size));
      data[bid].initialize(block_base_coord);
      set_block_activity(bid, true);
    }
  }

  TC_FORCE_INLINE bool get_block_activity(int bid) const {
    return bool((bitmap[(bid >> 6)] >> (bid & 63)) & 1);
  }

  TC_FORCE_INLINE void set_block_activity(int bid, bool activation) {
    if (activation) {
      bitmap[(bid >> 6)] |= 1ul << (bid & 63);
    } else {
      bitmap[(bid >> 6)] &= ~(1ul << (bid & 63));
    }
  }

  TC_FORCE_INLINE Block &get_block_from_global_coord(VectorI global_coord) {
    auto bid = local_coord_to_block_id(to_local(global_coord));
    if (grid_debug) {
      TC_ASSERT_INFO(0 <= bid && bid < num_blocks,
                     fmt::format("Using untouched block! (bid={})", bid));
    }
    return data[bid];
  }

  void clear_killed_blocks() {
    for (int i = 0; i < num_blocks; i++) {
      if (get_block_activity(i))
        if (data[i].killed)
          set_block_activity(i, false);
    }
  }

  void collect_blocks(std::vector<Block *> &blocks) {
    for (int i = 0; i < num_blocks; i++) {
      if (get_block_activity(i)) {
        if (data[i].killed) {
          set_block_activity(i, false);
        } else {
          blocks.push_back(&(data[i]));
        }
      }
    }
  }

  void reset() {
    std::fill(bitmap.begin(), bitmap.end(), 0);
    // blocks will be re-initialized when used for the next time
  }

  std::size_t num_active_blocks() const {
    std::size_t ret = 0;
    for (int i = 0; i < num_blocks; i++) {
      if (get_block_activity(i)) {
        ret += 1;
      }
    }
    return ret;
  }
};

template <typename Block>
struct TAncestors {
  static constexpr int dim = Block::dim;
  // TODO: staggered
  static constexpr int width = 3;
  static constexpr int size = pow<dim>(width);
  using VectorI = TVector<int, dim>;

  Block *data[size];

  TAncestors() {
    std::memset(data, 0, sizeof(data));
  }
  Block *&operator[](int i) {
    return data[i];
  }
  // Each coord of offset can be -1 ~ +1
  template <int dim_ = dim>
  std::enable_if_t<dim_ == 2, Block *&> operator[](VectorI offset) {
    return data[offset.x * width + offset.y + width + 1];
  }

  template <int dim_ = dim>
  std::enable_if_t<dim_ == 3, Block *&> operator[](VectorI offset) {
    if (grid_debug) {
      TC_ASSERT(VectorI(-1) <= offset && offset < VectorI(width));
    }
    return data[offset.x * (width * width) + offset.y * width + offset.z +
                width * width + width + 1];
  }
};

template <typename Block>
struct TGridScratchPad {
  static constexpr int dim = 3;
  TC_STATIC_ASSERT(dim == 3);
  using VectorI = typename Block::VectorI;
  using Node = typename Block::Node;

  static constexpr std::array<int, 3> scratch_size{
      Block::size[0] + 2, Block::size[1] + 2, Block::size[2] + 2};

  using VolumeData = Node[scratch_size[0]][scratch_size[1]][scratch_size[2]];

  Node linearized_data[scratch_size[0] * scratch_size[1] * scratch_size[2]];

  TC_STATIC_ASSERT(sizeof(VolumeData) == sizeof(linearized_data));
  VolumeData &data = *reinterpret_cast<VolumeData *>(
      &linearized_data[scratch_size[1] * scratch_size[2] + scratch_size[2] +
                       1]);

  TGridScratchPad(TAncestors<Block> &ancestors) {
    // TC_P(ancestors.data);
    // Gather linearized data
    RegionND<dim> region(VectorI(-1), VectorI(Block::size) + VectorI(1));
    auto bs = VectorI(Block::size);
    int p = 0;
    for (auto &ind : region) {
      VectorI block_offset = (ind.get_ipos() + bs) / bs - VectorI(1);
      Block *an_b = ancestors[block_offset];
      auto local_coord = (ind.get_ipos() + bs) % bs;
      if (an_b) {
        linearized_data[p] = an_b->node_local(local_coord);
      }
      p++;
    }
  }
};

template <typename Block_, typename bucket_size = TSize3D<128>>
class TaichiGrid {
 public:
  using Block = Block_;
  static constexpr int dim = Block::dim;
  static constexpr const std::array<int, dim> &block_size = Block::size;
  static constexpr int TC_GRID_CURRENT = 0;
  static constexpr int TC_GRID_PREVIOUS = 1;
  using VectorI = TVector<int, dim>;
  // static constexpr const std::array<int, dim> bucket_size = {128, 128, 128};
  using Node = typename Block::Node;
  using Particle = typename Block::Particle;
  using Ancestors = TAncestors<Block>;
  using GridScratchPad = TGridScratchPad<Block>;
  using PartitionFunction = std::function<int(VectorI)>;
  PartitionFunction part_func;

  void gc() {
    TC_NOT_IMPLEMENTED
  }

  int world_size;  // aka num. machines (processes)
  int world_rank;

  bool blocks_dirty;
  std::vector<Block *> blocks;

  using RootDomain = TRootDomain<Block, bucket_size>;
  // A mapping to root domains
  // TODO: this is slow
  std::unordered_map<uint64, std::unique_ptr<RootDomain>> root;
  std::unordered_map<uint64, std::unique_ptr<RootDomain>> root_previous;

  TC_FORCE_INLINE uint64 domain_hash(VectorI coord) {
    VectorI bucket_coord = div_floor(coord, bucket_size::VectorI());
    constexpr int coord_width = 20;
    uint64 rep = 0;
    for (int i = 0; i < dim; i++) {
      if (grid_debug) {
        bool ok = -(1 << (coord_width - 1)) <= bucket_coord[i] &&
                  bucket_coord[i] < (1 << (coord_width - 1));
        if (!ok) {
          // TC_P(coord);
          // TC_P(bucket_coord);
        }
        TC_ASSERT(-(1 << (coord_width - 1)) <= bucket_coord[i] &&
                  bucket_coord[i] < (1 << (coord_width - 1)));
      }
      rep = (rep << coord_width) + bucket_coord[i];
    }
    return rep;
  }

  RootDomain &get_root_domain(VectorI coord) {
    // Let's use a hash table first
    auto h = domain_hash(coord);
    return *root[h];
  }

  TaichiGrid() {
    blocks_dirty = false;
    if (with_mpi()) {
      MPI_Init(nullptr, nullptr);
      MPI_Comm_size(MPI_COMM_WORLD, &world_size);
      MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
      TC_P(world_rank);
      TC_ASSERT(world_size == 2 || world_size == 4);
      if (world_size == 2) {
        part_func = [](const VectorI coord) -> int {
          return int(coord.x >= 0);
        };
      } else {
        part_func = [](const VectorI coord) -> int {
          return 2 * int(coord.x >= 0) + int(coord.z >= 0);
        };
      }
    } else {
      part_func = [](const VectorI coord) -> int { return 0; };
      world_size = 1;
      world_rank = 0;
    }
  }

  ~TaichiGrid() {
    if (with_mpi()) {
      MPI_Finalize();
    }
  }

  TC_FORCE_INLINE RootDomain *get_root_domain_if_exist(
      const VectorI &coord,
      int which = TC_GRID_CURRENT) {
    auto h = domain_hash(coord);
    if (which == TC_GRID_CURRENT) {
      auto p = root.find(h);
      if (p == root.end()) {
        return nullptr;
      } else {
        return p->second.get();
      }
    } else {
      auto p = root_previous.find(h);
      if (p == root_previous.end()) {
        return nullptr;
      } else {
        return p->second.get();
      }
    }
  }

  TC_FORCE_INLINE Block *get_block_if_exist(const VectorI &coord,
                                            int which = TC_GRID_CURRENT) {
    auto domain = get_root_domain_if_exist(coord, which);
    if (domain) {
      auto bid = domain->global_coord_to_block_id(coord);
      // TC_P(bid);
      if (domain->get_block_activity(bid)) {
        return &domain->data[bid];
      }
    }
    return nullptr;
  }

  TC_FORCE_INLINE Node &node(const VectorI &coord) {
    // Step I find root domain
    auto &d = get_root_domain(coord);
    // Step II find block
    auto &b = d.get_block_from_global_coord(coord);
    return b.node_global(coord);
  }

  void update_block_list(bool force = false) {
    if (!force && !blocks_dirty)
      return;
    blocks.clear();
    for (auto &kv : root) {
      auto &domain = *kv.second;
      // domain->bitmap
      domain.collect_blocks(blocks);
    }
    blocks_dirty = false;
  }

  template <typename T>
  void for_each_block(const T &t) {
    update_block_list();
    tbb::parallel_for_each(blocks.begin(), blocks.end(),
                           [&](Block *block) { t(*block); });
  }

  template <typename T, typename R>
  std::result_of_t<T(Block &)> reduce(const T &t,
                                      const R &r,
                                      std::result_of_t<T(Block &)> v) {
    using V = std::result_of_t<T(Block &)>;
    update_block_list();
    return tbb::parallel_reduce(
        tbb::blocked_range<std::size_t>(std::size_t(0), blocks.size()), v,
        [&](tbb::blocked_range<std::size_t> &range, V v) {
          for (std::size_t i = range.begin(); i < range.end(); i++) {
            v = r(v, t(*blocks[i]));
          }
          return v;
        },
        r);
  }

  template <typename T, typename R>
  std::result_of_t<T(Block &)> reduce(const T &t, const R &r) {
    using V = std::result_of_t<T(Block &)>;
    return reduce(t, r, V(0));
  }

  template <typename T>
  std::result_of_t<T(Block &)> reduce(const T &t) {
    using V = std::result_of_t<T(Block &)>;
    return reduce(t, std::plus<V>(), V(0));
  }

  template <typename T>
  void for_each_node(const T &t) {
    update_block_list();
    tbb::parallel_for_each(blocks.begin(), blocks.end(),
                           [&](Block *block) { block->for_each_node(t); });
  }

  // Activate a root domain/block
  void touch(VectorI coord) {
    blocks_dirty = true;
    auto h = domain_hash(coord);
    if (root.find(h) == root.end()) {
      // TODO: support staggered blocks here
      // create_domain
      // TC_TRACE("creating domain");
      auto base_coord =
          div_floor(coord, bucket_size::VectorI()) * bucket_size::VectorI();
      root[h] = std::make_unique<RootDomain>(base_coord);
    }
    auto &domain = get_root_domain(coord);
    domain.touch(coord);
  }

  void expand() {
    update_block_list();
    RegionND<dim> region(VectorI(-1), VectorI(2));
    for (auto &b : blocks) {
      auto base_coord = b->base_coord;
      for (auto &offset : region) {
        touch(base_coord + VectorI(Block::size) * offset.get_ipos());
      }
    }
  }

  void reset() {
    for (auto &kv : root) {
      auto &domain = *kv.second;
      domain.reset();
    }
  }

  std::size_t num_active_blocks() const {
    int sum = 0;
    for (auto &kv : root) {
      auto &domain = *kv.second;
      sum += domain.num_active_blocks();
    }
    return sum;
  }

  void clear_killed_blocks() {
    blocks_dirty = true;
    for (auto &kv : root) {
      auto &domain = *kv.second;
      domain.clear_killed_blocks();
    }
  }

  enum {
    TAG_REQUEST_BLOCKS,
    TAG_REQUEST_BLOCKS_NUM,
    TAG_REPLY_BLOCKS,
    TAG_REPLY_BLOCKS_NUM,
  };

  void fetch_neighbours() {
    TC_ASSERT(with_mpi());
    update_block_list();
    std::vector<std::vector<VectorI>> requested_blocks;
    requested_blocks.resize(world_size);

    for (auto *b : blocks) {
      RegionND<dim> region(VectorI(-1), VectorI(2));
      for (auto &b : blocks) {
        auto base_coord = b->base_coord;
        for (auto &offset : region) {
          auto nb_coord = base_coord + VectorI(Block::size) * offset.get_ipos();
          auto nb_rank = part_func(nb_coord);
          if (nb_rank == world_rank) {
            continue;
          }
          auto nb = get_block_if_exist(nb_coord);
          if (!nb) {
            requested_blocks[nb_rank].push_back(nb_coord);
          }
        }
      }
    }

    MPI_Request reqs[world_size];
    MPI_Status stats[world_size];
    std::size_t blocks_to_send[world_size];
    std::size_t blocks_to_recv[world_size];

    const auto coord_buffer_size = 1000000;

    std::vector<VectorI> recv_buffer;
    recv_buffer.resize(coord_buffer_size);

    // Stage 1: send out requests
    for (int p = 0; p < world_size; p++) {
      if (p == world_rank)
        continue;
      // Send request to peer for the block
      blocks_to_recv[p] = requested_blocks[p].size();
      // For Isend, make sure the content does not change
      MPI_Isend(&blocks_to_recv[p], 1, MPI_INT32_T, p, TAG_REQUEST_BLOCKS_NUM,
                MPI_COMM_WORLD, &reqs[p]);
      MPI_Isend(requested_blocks[p].data(),
                requested_blocks[p].size() * VectorI::storage_elements,
                MPI_INT32_T, p, TAG_REQUEST_BLOCKS, MPI_COMM_WORLD, &reqs[p]);
      std::sort(requested_blocks[p].begin(), requested_blocks[p].end(),
                [](VectorI a, VectorI b) {
                  if (a.x == b.x) {
                    if (a.y == b.y) {
                      return a.z < b.z;
                    } else {
                      return a.y < b.y;
                    }
                  } else {
                    return a.x < b.x;
                  }
                });

      requested_blocks[p].resize(
          std::unique(requested_blocks[p].begin(), requested_blocks[p].end()) -
          requested_blocks[p].begin());
      TC_ASSERT(requested_blocks[p].size() < coord_buffer_size);
      // TC_INFO("rank {} asking for {} blocks from rank {}", world_rank,
      // requested_blocks[p].size(), p);
    }
    // TC_TRACE("Stage 1 messages sent");

    std::vector<std::vector<Block>> block_buffers;
    block_buffers.resize(world_size);
    for (int p = 0; p < world_size; p++) {
      if (p == world_rank)
        continue;

      int count;
      MPI_Recv(&count, 1, MPI_INT32_T, p, TAG_REQUEST_BLOCKS_NUM,
               MPI_COMM_WORLD, &stats[p]);

      // TC_INFO("Rank {} received request from rank {}, {} blocks", world_rank,
      // p, count);

      std::vector<VectorI> coords(count);
      MPI_Recv(coords.data(), count * VectorI::storage_elements, MPI_INT32_T, p,
               TAG_REQUEST_BLOCKS, MPI_COMM_WORLD, &stats[p]);

      // Prepare requested blocks
      auto &block_buffer = block_buffers[p];
      block_buffer.resize(count);
      int i = 0;
      for (auto &coord : coords) {
        TC_ASSERT(part_func(coord) == world_rank);
        auto b = get_block_if_exist(coord);
        if (b != nullptr) {
          std::memcpy(&block_buffer[i], b, sizeof(Block));
          i++;
        }
      }
      // Stage 2: send out blocks
      // Note: some blocks may be empty, so possibly blocks to send !=
      // requested_blocks
      blocks_to_send[p] = i;
      block_buffer.resize(i);
      MPI_Isend(&blocks_to_send[p], 1, MPI_INT32_T, p, TAG_REPLY_BLOCKS_NUM,
                MPI_COMM_WORLD, &reqs[p]);
      MPI_Isend(block_buffer.data(), block_buffer.size() * sizeof(Block),
                MPI_CHAR, p, TAG_REPLY_BLOCKS, MPI_COMM_WORLD, &reqs[p]);
      // TC_INFO("Rank {} sent {} blocks to rank {}", world_rank, i, p);
      // TODO: serialize to save communication. For now, we just take the whole
      // block
    }
    // TC_P(sizeof(Block));

    for (int p = 0; p < world_size; p++) {
      if (p == world_rank)
        continue;
      // MPI_Wait(&reqs[p], &stats[p]);
      // stats[p].

      int num_blocks;
      MPI_Recv(&num_blocks, 1, MPI_INT32_T, p, TAG_REPLY_BLOCKS_NUM,
               MPI_COMM_WORLD, &stats[p]);
      std::vector<Block> blocks(num_blocks);
      // TC_WARN("Receiving {} blocks", num_blocks);
      // Block blocks[num_blocks];
      MPI_Recv(&blocks[0], num_blocks * sizeof(Block), MPI_CHAR, p,
               TAG_REPLY_BLOCKS, MPI_COMM_WORLD, &stats[p]);
      for (auto &b : blocks) {
        touch(b.base_coord);
        memcpy(get_block_if_exist(b.base_coord), &b, sizeof(b));
      }
    }
  }

  // Advance
  template <typename T>
  void advance(const T &t, bool needs_expand = true) {
    if (world_size != 1) {
      fetch_neighbours();
    }
    // T takes (base_coord, Ancestor) and returns bool (true to keep, false to
    // discard)
    // TODO: an optimization can be, when T returns false, do not even
    //   initialize the block

    // Populate blocks at the next time step, if NOT killed
    TC_PROFILE("update_block_list1", update_block_list());
    auto old_blocks = blocks;
    // Swap two grids
    TC_PROFILE("swap grids", std::swap(root, root_previous));
    TC_PROFILE("reset", reset());
    {
      TC_PROFILER("populate new grid1");
      for (auto b : old_blocks) {
        if (!b->killed)
          touch(b->base_coord);
      }
    }
    if (needs_expand) {
      TC_PROFILE("expand", expand());
    }
    TC_PROFILE("update_block_list2", update_block_list());
    {
      TC_PROFILER("computation");
      tbb::parallel_for_each(blocks.begin(), blocks.end(), [&](Block *block) {
        Ancestors ancestors;
        RegionND<dim> region(VectorI(-1), VectorI(2));
        auto base_coord = block->base_coord;
        for (auto &offset : region) {
          auto an_coord = base_coord + VectorI(Block::size) * offset.get_ipos();
          auto b = get_block_if_exist(an_coord, TC_GRID_PREVIOUS);
          if (b) {
            ancestors[offset.get_ipos()] = b;
          }
        }
        bool ret = t(*block, ancestors);
        if (!ret) {
          block->killed = true;
        }
      });
    }
    TC_PROFILE("clear_killed_blocks", clear_killed_blocks());
  }

  std::size_t num_particles() {
    return reduce([](Block &b) { return b.particle_count; });
  }

  template <typename T>
  void serial_for_each_particle(const T &t) {
    update_block_list();
    for (auto b : blocks) {
      for (std::size_t i = 0; i < b->particle_count; i++) {
        t(b->particles[i]);
      }
    }
  }

  std::vector<Particle> gather_particles() {
    // TODO: fix alignment issues here
    std::vector<Particle> p;
    update_block_list();
    for (auto b : blocks) {
      /*
      TC_P(sizeof(Particle));
      TC_P(b->particle_count);
      TC_P(&b->particles[0]);
      TC_P(b->particles + b->particle_count)
      TC_P(p.size());
      for (int i = 0; i < b->particle_count; i++) {
        p.push_back(b->particles[i]);
      }
      */
      p.insert(p.end(), b->particles, b->particles + b->particle_count);
      // TC_P(p.size());
    }
    return p;
  }
};

struct TestParticle {
  Vector3f position, velocity;
};

using TestGrid =
    TaichiGrid<TBlock<Vector3f, TestParticle>, TSize3D<128, 128, 128>>;

template <typename Block>
void accumulate_dilated_grids(Block &b, TAncestors<Block> &an) {
  auto base_coord = b.base_coord;
  Region3D local_grid_region(Vector3i(-Block::dilation),
                             Vector3i(Block::dilation) + Vector3i(Block::size));
  for (auto ind : local_grid_region) {
    auto local_i = ind.get_ipos();
    auto global_i = base_coord + local_i;
    for (auto ab : an.data) {
      if (ab == nullptr)
        continue;
      if (ab->inside_dilated_global(global_i)) {
        b.node_local(local_i) += ab->node_global(global_i);
      }
    }
  }
}

template <typename Block>
void gather_particles(Block &b,
                      TAncestors<Block> &an,
                      typename Block::particle_to_grid_func const &func) {
  for (auto ab : an.data) {
    if (!ab) {
      continue;
    }
    // Gather particles
    for (std::size_t i = 0; i < ab->particle_count; i++) {
      auto &p = ab->particles[i];
      auto grid_pos = func(p);
      if (b.inside_undilated_global(grid_pos)) {
        b.add_particle(p);
      }
    }
  }
}

TC_NAMESPACE_END
