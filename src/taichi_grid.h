#include <atomic>
#include <mutex>
#include <functional>
#include <unordered_map>
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

constexpr bool grid_debug = true;

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
          int dilation_ = 0,
          int max_particles_per_node_ = 12,
          typename Meta_ = char>
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
  using Meta = Meta_;

  static constexpr const std::array<int, dim> size = {
      block_size_::x(), block_size_::y(), block_size_::z()};

  static constexpr int max_particles_per_node = max_particles_per_node_;

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
  int timestamp;
  bool killed;
  bool computed;
  Meta meta;

  // Grid data
  Node nodes[num_nodes];
  using VolumeNodeType = Node[grid_size[0]][grid_size[1]][grid_size[2]];

  static constexpr int max_num_particles = max_particles_per_node * num_nodes;
  Particle particles[max_num_particles];
  // Particle data
  std::size_t particle_count;

  TC_FORCE_INLINE void kill() {
    TC_ASSERT(!killed);
    killed = true;
  }

  TBlock() {
  }

  TBlock(VectorI base_coord, int timestamp) {
    initialize(base_coord, timestamp);
  }

  TC_FORCE_INLINE VolumeNodeType &get_node_volume() {
    return *reinterpret_cast<VolumeNodeType *>(
        &nodes[grid_size[1] * grid_size[2] * dilation +
               grid_size[2] * dilation + dilation]);
  }

  // This is cheaper because particles/nodes will not be initialized
  void initialize(VectorI base_coord, int timestamp) {
    this->base_coord = base_coord;
    this->killed = false;
    this->computed = false;
    this->particle_count = 0;
    this->timestamp = timestamp;
    // std::memset(nodes, 0, sizeof(nodes));
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

  Region3D global_region() const {
    return Region3D(base_coord, base_coord + block_size::VectorI());
  }
  Region3D local_region() const {
    return Region3D(VectorI(0), block_size::VectorI());
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
  std::mutex lock;
  int timestamp;

  static constexpr int num_blocks = pow<3>(128 / 8);

  TRootDomain(VectorI base_coord, int timestamp)
      : base_coord(base_coord), timestamp(timestamp) {
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
    std::lock_guard<std::mutex> _(lock);
    auto local_coord = global_coord - base_coord;
    auto bid = local_coord_to_block_id(local_coord);
    if (get_block_activity(bid)) {
      return;
    } else {
      auto block_base_coord = to_global(
          div_floor(local_coord, VectorI(block_size)) * VectorI(block_size));
      data[bid].initialize(block_base_coord, timestamp);
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
    std::lock_guard<std::mutex> _(lock);
    for (int i = 0; i < num_blocks; i++) {
      if (get_block_activity(i))
        if (data[i].killed)
          set_block_activity(i, false);
    }
  }

  void collect_blocks(std::vector<Block *> &blocks) {
    std::lock_guard<std::mutex> _(lock);
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
    std::lock_guard<std::mutex> _(lock);
    std::fill(bitmap.begin(), bitmap.end(), 0);
    // blocks will be re-initialized when used for the next time
  }

  std::size_t num_active_blocks() {
    std::lock_guard<std::mutex> _(lock);
    std::size_t ret = 0;
    for (int i = 0; i < num_blocks; i++) {
      if (get_block_activity(i)) {
        ret += 1;
      }
    }
    return ret;
  }
};

template <typename Block, int _width = 3>
struct TAncestors {
  static constexpr int dim = Block::dim;
  // TODO: staggered
  static constexpr int width = _width;
  TC_STATIC_ASSERT(width == 2 || width == 3);

  // If width == 2, each coord of offset can be 0 ~ +1
  // If width == 3, each coord of offset can be -1 ~ +1
  static constexpr int coord_offset = width == 3 ? 1 : 0;
  static constexpr int size = pow<dim>(width);
  using VectorI = TVector<int, dim>;

  Block *data[size];

  TAncestors() {
    std::memset(data, 0, sizeof(data));
  }

  Block *&operator[](int i) {
    return data[i];
  }

  template <int dim_ = dim>
  std::enable_if_t<dim_ == 2, Block *&> operator[](VectorI offset) {
    return data[offset.x * width + offset.y + (width + 1) * coord_offset];
  }

  template <int dim_ = dim>
  std::enable_if_t<dim_ == 3, Block *&> operator[](VectorI offset) {
    if (grid_debug) {
      TC_ASSERT(VectorI(-coord_offset) <= offset && offset < VectorI(width));
    }
    return data[offset.x * (width * width) + offset.y * width + offset.z +
                (width * width + width + 1) * coord_offset];
  }
};

template <typename Block>
struct TGridScratchPad {
  static constexpr int dim = Block::dim;
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
      } else {
        std::memset(&linearized_data[p], 0, sizeof(linearized_data[p]));
      }
      p++;
    }
  }

  Node &node(VectorI ind) {
    return data[ind.x][ind.y][ind.z];
  }
};

template <typename Block_, typename bucket_size = TSize3D<128>>
class TaichiGrid {
 public:
  using Block = Block_;
  static constexpr int dim = Block::dim;
  static constexpr const std::array<int, dim> &block_size = Block::size;
  using VectorI = TVector<int, dim>;
  // static constexpr const std::array<int, dim> bucket_size = {128, 128, 128};
  using Node = typename Block::Node;
  using Particle = typename Block::Particle;
  using Ancestors = TAncestors<Block, 3>;
  using PyramidAncestors = TAncestors<Block, 2>;
  using GridScratchPad = TGridScratchPad<Block>;
  using PartitionFunction = std::function<int(VectorI)>;
  PartitionFunction part_func;

  void gc() {
    TC_NOT_IMPLEMENTED
  }

  int world_size;  // aka num. machines (processes)
  int world_rank;
  static constexpr int master_rank = 0;

  // TODO: remove
  bool blocks_dirty;

  std::mutex root_lock;

  using RootDomain = TRootDomain<Block, bucket_size>;
  // A mapping to root domains
  // TODO: this is slow
  using RootDomains = std::unordered_map<uint64, std::unique_ptr<RootDomain>>;
  //using RootDomains = std::map<uint64, std::unique_ptr<RootDomain>>;
  RootDomains root;
  int current_timestamp;

  TaichiGrid() {
    current_timestamp = 0;
    blocks_dirty = false;
    if (with_mpi()) {
      MPI_Comm_size(MPI_COMM_WORLD, &world_size);
      MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
      TC_P(world_rank);
      TC_ASSERT(world_size == 1 || world_size == 2 || world_size == 4);
      if (world_size == 1) {
        part_func = [](const VectorI coord) -> int { return 0; };
      } else if (world_size == 2) {
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
  }

  // NOTE: current timestamp will be mod by 2
  TC_FORCE_INLINE uint64 domain_hash(VectorI coord, int timestamp) {
    TC_ASSERT(((current_timestamp - 2 < timestamp) &&
               (timestamp <= current_timestamp)));
    VectorI bucket_coord = div_floor(coord, bucket_size::VectorI());
    constexpr int coord_width = 20;
    uint64 rep = uint(timestamp % 2);
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

  RootDomain &get_root_domain(VectorI coord, int timestamp) {
    // Let's use a hash table first
    auto h = domain_hash(coord, timestamp);
    return *root[h];
  }

  TC_FORCE_INLINE RootDomain *get_root_domain_if_exist(const VectorI &coord,
                                                       int timestamp) {
    auto h = domain_hash(coord, timestamp);
    // critical region
    std::lock_guard<std::mutex> _(root_lock);
    auto p = root.find(h);
    if (p == root.end()) {
      return nullptr;
    } else {
      return p->second.get();
    }
  }

  TC_FORCE_INLINE Block *get_block_if_exist(const VectorI &coord,
                                            int timestamp = -1) {
    if (timestamp == -1) {
      timestamp = current_timestamp;
    }
    auto domain = get_root_domain_if_exist(coord, timestamp);
    if (domain) {
      auto bid = domain->global_coord_to_block_id(coord);
      // TC_P(bid);
      if (domain->get_block_activity(bid)) {
        return &domain->data[bid];
      }
    }
    return nullptr;
  }

  TC_FORCE_INLINE Node &node(const VectorI &coord, int timestamp = -1) {
    if (timestamp == -1) {
      timestamp = current_timestamp;
    }
    // Step I find root domain
    auto &d = get_root_domain(coord, timestamp);
    // Step II find block
    auto &b = d.get_block_from_global_coord(coord);
    return b.node_global(coord);
  }

  std::vector<Block *> get_block_list(int timestamp = -1) {
    if (timestamp == -1) {
      timestamp = current_timestamp;
    }
    std::vector<Block *> blocks;
    for (auto &kv : root) {
      auto &domain = *kv.second;
      if (domain.timestamp == timestamp) {
        domain.collect_blocks(blocks);
      }
    }
    return blocks;
  }

  template <typename T>
  void for_each_block(const T &t) {
    auto blocks = get_block_list();
    tbb::parallel_for_each(blocks.begin(), blocks.end(),
                           [&](Block *block) { t(*block); });
  }

  template <typename T>
  void map(const T &t) {
    return for_each_block(t);
  }

  template <typename T, typename R>
  std::result_of_t<T(Block &)> reduce(const T &t,
                                      const R &r,
                                      std::result_of_t<T(Block &)> v) {
    using V = std::result_of_t<T(Block &)>;
    auto blocks = get_block_list();
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
  std::result_of_t<T(Block &)> reduce_max(const T &t) {
    using V = std::result_of_t<T(Block &)>;
    auto max = [](const V &a, const V &b) -> V {
      TC_ASSERT(!std::isnan(a) && !std::isnan(b));
      if (a > b) {
        return a;
      } else {
        return b;
      }
    };
    return reduce(t, max, V(-std::numeric_limits<V>::infinity()));
  }

  template <typename T>
  std::result_of_t<T(Block &)> reduce(const T &t) {
    using V = std::result_of_t<T(Block &)>;
    return reduce(t, std::plus<V>(), V(0));
  }

  template <typename T>
  void for_each_node(const T &t) {
    auto blocks = get_block_list();
    tbb::parallel_for_each(blocks.begin(), blocks.end(),
                           [&](Block *block) { block->for_each_node(t); });
  }

  void touch_if_inside(VectorI coord) {
    if (part_func(coord) == world_rank) {
      touch(coord);
    }
  }

  // Activate a root domain/block
  void touch(VectorI coord, int timestamp = -1) {
    if (timestamp == -1) {
      timestamp = current_timestamp;
    }
    auto h = domain_hash(coord, timestamp);
    {
      // Serial region
      std::lock_guard<std::mutex> _(root_lock);

      if (root.find(h) == root.end()) {
        // TODO: support staggered blocks here
        // create_domain
        // TC_TRACE("creating domain");
        auto base_coord =
            div_floor(coord, bucket_size::VectorI()) * bucket_size::VectorI();
        root[h] = std::make_unique<RootDomain>(base_coord, timestamp);
      }
      auto &domain = get_root_domain(coord, timestamp);
      domain.touch(coord);
    }
  }

  void expand(int timestamp) {
    RegionND<dim> region(VectorI(-1), VectorI(2));
    for (auto &b : get_block_list(timestamp)) {
      auto base_coord = b->base_coord;
      for (auto &offset : region) {
        touch(base_coord + VectorI(Block::size) * offset.get_ipos(), timestamp);
      }
    }
  }

  void reset() {
    TC_NOT_IMPLEMENTED;
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
    TAG_REQUEST_BLOCK_NUM,
    TAG_REQUEST_BLOCKS,
    TAG_REPLY_BLOCK_NUM,
    TAG_REPLY_BLOCKS,
    TAG_REPLY_PARTICLE_NUM,
    TAG_REPLY_PARTICLES
  };

  std::vector<std::vector<VectorI>> requested_blocks;
  std::vector<std::vector<Block>> block_buffers;

  std::vector<Block *> fetch_neighbours(int timestamp) {
    constexpr bool debug = false;
    TC_ASSERT(with_mpi());
    requested_blocks.resize(world_size);

    {
      TC_PROFILER("calc blocks to fetch");
      for (auto b : get_block_list()) {
        RegionND<dim> region(VectorI(-1), VectorI(2));
        auto base_coord = b->base_coord;
        for (auto &offset : region) {
          auto nb_coord = base_coord + VectorI(Block::size) * offset.get_ipos();
          auto nb_rank = part_func(nb_coord);
          if (nb_rank == world_rank) {
            continue;
          }
          auto nb = get_block_if_exist(nb_coord, timestamp);
          if (!nb) {
            requested_blocks[nb_rank].push_back(nb_coord);
          }
        }
      }
    }

    MPI_Request reqs[world_size];
    MPI_Status stats[world_size];
    std::size_t num_sent_blocks[world_size];
    std::size_t num_requested_blocks[world_size];
    std::vector<Block> recv_blocks;

    const auto coord_buffer_size = 1000000;

    {
      // Stage 1:
      TC_PROFILER("send out requests");
      for (int p = 0; p < world_size; p++) {
        if (p == world_rank)
          continue;
        // Send request to peer for the block

        // Remove repeated ones
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

        requested_blocks[p].resize(std::unique(requested_blocks[p].begin(),
                                               requested_blocks[p].end()) -
                                   requested_blocks[p].begin());
        TC_ASSERT(requested_blocks[p].size() < coord_buffer_size);
        if (debug)
          TC_INFO("rank {} asking for {} blocks from rank {}", world_rank,
                  requested_blocks[p].size(), p);
        num_requested_blocks[p] = requested_blocks[p].size();
        // For Isend, make sure the content does not change
        MPI_Isend(&num_requested_blocks[p], 1, MPI_INT32_T, p,
                  TAG_REQUEST_BLOCK_NUM, MPI_COMM_WORLD, &reqs[p]);
        MPI_Isend(requested_blocks[p].data(),
                  num_requested_blocks[p] * VectorI::storage_elements,
                  MPI_INT32_T, p, TAG_REQUEST_BLOCKS, MPI_COMM_WORLD, &reqs[p]);
      }
    }
    // TC_TRACE("Stage 1 messages sent");

    std::vector<Block *> new_blocks;
    block_buffers.resize(world_size);
    {
      TC_PROFILER("reply blocks")
      for (int p = 0; p < world_size; p++) {
        if (p == world_rank)
          continue;

        int count;
        TC_PROFILE("Recv count",
                   MPI_Recv(&count, 1, MPI_INT32_T, p, TAG_REQUEST_BLOCK_NUM,
                            MPI_COMM_WORLD, &stats[p]));

        if (debug)
          TC_INFO("Rank {} received request from rank {}, {} blocks",
                  world_rank, p, count);

        std::vector<VectorI> coords(count);
        TC_PROFILE("Recv blocks",
                   MPI_Recv(coords.data(), count * VectorI::storage_elements,
                            MPI_INT32_T, p, TAG_REQUEST_BLOCKS, MPI_COMM_WORLD,
                            &stats[p]));

        // Prepare requested blocks
        auto &block_buffer = block_buffers[p];
        {
          TC_PROFILER("resize coord buffer");
          if (count > (int)block_buffer.size())
            block_buffer.resize(count);
        }
        num_sent_blocks[p] = 0;
        {
          TC_PROFILER("prepare blocks to reply");
          for (auto &coord : coords) {
            TC_ASSERT(part_func(coord) == world_rank);
            auto b = get_block_if_exist(coord, timestamp);
            if (b != nullptr) {
              // TC_P(coord);
              std::memcpy(&block_buffer[num_sent_blocks[p]++], b,
                          sizeof(Block));
            }
          }
        }
        // Stage 2: send out blocks
        // Note: some blocks may be empty, so possibly blocks to send !=
        // requested_blocks
        // Note: do not use block_buffer.size() here. That's just a upper bound!
        TC_PROFILE("Isend count",
                   MPI_Isend(&num_sent_blocks[p], 1, MPI_INT32_T, p,
                             TAG_REPLY_BLOCK_NUM, MPI_COMM_WORLD, &reqs[p]));
        TC_PROFILE(
            "Isend blocks",
            MPI_Isend(block_buffer.data(), num_sent_blocks[p] * sizeof(Block),
                      MPI_CHAR, p, TAG_REPLY_BLOCKS, MPI_COMM_WORLD, &reqs[p]));
        if (debug)
          TC_INFO("Rank {} sent {} blocks to rank {}", world_rank,
                  num_sent_blocks[p], p);
        // TODO: serialize to save communication. For now, we just take the
        // whole block
      }
      // TC_P(sizeof(Block));
    }

    {
      TC_PROFILER("receive and save blocks");
      for (int p = 0; p < world_size; p++) {
        if (p == world_rank)
          continue;

        int num_blocks;
        TC_PROFILE("Recv block count",
                   MPI_Recv(&num_blocks, 1, MPI_INT32_T, p, TAG_REPLY_BLOCK_NUM,
                            MPI_COMM_WORLD, &stats[p]));
        if (debug) {
          TC_WARN("rank {} Receiving {} blocks", world_rank, num_blocks);
          TC_P(sizeof(Block));
        }
        {
          TC_PROFILER("Resize buffer")
          if (num_blocks > (int)recv_blocks.size()) {
            recv_blocks.resize(num_blocks);
          }
        }

        // Note: do not use recv_blocks.size() for the # of blocks received.
        // That's just an upper bound.

        {
          TC_PROFILER("Recv blocks");
          MPI_Recv(&recv_blocks[0], num_blocks * sizeof(Block), MPI_CHAR, p,
                   TAG_REPLY_BLOCKS, MPI_COMM_WORLD, &stats[p]);
        }
        if (debug)
          TC_WARN("rank {} Received {} blocks", world_rank, num_blocks);
        {
          TC_PROFILER("save blocks");
          for (int i = 0; i < num_blocks; i++) {
            touch(recv_blocks[i].base_coord, timestamp);
            auto local_b =
                get_block_if_exist(recv_blocks[i].base_coord, timestamp);
            TC_ASSERT(local_b);
            new_blocks.push_back(local_b);
            std::memcpy(local_b, &recv_blocks[i], sizeof(Block));
          }
        }
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    return new_blocks;
  }

  // Advance
  template <typename T>
  void advance(const T &t,
               bool needs_expand = true,
               bool carry_particles = false,
               bool carry_nodes = true) {
    // T takes (base_coord, Ancestor) and should return void
    using result_type = std::result_of_t<T(Block &, Ancestors &)>;
    TC_STATIC_ASSERT((std::is_same<result_type, void>::value));

    TC_PROFILER("advance");

    const int old_timestamp = current_timestamp;
    const int new_timestamp = current_timestamp + 1;

    current_timestamp += 1;
    // Populate blocks at the next time step, if NOT killed
    {
      TC_PROFILER("populate new grid1");
      auto list = get_block_list(old_timestamp);
      for (auto b: list) {
        if (!b->killed) {
          touch(b->base_coord, new_timestamp);
        }
      }
      /*
      tbb::parallel_for(0, (int)list.size(), [&](int i) {
        auto &b = list[i];
        if (!b->killed) {
          touch(b->base_coord, new_timestamp);
        }
      });
      */
    }

    auto compute_block = [&](Block *block) {
      if (!inside(block->base_coord)) {
        block->kill();
        return;
      }
      Ancestors ancestors;
      RegionND<dim> region(VectorI(-1), VectorI(2));
      auto base_coord = block->base_coord;
      for (auto &offset : region) {
        auto an_coord = base_coord + VectorI(Block::size) * offset.get_ipos();
        auto b = get_block_if_exist(an_coord, old_timestamp);
        if (b) {
          ancestors[offset.get_ipos()] = b;
        }
      }
      auto direct_ancestor = ancestors[VectorI(0)];
      if (direct_ancestor) {
        if (carry_particles) {
          std::memcpy(block->particles, direct_ancestor->particles,
                      direct_ancestor->particle_count * sizeof(Particle));
          block->particle_count = direct_ancestor->particle_count;
        }
        if (carry_nodes) {
          std::memcpy(block->nodes, direct_ancestor->nodes,
                      sizeof(Block::nodes));
        }
        block->meta = direct_ancestor->meta;
      }
      t(*block, ancestors);
      block->computed = true;
    };
    if (world_size != 1) {
      TC_PROFILER("Communicate");
      tbb::task_group g;
      auto existing_new_blocks = get_block_list(new_timestamp);
      g.run([&] {
        // TC_PROFILER("fetch_neighbours")
        auto fetched_blocks = fetch_neighbours(old_timestamp);
        // NOTE: newly fetched blocks should propagate to the next timestamp for
        // expansion
        for (auto &b : fetched_blocks) {
          if (!b->killed) {
            touch(b->base_coord, new_timestamp);
          }
        }
      });

      g.run([&] {
        // Do some computation to overlap with communication
        tbb::parallel_for_each(
            existing_new_blocks.begin(), existing_new_blocks.end(),
            [&](Block *block) {
              // Make sure all neighbours are inside domain
              auto base_coord = block->base_coord;
              RegionND<dim> region(VectorI(-1), VectorI(2));
              bool ancesters_inside = true;
              for (auto &offset : region) {
                auto an_coord =
                    base_coord + VectorI(Block::size) * offset.get_ipos();
                if (!inside(an_coord)) {
                  ancesters_inside = false;
                }
              }
              if (ancesters_inside && !block->computed && !block->killed)
                compute_block(block);
            });
      });
      TC_PROFILE("computation & communication part 1", g.wait());
    }

    if (needs_expand) {
      TC_PROFILE("expand", expand(new_timestamp));
    }
    {
      TC_PROFILER("computation");
      auto new_blocks = get_block_list(new_timestamp);
      tbb::parallel_for_each(new_blocks.begin(), new_blocks.end(),
                             [&](Block *block) {
                               if (!block->computed && !block->killed)
                                 compute_block(block);
                             });
    }
    TC_PROFILE("clear_killed_blocks", clear_killed_blocks());
    {
      TC_PROFILER("make new root");
      // This may be slow due to destructor invocations (RootDomain, Block)
      RootDomains new_root;
      for (auto &kv : root) {
        auto &b = *kv.second;
        // Note: this works only for synchronous...
        if (b.timestamp == current_timestamp) {
          new_root.insert(std::move(kv));
        }
      }
      root = std::move(new_root);
    }
  }

  std::size_t num_particles() {
    return reduce([](Block &b) { return b.particle_count; });
  }

  template <typename T>
  void serial_for_each_particle(const T &t) {
    for (auto b : get_block_list()) {
      for (std::size_t i = 0; i < b->particle_count; i++) {
        t(b->particles[i]);
      }
    }
  }

  std::vector<Particle> gather_particles() {
    // TODO: fix alignment issues here
    std::vector<Particle> particles;
    for (auto b : get_block_list()) {
      particles.insert(particles.end(), b->particles,
                       b->particles + b->particle_count);
    }
    if (with_mpi()) {
      MPI_Request req;
      MPI_Status stats;
      if (is_master()) {
        for (int p = 0; p < world_size; p++) {
          if (p == world_rank) {
            continue;
          }
          int count;
          // TC_INFO("waiting for count");
          MPI_Recv(&count, 1, MPI_INT32_T, p, TAG_REPLY_PARTICLE_NUM,
                   MPI_COMM_WORLD, &stats);
          // TC_INFO("receiving {} particles", count);
          std::vector<Particle> remote_particles(count);
          MPI_Recv(remote_particles.data(), count * sizeof(Particle), MPI_CHAR,
                   p, TAG_REPLY_PARTICLES, MPI_COMM_WORLD, &stats);
          particles.insert(particles.end(), remote_particles.begin(),
                           remote_particles.end());
        }
      } else {
        int count = particles.size();
        // TC_INFO("sending {} particles", count);
        MPI_Isend(&count, 1, MPI_INT32_T, master_rank, TAG_REPLY_PARTICLE_NUM,
                  MPI_COMM_WORLD, &req);
        MPI_Isend(particles.data(), count * sizeof(Particle), MPI_CHAR,
                  master_rank, TAG_REPLY_PARTICLES, MPI_COMM_WORLD, &req);
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }
    return particles;
  }

  TC_FORCE_INLINE bool inside(VectorI coord) {
    return part_func(div_floor(coord, VectorI(block_size)) *
                     VectorI(block_size)) == world_rank;
  }

  TC_FORCE_INLINE bool is_master() {
    return world_rank == master_rank;
  }

  template <typename T>
  void coarsen_to(TaichiGrid &coarse, const T &t) {
    // TODO: parallelize
    for (auto b : get_block_list()) {
      coarse.touch(div_floor(b->base_coord, VectorI(2)));
    }
    coarse.for_each_block([&](Block &b) {
      PyramidAncestors an;
      for (auto ind : Region3D(VectorI(0), VectorI(2))) {
        an[ind.get_ipos()] = get_block_if_exist(
            b.base_coord * VectorI(2) + ind.get_ipos() * VectorI(block_size));
      }
      t(b, an);
    });
  }

  template <typename T>
  void refine_from(TaichiGrid &coarse, const T &t) {
    for_each_block([&](Block &b) {
      auto ancestor_coord = div_floor(b.base_coord, VectorI(block_size) * 2) *
                            VectorI(block_size);
      auto ancestor = coarse.get_block_if_exist(ancestor_coord);
      if (grid_debug) {
        TC_ASSERT(ancestor);
      }
      t(b, *ancestor);
    });
  }
};

struct TestParticle {
  Vector3f position, velocity;
};

using TestGrid =
    TaichiGrid<TBlock<Vector3f, TestParticle>, TSize3D<128, 128, 128>>;

template <typename Block>
void stitch_dilated_grids(Block &b, TAncestors<Block> &an) {
  auto base_coord = b.base_coord;
  Region3D local_grid_region(Vector3i(-Block::dilation),
                             Vector3i(Block::dilation) + Vector3i(Block::size));
  for (auto ind : local_grid_region) {
    auto local_i = ind.get_ipos();
    auto global_i = base_coord + local_i;
    for (auto ab : an.data) {
      if (ab == nullptr || ab->base_coord == b.base_coord)
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

class MPIEnvironment {
 public:
  MPIEnvironment() {
    if (with_mpi())
      MPI_Init(nullptr, nullptr);
  }
  ~MPIEnvironment() {
    if (with_mpi())
      MPI_Finalize();
  }
};

template <typename T, typename block_size_>
struct LerpField {
  using block_size = block_size_;
  using Vector = TVector<real, 3>;
  using VectorI = TVector<int, 3>;
  using Region = TRegion<3>;
  T data[block_size::x()][block_size::y()][block_size::z()];
  Vector scale, inv_scale;
  Vector translate;

  LerpField(Vector scale, Vector translate)
      : scale(scale), inv_scale(Vector(1.0_f) / scale), translate(translate) {
  }

  TC_FORCE_INLINE T *linear_data() {
    return &data[0][0][0];
  }

  TC_FORCE_INLINE int linearize(const VectorI &ivec) {
    return ivec[0] * (block_size::y() * block_size::z()) +
           ivec[1] * block_size::z() + ivec[2];
  }

  TC_FORCE_INLINE T sample_global(Vector vec) {
    // World frame to local frame
    return sample_local(vec * scale - translate);
  }

  TC_FORCE_INLINE T sample(Vector vec) {
    return sample_global(vec);
  }

  TC_FORCE_INLINE T sample_local(Vector vec) {
    auto ivec = vec.floor().template cast<int>();
    auto fract = vec - ivec.template cast<real>();
    auto ind = linearize(ivec);
    const auto &rx = fract.x;
    const auto &ry = fract.y;
    const auto &rz = fract.z;
#define V(i, j, k)                                               \
  (linear_data()[ind + i * (block_size::y() * block_size::z()) + \
                 j * block_size::z() + k])
    T vx0 = (1 - ry) * ((1 - rz) * V(0, 0, 0) + rz * V(0, 0, 1)) +
            ry * ((1 - rz) * V(0, 1, 0) + rz * V(0, 1, 1));
    T vx1 = (1 - ry) * ((1 - rz) * V(1, 0, 0) + rz * V(1, 0, 1)) +
            ry * ((1 - rz) * V(1, 1, 0) + rz * V(1, 1, 1));
#undef V
    return (1 - rx) * vx0 + rx * vx1;
  }

  Region local_region() {
    return Region(VectorI(0), block_size::VectorI(), translate);
  }

  TC_FORCE_INLINE Vector node_pos(VectorI ind) {
    return (ind.template cast<real>() + translate) * inv_scale;
  }

  TC_FORCE_INLINE T &node(VectorI ind) {
    return data[ind.x][ind.y][ind.z];
  }
};

TC_NAMESPACE_END
