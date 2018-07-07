#include "taichi_grid.h"
#include <taichi/visual/texture.h>
#include <taichi/system/threading.h>

TC_NAMESPACE_BEGIN

constexpr auto N = 128;
constexpr int grid_resolution[3] = {N, N, N / 8};
using Block = TestGrid::Block;

class AdvectionTestBase {
 public:
  using Particle = typename TestGrid::Block::Particle;
  real dx, inv_dx, dt;
  int total_steps;
  std::function<Vector3(Vector3)> velocity_func;
  Vector3 domain_size;
  Vector3i cell_range;
  Vector3i node_range;
  std::shared_ptr<Texture> tex;
  int current_frame;

  AdvectionTestBase() {
    current_frame = 0;
    velocity_func = [](const Vector3 p) {
      return Vector3(p.y - 0.5_f, 0.5_f - p.x, 0.0_f);
    };
    tex = create_instance<Texture>("taichi", Dict().set("scale", 0.95_f));
    // velocity_func = [](const Vector3 p) { return Vector3(1, 2, 3); };
    dx = 1.0_f / grid_resolution[0];
    inv_dx = 1.0_f / dx;
    // Half circle
    total_steps = grid_resolution[0];
    dt = pi / total_steps;
    node_range =
        Vector3i(grid_resolution[0], grid_resolution[1], grid_resolution[2]);
    cell_range = node_range - Vector3i(1);
    domain_size = cell_range.cast<real>() * dx;
  }

  virtual void iterate() = 0;

  void advance() {
    current_frame += 1;
    TC_P(current_frame);
    for (int i = 0; i < 100; i++) {
      iterate();
    }
    write(get_filename(current_frame));
  }

  virtual void write(std::string fn) = 0;

  std::string get_filename(int frame) {
    return fmt::format("/tmp/outputs/{:05d}.tcb", current_frame);
  }
};

class AdvectionTestBruteForce : public AdvectionTestBase {
 public:
  std::vector<Particle> particles;
  Vector3 velocity_grid[grid_resolution[0]][grid_resolution[1]]
                       [grid_resolution[2]];
  AdvectionTestBruteForce() : AdvectionTestBase() {
    for (auto ind : Region3D(Vector3i(0), node_range, Vector3(0.0_f))) {
      velocity_grid[ind.i][ind.j][ind.k] = velocity_func(ind.get_pos() * dx);
    }

    for (auto ind : Region3D(Vector3i(0), cell_range)) {
      Vector3 scale(1.2);
      auto particle_pos = ind.get_pos() * dx;
      auto pos = (particle_pos - Vector3(0.5_f)) * scale + Vector3(0.5_f);
      if (tex->sample(pos).x < 1) {
        particles.push_back(Particle{particle_pos});
      }
    }
  }

  TC_FORCE_INLINE Vector3 sample_velocity(Vector3 position) {
    auto xi = (int)std::floor(position.x * inv_dx);
    auto yi = (int)std::floor(position.y * inv_dx);
    auto zi = (int)std::floor(position.z * inv_dx);
    real rx = position.x * inv_dx - xi;
    real ry = position.y * inv_dx - yi;
    real rz = position.z * inv_dx - zi;
#define V(x, y, z) velocity_grid[xi + (x)][yi + (y)][zi + (z)]
    Vector3 vx0 = (1 - ry) * ((1 - rz) * V(0, 0, 0) + rz * V(0, 0, 1)) +
                  ry * ((1 - rz) * V(0, 1, 0) + rz * V(0, 1, 1));
    Vector3 vx1 = (1 - ry) * ((1 - rz) * V(1, 0, 0) + rz * V(1, 0, 1)) +
                  ry * ((1 - rz) * V(1, 1, 0) + rz * V(1, 1, 1));
#undef V
    return (1 - rx) * vx0 + rx * vx1;
  }

  void iterate() override {
    for (auto &p : particles) {
      // RK2 advection
      // auto intermediate_p =
      //    p.position + dt * 0.5_f * sample_velocity(p.position);
      // p.position += dt * sample_velocity(intermediate_p);
      p.position += dt * sample_velocity(p.position);
    }
  }

  void write(std::string fn) override {
    OptiXScene scene;
    for (auto &p : particles) {
      scene.particles.push_back(
          OptiXParticle{Vector4(p.position * 3.0_f, 0.03_f)});
    }
    write_to_binary_file(scene, fn);
  }
};

class AdvectionTestPangu : public AdvectionTestBase {
 public:
  TestGrid grid;
  AdvectionTestPangu() : AdvectionTestBase() {
    for (auto ind : Region3D(Vector3i(0), node_range, Vector3(0.0_f))) {
      auto coord = ind.get_ipos();
      grid.touch(coord);
      grid.node(coord) = velocity_func(ind.get_pos() * dx);
    }
    TC_TRACE("grid initialized");

    for (auto ind : Region3D(Vector3i(0), cell_range)) {
      Vector3 scale(1.2);
      auto particle_pos = ind.get_pos() * dx;
      auto pos = (particle_pos - Vector3(0.5_f)) * scale + Vector3(0.5_f);
      if (tex->sample(pos).x < 1) {
        auto p = Particle{particle_pos};
        auto b =
            grid.get_block_if_exist((particle_pos * inv_dx + Vector3(0.5_f))
                                        .floor()
                                        .template cast<int>());
        TC_ASSERT(b);
        b->add_particle(p);
      }
    }
    TC_P(grid.num_particles());
  }

  TC_FORCE_INLINE Vector3 sample_velocity(Block &b,
                                          Vector3 position,
                                          TestGrid::GridScratchPad &s) {
    position -= b.base_coord.template cast<real>() * dx;
    auto xi = (int)std::floor(position.x * inv_dx);
    auto yi = (int)std::floor(position.y * inv_dx);
    auto zi = (int)std::floor(position.z * inv_dx);
    real rx = position.x * inv_dx - xi;
    real ry = position.y * inv_dx - yi;
    real rz = position.z * inv_dx - zi;
#define V(x, y, z) s.data[xi + (x)][yi + (y)][zi + (z)]
    Vector3 vx0 = (1 - ry) * ((1 - rz) * V(0, 0, 0) + rz * V(0, 0, 1)) +
                  ry * ((1 - rz) * V(0, 1, 0) + rz * V(0, 1, 1));
    Vector3 vx1 = (1 - ry) * ((1 - rz) * V(1, 0, 0) + rz * V(1, 0, 1)) +
                  ry * ((1 - rz) * V(1, 1, 0) + rz * V(1, 1, 1));
#undef V
    return (1 - rx) * vx0 + rx * vx1;
  }

  void iterate() override {
    TC_PROFILER("advance");
    grid.advance(
        [&](Block &b, TestGrid::Ancestors &an) {
          TC_ASSERT(an[Vector3i(0)]);
          std::memcpy(b.nodes, an[Vector3i(0)]->nodes, sizeof(b.nodes));
          TestGrid::GridScratchPad scratch(an);
          gather_particles(b, an, [&](Particle &p) {
            return (p.position * inv_dx + Vector3f(0.5_f))
                .floor()
                .template cast<int>();
          });

          for (std::size_t i = 0; i < b.particle_count; i++) {
            auto &p = b.particles[i];
            // RK2 advection
            // auto intermediate_p =
            //    p.position + dt * 0.5_f * sample_velocity(p.position);
            // p.position += dt * sample_velocity(intermediate_p);
            auto sampled = sample_velocity(b, p.position, scratch);
            p.position += dt * sampled;
          }
          return true;
        },
        false);
  }

  void write(std::string fn) override {
    OptiXScene scene;
    for (auto &p : grid.gather_particles()) {
      scene.particles.push_back(
          OptiXParticle{Vector4(p.position * 3.0_f, 0.03_f)});
    }
    write_to_binary_file(scene, fn);
  }
};

auto advection = [](const std::vector<std::string> &params) {
  // auto _ = ThreadedTaskManager::TbbParallelismControl(1);
  std::unique_ptr<AdvectionTestBase> advection;
  TC_P(params[0]);
  if (params[0] == "bf") {
    advection = std::make_unique<AdvectionTestBruteForce>();
  } else {
    advection = std::make_unique<AdvectionTestPangu>();
  }
  for (int t = 0; t < advection->total_steps; t++) {
    advection->advance();
    print_profile_info();
  }
};

TC_REGISTER_TASK(advection);

TC_TEST("AdvectionTestBruteForce") {
  TC_WARN("Skipping advection test");
  return;
  auto advection = std::make_unique<AdvectionTestBruteForce>();

  // Test tri-linear interpolation
  for (int i = 0; i < 100000; i++) {
    auto p = advection->domain_size * Vector3::rand();
    // Only for linear velocity funcs
    TC_CHECK_EQUAL(advection->sample_velocity(p), advection->velocity_func(p),
                   1e-6_f);
  }

  // Test advection
  auto initial_particles = advection->particles;
  CHECK(initial_particles.size() > 1000);
  auto fraction = (real)initial_particles.size() / advection->cell_range.prod();
  CHECK(0.5_f < fraction);
  CHECK(fraction < 0.65_f);

  for (int t = 0; t < advection->total_steps; t++) {
    advection->iterate();
  }

  for (int i = 0; i < (int)initial_particles.size(); i++) {
    auto p = advection->particles[i].position;
    // Mirror against (0.5, 0.5)
    TC_CHECK_EQUAL(initial_particles[i].position,
                   Vector3(1.0_f - p.x, 1.0_f - p.y, p.z), 1e-5_f);
  }
}

TC_NAMESPACE_END
