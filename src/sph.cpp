#include "taichi_grid.h"
#include <taichi/visual/texture.h>
#include <taichi/system/threading.h>

TC_NAMESPACE_BEGIN

constexpr auto N = 20;
constexpr std::array<int, 3> grid_resolution{N, N, N};

struct Particle {
  Vector3f position, velocity;
};

using Block = TBlock<char, Particle>;

class SPHTestBase {
 public:
  real dx, inv_dx, dt;  // Cell size and dt
  real h, inv_h;
  real frame_dt;
  int total_frames;
  using VectorI = Vector3i;
  using Vector = Vector3f;
  Vector3 domain_size;
  Vector3i node_range;
  int current_frame;
  Vector3 gravity;

  SPHTestBase() {
    gravity = Vector3(0, -100, 0);
    current_frame = 0;
    dx = 1.0_f / grid_resolution[0];
    frame_dt = 0.1_f;
    dt = 0.0003_f;
    inv_dx = 1.0_f / dx;
    // Half circle
    total_frames = 128;
    domain_size = VectorI(grid_resolution).template cast<real>() * dx;
    h = dx / 2;
    inv_h = 1.0_f / h;
  }

  virtual void substep() = 0;

  void advance() {
    TC_P(current_frame);
    for (int i = 0; i < frame_dt / dt; i++) {
      substep();
    }
    current_frame += 1;
    output(get_filename(current_frame));
  }

  std::string get_filename(int frame) {
    return fmt::format("/tmp/outputs/{:05d}.tcb", current_frame);
  }

  virtual void output(std::string fn) = 0;
};

class SPHTestBruteForce : public SPHTestBase {
 public:
  std::vector<Particle> particles;
  SPHTestBruteForce() : SPHTestBase() {
    Region3D region(Vector3i(0), Vector3i(N / 2));
    for (auto ind : region) {
      particles.push_back(Particle{(ind.get_pos()) * h, Vector3f(0)});
    }
  }

  void substep() override {
    const real c = 315.0_f / (64.0_f * pi * pow<9>(h));
    const real rho0 = 1;
    const real k = 1e-8;  // TODO: too small?
    for (auto &p : particles) {
      // Compute pressure
      real rho = 0;
      for (auto &q : particles) {
        auto dpos = q.position - p.position;
        auto dpos2 = length2(dpos);
        if (dpos2 < h * h) {
          rho += std::max(0.0_f, pow<3>(h * h - dpos2));
        }
      }
      rho *= c;
      // TC_P(rho);
      p.position[3] = 1.0_f / rho;
      p.velocity[3] = k * (pow<7>(rho / rho0) - 1);
    }

    for (auto &p : particles) {
      auto inv_rho = p.position[3];
      Vector3f pressure;
      for (auto &q : particles) {
        auto dpos = q.position - p.position;
        auto dpos2 = length2(dpos);
        if (dpos2 < h * h) {
          auto inv_rho1 = q.position[3];
          auto grad = -6 * (h * h - dpos2) * dpos;
          pressure += (p.velocity[3] * inv_rho * inv_rho +
                       q.velocity[3] * inv_rho1 * inv_rho1) *
                      grad;
        }
      }

      Vector3f force = (k * pressure + gravity) * dt;

      p.velocity[3] = 0;
      p.position[3] = 0;
      p.velocity += force * dt;
      p.position += p.velocity * dt;
      if (p.position.y < h / 2) {
        p.position.y = h / 2;
        p.velocity.y = std::max(p.velocity.y, 0.0_f);
      }
      if (p.position.x < 0) {
        p.position.x = 0;
        p.velocity.x = std::max(p.velocity.x, 0.0_f);
      }
      if (p.position.z < 0) {
        p.position.z = 0;
        p.velocity.z = std::max(p.velocity.z, 0.0_f);
      }
      if (p.position.x > 1) {
        p.position.x = 1;
        p.velocity.x = std::min(p.velocity.x, 0.0_f);
      }
      if (p.position.z > 1) {
        p.position.z = 1;
        p.velocity.z = std::min(p.velocity.z, 0.0_f);
      }
    }
  }

  void output(std::string fn) {
    OptiXScene scene;
    for (auto &p : particles) {
      scene.particles.push_back(
          OptiXParticle{Vector4(p.position * 10.0_f, h * 10)});
    }
    write_to_binary_file(scene, fn);
  }
};

class SPHTestPangu : public SPHTestBase {
 public:
  using Grid = TaichiGrid<Block>;
  Grid grid;
  SPHTestPangu() : SPHTestBase() {
    for (auto ind : Region3D(Vector3i(0), node_range, Vector3(0.0_f))) {
      auto coord = ind.get_ipos();
      grid.touch(coord);
    }
    TC_TRACE("grid initialized");
    TC_P(grid.num_particles());
  }

  void substep() override {
    current_frame += 1;
    TC_P(current_frame);
    grid.advance(
        [&](Grid::Block &b, Grid::Ancestors &an) {
          TC_ASSERT(an[Vector3i(0)]);
          std::memcpy(b.nodes, an[Vector3i(0)]->nodes, sizeof(b.nodes));
          Grid::GridScratchPad scratch(an);
          Vector particle_range[2]{
              b.base_coord.template cast<real>() - Vector3(0.5_f),
              (b.base_coord + Vector3i(Block::size)).template cast<real>() -
                  Vector3(0.5_f)};
          for (auto ab : an.data) {
            if (!ab) {
              continue;
            }
            // Gather particles
            for (std::size_t i = 0; i < ab->particle_count; i++) {
              auto &p = ab->particles[i];
              auto grid_pos = p.position * inv_dx;
              if (particle_range[0] <= grid_pos &&
                  grid_pos < particle_range[1]) {
                b.add_particle(p);
              }
            }
          }
          for (std::size_t i = 0; i < b.particle_count; i++) {
            auto &p = b.particles[i];
          }
          return true;
        },
        false);
    output(get_filename(current_frame));
  }

  void output(std::string fn) {
    OptiXScene scene;
    for (auto &p : grid.gather_particles()) {
      scene.particles.push_back(
          OptiXParticle{Vector4(p.position * 3.0_f, 0.3_f)});
    }
    write_to_binary_file(scene, fn);
  }
};

auto sph = [](const std::vector<std::string> &params) {
  // auto _ = ThreadedTaskManager::TbbParallelismControl(1);
  std::unique_ptr<SPHTestBase> sph;
  TC_P(params[0]);
  if (params[0] == "bf") {
    sph = std::make_unique<SPHTestBruteForce>();
  } else {
    sph = std::make_unique<SPHTestPangu>();
  }
  for (int t = 0; t < sph->total_frames; t++) {
    sph->advance();
  }
};

TC_REGISTER_TASK(sph);

TC_NAMESPACE_END
