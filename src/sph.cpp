#include "taichi_grid.h"
#include <taichi/visual/texture.h>
#include <taichi/system/threading.h>
#include <taichi/visual/gui.h>

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
      p.velocity[3] = k * std::max(0.0_f, (pow<7>(rho / rho0) - 1.0_f));
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

auto sph2d = [](const std::vector<std::string> &params) {
  using Vector = Vector2;
  struct Particle {
    Vector position, velocity;
    real pressure, inv_density;
    bool movable;
  };
  real dt = 0.00001;
  real dx = 0.02;
  const auto h = dx;

  std::vector<Particle> particles;

  int N = 24;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      bool movable = true;
      if (i < 2 || j < 2 || i >= N - 2 || j >= N - 2) {
        movable = false;
      } else {
        // if (i != N / 2 || j != N / 2) {
        if ((j < 5 || j > 15) || (i < 5 || i > 11)) {
          continue;
        }
      }
      particles.push_back(
          {Vector(i, j) * dx * 0.8_f + Vector(0.2), Vector(0), 0, 0, movable});
    }
  }

  GUI gui("SPH 2D", 800, 800);
  auto gravity = Vector(0, -100);

  real rho0 = 0;
  for (auto &p : particles) {
    // Compute pressure
    real rho = 0;
    for (auto &q : particles) {
      auto dpos = q.position - p.position;
      auto dpos2 = length2(dpos);
      if (dpos2 < h * h) {
        rho += pow<3>(h * h - dpos2);
      }
    }
    rho0 = std::max(rho0, rho);
  }
  rho0 *= 1.0_f;
  TC_P(rho0);

  while (1) {
    // const real k = 1e6_f;
    const real k = 1e11;
    // const real c = 315.0_f / (64.0_f * pi * pow<9>(h));
    for (int K = 0; K < 100; K++) {
      for (auto &p : particles) {
        // Compute pressure
        real rho = 0;
        for (auto &q : particles) {
          auto dpos = q.position - p.position;
          auto dpos2 = length2(dpos);
          if (dpos2 < h * h) {
            rho += pow<3>(h * h - dpos2);
          }
        }
        rho /= rho0;
        p.inv_density = 1.0_f / rho;
        p.pressure = k * (pow<7>(rho) - 1);
      }

      for (auto &p : particles) {
        if (!p.movable) {
          continue;
        }
        auto inv_rho = p.inv_density;
        Vector pressure;
        for (auto &q : particles) {
          auto dpos = q.position - p.position;
          auto dpos2 = length2(dpos);
          if (dpos2 < h * h) {
            auto inv_rho1 = q.inv_density;
            auto grad = -6 * pow<2>(h * h - dpos2) * dpos;
            pressure += (p.pressure * inv_rho * inv_rho +
                         q.pressure * inv_rho1 * inv_rho1) *
                        grad;
            /*
            pressure += (p.pressure + q.pressure) * grad;
            */
          }
        }

        p.velocity += (pressure + gravity) * dt;
        p.velocity *= (1 - dt * 10);
      }

      for (auto &p : particles) {
        p.position += p.velocity * dt;
        if (p.position.x < 0.1) {
          p.position.x = 0.1;
          p.velocity.x = std::max(p.velocity.x, 0.0_f);
        }
        if (p.position.y < 0.1) {
          p.position.y = 0.1;
          p.velocity.y = std::max(p.velocity.y, 0.0_f);
        }
        if (p.position.x > 1 - dx) {
          p.position.x = 1 - dx;
          p.velocity.x = std::min(p.velocity.x, 0.0_f);
        }
        if (p.position.y > 1 - dx) {
          p.position.y = 1 - dx;
          p.velocity.y = std::min(p.velocity.y, 0.0_f);
        }
      }
    }

    gui.get_canvas().clear(Vector4(0.5));
    for (auto &p : particles) {
      auto coord = (p.position / dx * 24.0_f).template cast<int>();
      Vector4 color;
      if (p.movable) {
        color = Vector4((1.0 / p.inv_density) * 0.3_f);
      } else {
        color = Vector4(1, 0, 0, 0);
      }
      for (auto ind : Region2D(Vector2i(-2), Vector2i(3))) {
        if (gui.canvas->img.inside(coord + ind.get_ipos())) {
          gui.canvas->img[coord + ind.get_ipos()] = color;
        }
      }
    }
    gui.update();
  }

};
TC_REGISTER_TASK(sph2d);

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
