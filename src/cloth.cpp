#include "taichi_grid.h"

TC_NAMESPACE_BEGIN

struct ClothNode {
  Vector3 x, v;
};

static constexpr int n = 256;
class ClothSimulation {
 public:
  ClothNode nodes[n][n];
  Vector3 sphere_center = Vector3(0.0_f);
  real sphere_radius = 0.199_f;
  Vector3 gravity = Vector3(0, -9.8_f, 0);
  real stiffness = 300.0_f;
  real inv_m = n * n;
  real total_time = 3.0;
  int num_frames = 200;
  int substeps = 500;

 public:
  ClothSimulation() {
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        auto &node = nodes[i][j];
        real x = (real)i / (n - 1);
        real y = (real)j / (n - 1);
        node.x = Vector3(x - 0.5_f, 0.2_f, y - 0.5_f);
        node.v = Vector3(0);
      }
    }
  }

  virtual void advance(real dt) {
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        // Compute forces
        auto &node = nodes[i][j];
        Vector3 f(0);
        auto accumulate = [&](int di, int dj) {
          auto dx = (nodes[i + di][j + dj].x - nodes[i][j].x);
          real strain = length(dx) -
                        std::sqrt(real(di * di + dj * dj)) * (1.0_f / (n - 1));
          // TC_P(strain);
          f += normalized(dx) * (stiffness * strain);
        };
        if (i > 0) {
          accumulate(-1, 0);
        }
        if (j > 0) {
          accumulate(0, -1);
        }
        if (i + 1 < n) {
          accumulate(1, 0);
        }
        if (j + 1 < n) {
          accumulate(0, 1);
        }
        if (i > 0 && j > 0) {
          accumulate(-1, -1);
        }
        if (i > 0 && j + 1 < n) {
          accumulate(-1, 1);
        }
        if (i + 1 < n && j > 0) {
          accumulate(1, -1);
        }
        if (i + 1 < n && j + 1 < n) {
          accumulate(1, 1);
        }
        node.v += (f * inv_m + gravity) * dt;
      }
    }
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        auto &node = nodes[i][j];
        // advection, collision
        node.x += node.v * dt;
        real penetration = sphere_radius - length(node.x - sphere_center);
        if (penetration > 0) {
          // Project position and velocity
          auto normal = normalized(node.x - sphere_center);
          node.x += normal * penetration;
          node.v += -std::min(0.0_f, dot(node.v, normal)) * normal;
        }
      }
    }
  }

  virtual void write(std::string fn) {
    if (false) {
      PLYWriter ply(fn);
      for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - 1; j++) {
          ply.add_face({
              PLYWriter::Vertex{nodes[i][j].x, Vector3(1)},
              PLYWriter::Vertex{nodes[i + 1][j].x, Vector3(1)},
              PLYWriter::Vertex{nodes[i + 1][j + 1].x, Vector3(1)},
              PLYWriter::Vertex{nodes[i][j + 1].x, Vector3(1)},
          });
        }
      }
    } else {
      OptiXMesh mesh;
      auto scene = OptiXScene();
      for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
          auto pos = 5.0_f * nodes[i][j].x + Vector3(0, 5, 0);
          mesh.vertices.push_back({pos, Vector3(0), Vector3(0), Vector2(0)});
          scene.particles.push_back(OptiXParticle{Vector4(pos, 0.003_f)});
        }
      }
      auto V = [&](int i, int j) -> unsigned int { return i * n + j; };
      for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - 1; j++) {
          mesh.faces.push_back({V(i, j), V(i + 1, j), V(i + 1, j + 1)});
          mesh.faces.push_back({V(i, j), V(i + 1, j + 1), V(i, j + 1)});
        }
      }
      TC_TIME(mesh.recompute_normals());
      scene.meshes.push_back(mesh);
      TC_TIME(write_to_binary_file(scene, fn));
    }
  }

  void run() {
    std::experimental::filesystem::create_directories("/tmp/outputs");
    for (int i = 0; i < num_frames; i++) {
      {
        TC_PROFILER("simulation");
        TC_TRACE("Simulating frame {}/{}", i, num_frames);
        for (int j = 0; j < substeps; j++) {
          this->advance(total_time / (num_frames * substeps));
        }
      }
      {
        TC_PROFILER("writing");
        this->write(fmt::format("/tmp/outputs/{:05d}.tcb", i));
      }
      print_profile_info();
    }
  }
};

constexpr auto block_size = 16;
TC_STATIC_ASSERT(n % block_size == 0);

struct BlockedCloth {
  static constexpr auto scratch_size = pow<2>(block_size + 2);
  ClothNode scratch[block_size + 2][block_size + 2];
  using LinearizedScratchType = ClothNode[scratch_size];

  TC_FORCE_INLINE LinearizedScratchType &linearized_scratch() {
    return *reinterpret_cast<LinearizedScratchType *>(&scratch[0][0]);
  }
  TC_STATIC_ASSERT(sizeof(scratch) == sizeof(LinearizedScratchType));

  TC_FORCE_INLINE static constexpr int linearized_offset(int x, int y) {
    return x * (block_size + 2) + y;
  }

  BlockedCloth() = default;
};

struct PanguClothSimulation : public ClothSimulation {
  static constexpr auto block_dim = n / block_size;
  static constexpr auto scratch_size = BlockedCloth::scratch_size;
  BlockedCloth blocks[2][block_dim][block_dim];
  int T;

  PanguClothSimulation() : ClothSimulation() {
    T = 0;
    for (int I = 0; I < block_dim; I++) {
      for (int J = 0; J < block_dim; J++) {
        blocks[0][I][J] = BlockedCloth();
        blocks[1][I][J] = BlockedCloth();
        for (int i = 0; i < block_size; i++) {
          for (int j = 0; j < block_size; j++) {
            blocks[T % 2][I][J].scratch[i + 1][j + 1] =
                nodes[I * block_size + i][J * block_size + j];
          }
        }
      }
    }
  }

  void advance(real dt) override {
    T += 1;
    auto task = [&](int block_id) {
      auto I = block_id / block_dim;
      auto J = block_id % block_dim;
      advance_block(I, J, dt);
    };
    tbb::parallel_for(0, block_dim * block_dim, task);
  }

  void advance_block(int I, int J, real dt) {
    auto &B = blocks[T % 2][I][J];  // Block to update
    auto &previous_blocks = blocks[(T + 1) % 2];
    // Gather interior nodes from previous block
    std::memcpy(&B.linearized_scratch()[0],
                &blocks[(T + 1) % 2][I][J].linearized_scratch()[0],
                sizeof(BlockedCloth::LinearizedScratchType));

    // Gather ghost nodes

    // Corners
    if (I > 0 && J > 0) {
      B.scratch[0][0] =
          previous_blocks[I - 1][J - 1].scratch[block_size][block_size];
    }
    if (I > 0 && J + 1 < block_dim) {
      B.scratch[0][block_size + 1] =
          previous_blocks[I - 1][J + 1].scratch[block_size][1];
    }
    if (I + 1 < block_dim && J > 0) {
      B.scratch[block_size + 1][0] =
          previous_blocks[I + 1][J - 1].scratch[1][block_size];
    }
    if (I + 1 < block_dim && J + 1 < block_dim) {
      B.scratch[block_size + 1][block_size + 1] =
          previous_blocks[I + 1][J + 1].scratch[1][1];
    }

    // Edges
    if (I > 0) {
      for (int j = 1; j <= block_size; j++) {
        B.scratch[0][j] = previous_blocks[I - 1][J].scratch[block_size][j];
      }
    }
    if (I + 1 < block_dim) {
      for (int j = 1; j <= block_size; j++) {
        B.scratch[block_size + 1][j] = previous_blocks[I + 1][J].scratch[1][j];
      }
    }
    if (J > 0) {
      for (int i = 1; i <= block_size; i++) {
        B.scratch[i][0] = previous_blocks[I][J - 1].scratch[i][block_size];
      }
    }
    if (J + 1 < block_dim) {
      for (int i = 1; i <= block_size; i++) {
        B.scratch[i][block_size + 1] = previous_blocks[I][J + 1].scratch[i][1];
      }
    }

    // Compute force
    for (int i = 1; i <= block_size; i++) {
      for (int j = 1; j <= block_size; j++) {
        // Deal with node (i, j)
        auto &node = B.scratch[i][j];
        Vector3 f(0);
        auto accumulate = [&](int di, int dj) {
          auto dx = (B.scratch[i + di][j + dj].x - node.x);
          real strain = length(dx) -
                        std::sqrt(real(di * di + dj * dj)) * (1.0_f / (n - 1));
          f += normalized(dx) * (stiffness * strain);
        };

        int u = I * block_size + i - 1;
        int v = J * block_size + j - 1;
        if (u > 0) {
          accumulate(-1, 0);
        }
        if (v > 0) {
          accumulate(0, -1);
        }
        if (u + 1 < n) {
          accumulate(1, 0);
        }
        if (v + 1 < n) {
          accumulate(0, 1);
        }
        if (u > 0 && v > 0) {
          accumulate(-1, -1);
        }
        if (u > 0 && v + 1 < n) {
          accumulate(-1, 1);
        }
        if (u + 1 < n && v > 0) {
          accumulate(1, -1);
        }
        if (u + 1 < n && v + 1 < n) {
          accumulate(1, 1);
        }
        node.v += (f * inv_m + gravity) * dt;
      }
    }

    for (int i = 1; i <= block_size; i++) {
      for (int j = 1; j <= block_size; j++) {
        auto &node = B.scratch[i][j];
        // Advection, collision
        node.x += node.v * dt;
        real penetration = sphere_radius - length(node.x - sphere_center);
        if (penetration > 0) {
          // Project position and velocity
          auto normal = normalized(node.x - sphere_center);
          node.x += normal * penetration;
          node.v += -std::min(0.0_f, dot(node.v, normal)) * normal;
        }
      }
    }
  }

  void write(std::string fn) override {
    for (int I = 0; I < block_dim; I++) {
      for (int J = 0; J < block_dim; J++) {
        for (int i = 0; i < block_size; i++) {
          for (int j = 0; j < block_size; j++) {
            nodes[I * block_size + i][J * block_size + j] =
                blocks[T % 2][I][J].scratch[i + 1][j + 1];
          }
        }
      }
    }
    ClothSimulation::write(fn);
  }
};

TC_TEST("Block transfer") {
  auto sim = std::make_unique<PanguClothSimulation>();
  for (int k = 0; k < 10; k++) {
    sim->advance(0);
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        real x = (real)i / (n - 1);
        real y = (real)j / (n - 1);
        auto pos = Vector3(x - 0.5_f, 0.2_f, y - 0.5_f);
        real dist = length2(sim->nodes[i][j].x - pos);
        CHECK(dist < 1e-5_f);
      }
    }
  }
}

auto test_cloth = [](const std::vector<std::string> &parameters) {
  taichi::run_tests();
  auto sim = std::make_unique<PanguClothSimulation>();
  // auto sim = std::make_unique<ClothSimulation>();
  sim->run();
};

TC_REGISTER_TASK(test_cloth);

TC_NAMESPACE_END
