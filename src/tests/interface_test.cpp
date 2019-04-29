#define TESTING
#include <taichi/util.h>
#include <taichi/io/io.h>
#include <taichi/testing.h>
#include "fem_interface.h"

void fem_solve_test(void *input, void *output) {
  using namespace taichi::fem_interface;
  FEMInterface interface(input, output);

  // This is a fake "FEM solver"
  using block_size = FEMInputs::ScalarGrid::block_size;
  for (auto &density_block : interface.param.density.blocks) {
    FEMOutputs::VectorGrid::Block displacement_block;
    for (int i = 0; i < 3; i++) {
      displacement_block.base_coordinates[i] =
          density_block.base_coordinates[i];
    }
    for (int ii = 0; ii < block_size::x; ii++) {
      for (int jj = 0; jj < block_size::y; jj++) {
        for (int kk = 0; kk < block_size::z; kk++) {
          // Set displacement
          for (int p = 0; p < 3; p++) {
            displacement_block.get(ii, jj, kk)[p] =
                density_block.get(ii, jj, kk) + p;
          }
        }
      }
    }
    interface.outputs.displacements.blocks.push_back(displacement_block);
  }

  TC_P(interface.outputs.displacements.blocks.size());
}

TC_NAMESPACE_BEGIN

using namespace fem_interface;

TC_TEST("fem_interface") {
  CoreState::set_trigger_gdb_when_crash(true);
  {
    std::unique_ptr<FEMInterface> inter_ptr = std::make_unique<FEMInterface>();
    FEMInterface &interface = *inter_ptr;
    FEMInputs &param = interface.param;
    param.version_major = FEM_INTERFACE_VERSION_MAJOR;
    param.version_minor = FEM_INTERFACE_VERSION_MINOR;
    param.resolution[0] = 128;
    param.resolution[1] = 256;
    param.resolution[2] = 512;
    param.use_density_only = true;
    param.dx = 0.01f;

    for (int i = 0; i < 17; i++) {
      ForceOnNode f;
      f.coord[0] = i * 2 + 1;
      f.coord[1] = i * i + 14;
      f.coord[2] = i / 3 + 1;
      f.f[0] = std::sqrt(1.0 * i);
      f.f[1] = i * i;
      f.f[2] = i * i * i / 10000.0;
      param.forces.push_back(f);
    }

    DirichletOnNode dirichlet;

    dirichlet.coord[0] = 95;
    dirichlet.coord[1] = 6;
    dirichlet.coord[2] = 3;
    dirichlet.axis = 2;
    dirichlet.value = 0.1234;
    param.dirichlet_nodes.push_back(dirichlet);

    dirichlet.coord[0] = 94;
    dirichlet.coord[1] = 11;
    dirichlet.coord[2] = 16;
    dirichlet.axis = 1;
    dirichlet.value = 0.5432;
    param.dirichlet_nodes.push_back(dirichlet);

    param.krylov.tolerance = 1e-5;
    param.krylov.max_iterations = 10000;
    param.krylov.restart_iterations = 1000;
    param.krylov.print_residuals = true;

    param.caller_method = "topology_optimization";

    param.extra_ints["multigrid_num_pre_smoothing"] = 2;
    param.extra_ints["multigrid_num_post_smoothing"] = 3;

    param.extra_strings["log_file_path"] = "/tmp/fem_solver.log";

    param.extra_doubles["smoother_jacobi_damping"] = 0.66667f;

    param.global_mu = 2.0f;
    param.global_lambda = 1.0f;

    // Create sparse density field
    int centers[3];
    for (int i = 0; i < 3; i++) {
      centers[i] = param.resolution[i] / 2;
    }

    int depth = 6;

    auto node_inside = [&](int *coord) -> bool {
      for (int i = 0; i < 3; i++) {
        if (centers[i] - depth < coord[i] && coord[i] < centers[i] + depth) {
          return true;
        }
      }
      return false;
    };

    auto cell_inside = [&](int *coord) -> bool {
      for (int i = 0; i < 3; i++) {
        if (centers[i] - depth < coord[i] &&
            coord[i] < centers[i] + depth - 1) {
          return true;
        }
      }
      return false;
    };

    using block_size = FEMInputs::ScalarGrid::block_size;

    CHECK(param.resolution[0] % block_size::x == 0);
    CHECK(param.resolution[1] % block_size::y == 0);
    CHECK(param.resolution[2] % block_size::z == 0);

    for (int i = 0; i < param.resolution[0]; i += block_size::x) {
      for (int j = 0; j < param.resolution[1]; j += block_size::y) {
        for (int k = 0; k < param.resolution[2]; k += block_size::z) {
          bool active = false;
          for (int ii = 0; ii < block_size::x; ii++) {
            for (int jj = 0; jj < block_size::y; jj++) {
              for (int kk = 0; kk < block_size::z; kk++) {
                int coord[3] = {i + ii, j + jj, k + kk};
                if (node_inside(coord)) {
                  active = true;
                }
              }
            }
          }
          if (!active) {
            continue;
          }
          FEMInputs::ScalarGrid::Block block;
          block.base_coordinates[0] = i;
          block.base_coordinates[1] = j;
          block.base_coordinates[2] = k;
          block.reset_data();
          for (int ii = 0; ii < block_size::x; ii++) {
            for (int jj = 0; jj < block_size::y; jj++) {
              for (int kk = 0; kk < block_size::z; kk++) {
                int coord[3] = {i + ii, j + jj, k + kk};
                if (cell_inside(coord)) {
                  // Set density
                  block.get(ii, jj, kk) = i * jj + k - j + kk * ii;
                }
              }
            }
          }
          param.density.blocks.push_back(block);
        }
      }
    }
    TC_P(param.density.blocks.size());

    interface.preserve_output(param.density.blocks.size());
    auto nblocks = param.density.blocks.size();
    write_to_binary_file(param, "/tmp/fem_test.tcb");
    interface.run();
    FEMOutputs &outputs = interface.outputs;

    TC_P(outputs.displacements.blocks.size());
    CHECK(outputs.displacements.blocks.size() == nblocks);

    for (auto &displacement_block : outputs.displacements.blocks) {
      int i = displacement_block.base_coordinates[0];
      int j = displacement_block.base_coordinates[1];
      int k = displacement_block.base_coordinates[2];
      for (int ii = 0; ii < block_size::x; ii++) {
        for (int jj = 0; jj < block_size::y; jj++) {
          for (int kk = 0; kk < block_size::z; kk++) {
            // Test displacement
            int coord[3] = {i + ii, j + jj, k + kk};
            if (cell_inside(coord)) {
              for (int p = 0; p < 3; p++) {
                CHECK(displacement_block.get(ii, jj, kk)[p] ==
                      i * jj + k - j + kk * ii + p);
              }
            }
          }
        }
      }
    }
  }

  {
    FEMInputs param;
    read_from_binary_file(param, "/tmp/fem_test.tcb");
    CHECK(param.version_major == FEM_INTERFACE_VERSION_MAJOR);
    CHECK(param.version_minor == FEM_INTERFACE_VERSION_MINOR);
    CHECK(param.resolution[0] == 128);
    CHECK(param.resolution[1] == 256);
    CHECK(param.resolution[2] == 512);
    CHECK(param.dx == 0.01f);
    CHECK(param.use_density_only == true);

    CHECK(param.forces.size() == 17u);
    for (int i = 0; i < 17; i++) {
      ForceOnNode f = param.forces[i];
      CHECK(f.coord[0] == i * 2 + 1);
      CHECK(f.coord[1] == i * i + 14);
      CHECK(f.coord[2] == i / 3 + 1);
      CHECK(f.f[0] == std::sqrt(1.0 * i));
      CHECK(f.f[1] == i * i);
      CHECK(f.f[2] == i * i * i / 10000.0);
    }

    CHECK(param.dirichlet_nodes.size() == 2u);
    CHECK(param.dirichlet_nodes[0].coord[0] == 95);
    CHECK(param.dirichlet_nodes[0].coord[1] == 6);
    CHECK(param.dirichlet_nodes[0].coord[2] == 3);
    CHECK(param.dirichlet_nodes[0].axis == 2);
    CHECK(param.dirichlet_nodes[0].value == 0.1234);

    CHECK(param.dirichlet_nodes[1].coord[0] == 94);
    CHECK(param.dirichlet_nodes[1].coord[1] == 11);
    CHECK(param.dirichlet_nodes[1].coord[2] == 16);
    CHECK(param.dirichlet_nodes[1].axis == 1);
    CHECK(param.dirichlet_nodes[1].value == 0.5432);

    CHECK(param.krylov.tolerance == 1e-5);
    CHECK(param.krylov.max_iterations == 10000);
    CHECK(param.krylov.restart_iterations == 1000);
    CHECK(param.krylov.print_residuals == true);

    CHECK(param.caller_method == "topology_optimization");

    CHECK(param.extra_ints.size() == 2u);
    CHECK(param.extra_ints.find("multigrid_num_pre_smoothing") !=
          param.extra_ints.end());
    CHECK(param.extra_ints["multigrid_num_pre_smoothing"] == 2);

    CHECK(param.extra_ints.find("multigrid_num_post_smoothing") !=
          param.extra_ints.end());
    CHECK(param.extra_ints["multigrid_num_post_smoothing"] == 3);

    CHECK(param.extra_strings.find("log_file_path") !=
          param.extra_strings.end());
    CHECK(param.extra_strings["log_file_path"] == "/tmp/fem_solver.log");

    CHECK(param.extra_doubles.find("smoother_jacobi_damping") !=
          param.extra_doubles.end());
    CHECK(param.extra_doubles["smoother_jacobi_damping"] == 0.66667f);

    CHECK(param.global_mu == 2.0f);
    CHECK(param.global_lambda == 1.0f);

    TC_P(param.density.blocks.size());

    CHECK(param.get_solver_state_ptr() == nullptr);
    auto addr = 0x1234567890543200ul;
    param.set_solver_state_ptr((void *)(addr));
    CHECK(reinterpret_cast<uint64>(param.get_solver_state_ptr()) == addr);
  }
}

TC_NAMESPACE_END
