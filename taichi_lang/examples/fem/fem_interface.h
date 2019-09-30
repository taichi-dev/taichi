#pragma once

#define FEM_INTERFACE_VERSION_MAJOR 3
#define FEM_INTERFACE_VERSION_MINOR 1

////////////////////////////////////////////////////////////////////////////////
//                           The C Interface                                  //
////////////////////////////////////////////////////////////////////////////////
extern "C" void fem_solve(void *input, void *output);
extern "C" void fem_solve_test(void *input, void *output);

#include <vector>
#include <map>
#include <cmath>
#include <iostream>
#include <fstream>
#include <cstring>
#include <cstdlib>
#include <cinttypes>
#include <type_traits>

////////////////////////////////////////////////////////////////////////////////
//                          The C++ Interface                                 //
////////////////////////////////////////////////////////////////////////////////

#ifdef TC_INCLUDED
#include <taichi/util.h>
TC_NAMESPACE_BEGIN
#else
#include "serialization.h"
#define TC_P(x)
#define assert_info(x, info) assert(x);
#include <cassert>
#endif

template <typename T>
void release_memory(std::vector<T> &vec) {
  std::vector<T>().swap(vec);
}

namespace fem_interface {

////////////////////////////////////////////////////////////////////////////////
//                      FEM-related data structures                           //
////////////////////////////////////////////////////////////////////////////////

struct ForceOnNode {
  int coord[3];
  double f[3];

  TC_IO_DECL {
    TC_IO(coord);
    TC_IO(f);
  }
};

struct DirichletOnNode {
  int coord[3];
  int axis;
  double value;

  TC_IO_DECL {
    TC_IO(coord);
    TC_IO(axis);
    TC_IO(value);
  }
};

struct DirichletOnCell {
  int coord[3];
  int axis;
  double value;

  TC_IO_DECL {
    TC_IO(coord);
    TC_IO(axis);
    TC_IO(value);
  }
};

// Krylov solver parameters
struct KrylovSolverParameters {
  double tolerance;
  int max_iterations;
  int restart_iterations;
  bool print_residuals;

  TC_IO_DECL {
    TC_IO(tolerance);
    TC_IO(max_iterations);
    TC_IO(restart_iterations);
    TC_IO(print_residuals);
  }
};

template <int x_, int y_, int z_>
struct BlockSize {
  static constexpr int x = x_;
  static constexpr int y = y_;
  static constexpr int z = z_;
  static constexpr std::size_t size = x * y * z;
  static constexpr int get_dim(int i) {
    return (i == 0 ? x : (i == 1 ? y : z));
  }
};

template <typename T, typename block_size_>
class BlockedGrid {
 public:
  // TODO: remove resolution in the next major version
  int resolution[3];
  using block_size = block_size_;

  TC_IO_DEF(resolution, blocks);

  struct Block {
    int base_coordinates[3];
    using block_size = block_size_;
    T data[block_size::size];

    TC_IO_DECL {
      TC_IO(base_coordinates)
      if (TC_SERIALIZER_IS(TextSerializer)) {
        int index = 0;
        for (int i = 0; i < block_size::x; i++) {
          for (int j = 0; j < block_size::y; j++) {
            for (int k = 0; k < block_size::z; k++) {
              std::string index_str = "[" + std::to_string(i) + "," +
                                      std::to_string(j) + "," +
                                      std::to_string(k) + "]";
              serializer(
                  index_str.c_str(),
                  const_cast<typename std::decay<decltype(*this)>::type *>(this)
                      ->get(i, j, k));
              index++;
            }
          }
        }
      } else {
        TC_IO(data);
      }
    }

    void reset_data() {
      std::memset(data, 0, sizeof(data));
    }

    T &get(int i, int j, int k) {
      return data[i * block_size::y * block_size::z + j * block_size::y + k];
    }
  };

  void clear() {
    release_memory(blocks);
  }

  std::vector<Block> blocks;
};

using BlockedGridScalar = BlockedGrid<double, BlockSize<4, 4, 4>>;
using BlockedGridVector = BlockedGrid<double[4], BlockSize<4, 4, 4>>;

struct FEMDataBase {
  using T = double;
  using ScalarGrid = BlockedGridScalar;
  using VectorGrid = BlockedGridVector;
  static constexpr int dim = 3;

  // Simplified semantic versioning
  // Increase major version number when there is a breaking API change
  int version_major = FEM_INTERFACE_VERSION_MAJOR;
  // Change minor verison for incremental and non-breaking API change
  int version_minor = FEM_INTERFACE_VERSION_MINOR;
  // No patch version.

  std::map<std::string, std::string> extra_strings;
  std::map<std::string, double> extra_doubles;
  std::map<std::string, int> extra_ints;

  TC_IO_DECL {
    TC_IO(version_major);
    TC_IO(version_minor);
    assert(version_major == FEM_INTERFACE_VERSION_MAJOR);
    assert(version_minor <= FEM_INTERFACE_VERSION_MINOR);
    TC_IO(extra_strings);
    TC_IO(extra_ints);
    TC_IO(extra_doubles);
  }

  static_assert(sizeof(void *) == sizeof(uint64_t), "");
  void set_solver_state_ptr(void *ptr) {
    // TODO: add a member in the next version
    auto addr = reinterpret_cast<uint64_t>(ptr);
    uint32_t lower = uint32_t(addr & ((1UL << 32) - 1));
    uint32_t higher = uint32_t(addr >> 32);
    extra_ints["state_ptr_lower"] = lower;
    extra_ints["state_ptr_higher"] = higher;
  }

  void *get_solver_state_ptr() {
    if (extra_ints.find("state_ptr_lower") == extra_ints.end()) {
      return nullptr;
    }
    uint64_t lower = (uint32_t)extra_ints["state_ptr_lower"];
    uint64_t higher = (uint32_t)extra_ints["state_ptr_higher"];
    return reinterpret_cast<void *>((higher << 32) + lower);
  }

  int &get_ref_int(const std::string &key_name, int default_val) {
    if (extra_ints.find(key_name) == extra_ints.end()) {
      extra_ints[key_name] = default_val;
    }
    return extra_ints[key_name];
  }

  double &get_ref_double(const std::string &key_name, double default_val) {
    if (extra_doubles.find(key_name) == extra_doubles.end()) {
      extra_doubles[key_name] = default_val;
    }
    return extra_doubles[key_name];
  }
};

struct FEMInputs : public FEMDataBase {
  int resolution[dim];
  double dx;

  KrylovSolverParameters krylov;
  std::size_t preserved_buffer_size;

  // Cell parameters
  bool use_density_only;  // density only or mu & lambda?
  ScalarGrid density;
  ScalarGrid mu;
  ScalarGrid lambda;

  double global_mu, global_lambda;
  int mg_level, explicit_mg_level;
  int pre_and_post_smoothing_iter;
  int bottom_smoothing_iter;
  double jacobi_damping;

  std::string caller_method;

  // Boundary conditions
  std::vector<DirichletOnNode> dirichlet_nodes;
  std::vector<DirichletOnCell> dirichlet_cells;
  std::vector<ForceOnNode> forces;

  TC_IO_DECL {
    FEMDataBase::io(serializer);
    TC_IO(dx);
    TC_IO(preserved_buffer_size);
    TC_IO(use_density_only);
    if (use_density_only) {
      TC_IO(global_mu);
      TC_IO(global_lambda);
      TC_IO(density);
    } else {
      TC_IO(mu);
      TC_IO(lambda);
    }
    TC_IO(mg_level);
    TC_IO(explicit_mg_level);
    TC_IO(pre_and_post_smoothing_iter);
    TC_IO(bottom_smoothing_iter);
    TC_IO(jacobi_damping);
    TC_IO(resolution);
    TC_IO(forces);
    TC_IO(dirichlet_nodes);
    TC_IO(dirichlet_cells);
    TC_IO(krylov);
    TC_IO(caller_method);
  }

  int &keep_state() {
    std::string key_name = "keep_state";
    if (extra_ints.find(key_name) == extra_ints.end()) {
      extra_ints[key_name] = 0;
    }
    return extra_ints[key_name];
  }

  double &min_fraction() {
    std::string key_name = "min_fraction";
    if (extra_doubles.find(key_name) == extra_doubles.end()) {
      extra_doubles[key_name] = 1e-3f;
    }
    return extra_doubles[key_name];
  }

  int &defect_correction_iter() {
    return get_ref_int("defect_correction_iter", 10);
  }

  int &boundary_smoothing() {
    return get_ref_int("boundary_smoothing", 5);
  }

  int &forced_reuse() {
    return get_ref_int("forced_reuse", 0);
  }

  int &global_smoothing_iter() {
    return get_ref_int("global_smoothing_iter", 5);
  }

  double &minimum_stiffness() {
    return get_ref_double("minimum_stiffness", 0.0f);
  }

  double &minimum_density() {
    return get_ref_double("minimum_density", 0.0f);
  }

  double &penalty() {
    return get_ref_double("penalty", 0.0f);
  }

  int &defect_correction_cg_iter() {
    return get_ref_int("defect_correction_cg_iter", 3);
  }

  int &solver_type() {
    return get_ref_int("solver_type", 1);
  }

  // Note: these are temporary variables and do not need to be stored.
  double tmp_penalty = 0.0f;
  bool prepared = false;
  double tmp_minimum_stiffness;
  double tmp_minimum_density;

  void prepare_stiffness_mapping() {
    tmp_penalty = penalty();
    tmp_minimum_stiffness = minimum_stiffness();
    tmp_minimum_density = minimum_density();
    prepared = true;
  }

  double stiffness_mapping(double d) const {
    assert(prepared);
    if (tmp_penalty) {
      return (1 - tmp_minimum_stiffness) *
                 std::pow(std::max(tmp_minimum_density, d), tmp_penalty) +
             tmp_minimum_stiffness;
    } else {
      // for backward compatibility
      return d;
    }
  }

  void clear() {
    density.clear();
    mu.clear();
    lambda.clear();
    release_memory(dirichlet_nodes);
    release_memory(dirichlet_cells);
    release_memory(forces);
  }
};

struct FEMOutputs : public FEMDataBase {
  VectorGrid displacements;
  ScalarGrid residual_field;
  std::string info;
  std::vector<double> residuals;
  int error_code;
  bool success;

  TC_IO_DECL {
#if !defined(TC_PLATFORM_OSX)
    FEMDataBase::io(serializer);
    TC_IO(displacements);
    TC_IO(residual_field);
    TC_IO(info);
    TC_IO(residuals);
    TC_IO(error_code);
    TC_IO(success);
#else
    TC_NOT_IMPLEMENTED
#endif
  }

  void record_residual(double res) {
    residuals.push_back(res);
  }

  int get_num_iterations() const {
    return (int)residuals.size();
  }
};

class FEMInterface {
 public:
  using T = double;
  FEMInputs param;
  FEMOutputs outputs;
  BinaryOutputSerializer output_serializer;
  BinaryInputSerializer input_serializer;
  enum class Mode { CALLER, CALLEE };
  Mode mode;
  bool preserved = false;

  // This is actual data storage for the caller-side input serializer
  // and callee-side output_serializer.
  // This should be allocated in the callee-side and released in caller_side
  void *actual_data;
  void **output_data;

  FEMInterface() {
    // Caller size, write to data
    mode = Mode::CALLER;
  }

  // Changing interface
  // output_data now is actually a pointer to void* (i.e. void **), instead of
  // void *
  FEMInterface(void *input_data, void *output_data) {
    assert(input_data != nullptr);
    assert(output_data != nullptr);

    // Callee side, read from raw data
    mode = Mode::CALLEE;

    input_serializer.initialize(input_data);
    param.io(input_serializer);
    input_serializer.finalize();
    release_memory(input_serializer.data);
    this->output_data = reinterpret_cast<void **>(output_data);
  }

  ~FEMInterface() {
#if !defined(TC_PLATFORM_OSX)
    // This is where the interface actually serializes the solution
    if (mode == Mode::CALLEE) {
      preserve_output();
      TC_ASSERT(preserved);
      output_serializer("outputs", outputs);
      output_serializer.finalize();
      char *data = new char[output_serializer.data.size()];
      memcpy(data, output_serializer.data.data(),
             sizeof(char) * output_serializer.data.size());
      *output_data = data;
    }
#else
    TC_NOT_IMPLEMENTED
#endif
  }

  void preserve_output(std::size_t num_active_blocks) {
    TC_ASSERT(mode == Mode::CALLER);
    preserved = true;
    // 1M for extra info
    constexpr std::size_t tolerance = 1U << 20;
    param.preserved_buffer_size =
        tolerance + num_active_blocks * sizeof(FEMOutputs::VectorGrid::Block);
    TC_P(num_active_blocks);
    TC_P(param.preserved_buffer_size);
  }

  void preserve_output() {
    TC_ASSERT(mode == Mode::CALLEE);
    preserved = true;
    // Initialize without allocating space
    output_serializer.initialize();
  }

  // Note: this will release unless input data
  void run() {
    assert(mode == Mode::CALLER);
    finalize_input();
    assert(param.preserved_buffer_size != 0);
    // Release unless input data
    param.clear();
#ifndef TESTING
    fem_solve(&(*output_serializer.data.begin()), &actual_data);
#else
    fem_solve_test(&(*output_serializer.data.begin()), &actual_data);
#endif
    input_serializer.initialize(actual_data);
    input_serializer("outputs", outputs);
    delete[] reinterpret_cast<char *>(actual_data);
  }

 private:
  void finalize_input() {
    assert(mode == Mode::CALLER);
    output_serializer.initialize();
    param.io(output_serializer);
    output_serializer.finalize();
  }
};

static_assert(Serializer::has_io<FEMOutputs>::value, "");
static_assert(std::is_same<FEMOutputs, Serializer::has_io<FEMOutputs &>::T__>(),
              "");
static_assert(Serializer::has_io<FEMOutputs &>::value, "");
static_assert(Serializer::has_io<FEMOutputs &&>::value, "");
}  // namespace fem_interface
#ifdef TC_INCLUDED
TC_NAMESPACE_END
#endif
