/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#include <taichi/math/averager.h>

#include "bidirectional_renderer.h"
#include "hash_grid.h"
#include "markov_chain.h"

TC_NAMESPACE_BEGIN

class UPSRenderer : public BidirectionalRenderer {
  // Same as VCM except that we store eye paths...
 protected:
  int num_stages;
  real initial_radius;
  HashGrid hash_grid;
  std::vector<Path> eye_paths;
  std::vector<Path> eye_paths_for_connection;
  real radius;
  int n_samples_per_stage;
  Array2D<Vector3> bdpm_image;
  real alpha;
  bool use_vc;
  bool use_vm;
  bool shrinking_radius;

 public:
  virtual void initialize(const Config &config) override {
    BidirectionalRenderer::initialize(config);
    num_stages = 0;
    initial_radius = config.get<real>("initial_radius");
    use_vc = config.get("use_vc", true);
    use_vm = config.get("use_vm", true);
    alpha = config.get("alpha", 0.66667f);
    shrinking_radius = config.get("shrinking_radius", true);
    bdpm_image.initialize(Vector2i(width, height), Vector3(0.0_f));
    radius = initial_radius;
    n_samples_per_stage = width * height / stage_frequency;
  }

  PathContribution vertex_merge(const Path &full_light_path) {
    PathContribution pc;
    real radius2 = radius * radius;
    for (int num_light_vertices = 2;
         num_light_vertices <= (int)full_light_path.size();
         num_light_vertices++) {
      Path light_path(full_light_path.begin(),
                      full_light_path.begin() + num_light_vertices);
      Vector3 merging_pos = light_path.back().pos;
      int *begin = hash_grid.begin(merging_pos),
          *end = hash_grid.end(merging_pos);

      for (int *eye_path_id_pointer = begin; eye_path_id_pointer < end;
           eye_path_id_pointer++) {
        int eye_path_id = *eye_path_id_pointer;
        Path eye_path = eye_paths[eye_path_id];
        int path_length = (int)light_path.size() + (int)eye_path.size() - 2;
        int num_eye_vertices = (int)eye_path.size();
        Vertex merging_vertex_light = light_path.back();
        Vertex merging_vertex_eye = eye_path.back();
        assert_info(light_path.size() >= 1, "light path empty");
        assert_info(eye_path.size() >= 1, "eye path empty");
        if (SurfaceEventClassifier::is_delta(merging_vertex_eye.event) ||
            SurfaceEventClassifier::is_delta(merging_vertex_light.event)) {
          // Do not connect Delta BSDF
          continue;
        }
        Vector3 v = merging_vertex_eye.pos - merging_vertex_light.pos;
        if (min_path_length <= path_length && path_length <= max_path_length &&
            dot(merging_vertex_eye.normal, merging_vertex_light.normal) > eps &&
            dot(v, v) <= radius2) {
          // Screen coordinates
          Vector3 camera_direction =
              normalize(eye_path[1].pos - eye_path[0].pos);
          real screen_u, screen_v;
          camera->get_pixel_coordinate(camera_direction, screen_u, screen_v);
          if (!(0 <= screen_u && screen_u < 1 && 0 <= screen_v &&
                screen_v < 1)) {
            continue;
          }
          screen_u = clamp(screen_u, 0.0_f, 1.0_f);
          screen_v = clamp(screen_v, 0.0_f, 1.0_f);
          eye_path.back().connected = true;
          Path full_path;
          full_path.resize(num_eye_vertices + num_light_vertices -
                           1);  // note that last light vertex is deleted
          for (int i = 0; i < num_eye_vertices; i++)
            full_path[i] = eye_path[i];
          for (int i = 0; i < num_light_vertices - 1; i++)
            full_path[path_length - i] = light_path[i];

          // Evaluateh
          Vector3 f = path_throughput(full_path).cast<real>();
          if (f.max() <= 0.0_f) {
            // printf("f\n");
            continue;
          }
          double p = path_pdf(full_path, num_eye_vertices, num_light_vertices);
          if (p <= 0.0_f) {
            // printf("p\n");
            continue;
          }
          double w = mis_weight(full_path, num_eye_vertices, num_light_vertices,
                                use_vc, n_samples_per_stage);
          if (w <= 0.0_f) {
            // printf("w\n");
            continue;
          }
          Vector3 c = f * real(w / p);
          if (c.max() <= 0.0)
            continue;
          pc.push_back(Contribution(screen_u, screen_v, path_length, c));
        }
      }
    }
    return pc;
  }

  virtual void render_stage() override {
    radius = initial_radius *
             (shrinking_radius
                  ? (real)pow(num_stages + 1.0_f, -(1.0_f - alpha) / 2.0f)
                  : 1);
    vm_pdf_constant = pi * radius * radius;
    hash_grid.initialize(radius, width * height * 10 + 7);
    eye_paths.clear();
    eye_paths_for_connection.clear();
    // 1. Generate eye paths (importons)
    for (int k = 0; k < n_samples_per_stage; k++) {
      auto state_sequence = RandomStateSequence(sampler, sample_count * 2 + k);
      Path eye_path = trace_eye_path(state_sequence);
      eye_paths_for_connection.push_back(eye_path);
      if (use_vm) {
        // TODO: correct??? more efficient???
        for (int num_eye_vertices = 2; num_eye_vertices <= (int)eye_path.size();
             num_eye_vertices++) {
          Path partial_eye_path(eye_path.begin(),
                                eye_path.begin() + num_eye_vertices);
          hash_grid.push_back_to_all_cells_in_range(
              partial_eye_path.back().pos, radius, (int)eye_paths.size());
          eye_paths.push_back(partial_eye_path);
        }
      }
    }
    hash_grid.build_grid();

    // 2. Generate light paths (photons)
    for (int k = 0; k < n_samples_per_stage; k++) {
      auto state_sequence = RandomStateSequence(
          sampler, sample_count * 2 + n_samples_per_stage + k);
      Path light_path = trace_light_path(state_sequence);
      if (use_vm) {
        write_path_contribution(vertex_merge(light_path));
      }
      if (use_vc) {
        write_path_contribution(connect(eye_paths_for_connection[k], light_path,
                                        -1, -1,
                                        (int)use_vm * n_samples_per_stage));
      }
    }

    sample_count += n_samples_per_stage;
  }
};

TC_IMPLEMENTATION(Renderer, UPSRenderer, "ups");

class MCMCUPSRenderer : public UPSRenderer {
 public:
  struct MCMCState {
    AMCMCPPMMarkovChain chain;
    double sc;
    PathContribution pc;

    double p_star(int c) {
      if (c == con) {
        return pc.get_total_contribution();
      } else {
        return pc.get_total_contribution() > 0;
      }
    }
  };

  enum MarkovChainTag { con = 0, vis = 1 };
  MCMCState states[2];
  long long accepted;
  long long mutated;
  real mutation_strength;
  RunningAverage normalizers[2];
  std::vector<PathContribution> all_pcs[2];
  bool use_vis_chain;
  bool use_con_chain;
  bool chain_exchange;
  bool markov_chain_mis;
  bool mutation_expectation;
  real large_step_probabilities[2];
  real target_mutation_acceptance;
  real large_step_prob;

  virtual void initialize(const Config &config) override {
    UPSRenderer::initialize(config);
    large_step_prob = config.get("large_step_prob", 0.3f);
    large_step_probabilities[vis] = large_step_prob;
    use_vis_chain = config.get("use_vis_chain", true);
    use_con_chain = config.get("use_con_chain", true);
    large_step_probabilities[con] = use_vis_chain ? 0 : large_step_prob;
    TC_P(large_step_probabilities[con]);
    TC_P(large_step_probabilities[vis]);
    assert_info(use_vis_chain || use_con_chain,
                "Must use at least one Markov chain...");
    chain_exchange = config.get("chain_exchange", true);
    if (chain_exchange && !(use_vis_chain && use_con_chain)) {
      printf("Warning: Only one chain used, Chain Exchange turned OFF.\n");
      chain_exchange = false;
    }
    markov_chain_mis = config.get("markov_chain_mis", true);
    if (markov_chain_mis && !(use_vis_chain && use_con_chain)) {
      printf("Warning: Only one chain used, Markov chain MIS turned OFF.\n");
      markov_chain_mis = false;
    }
    mutation_expectation = config.get("mutation_expectation", true);
    target_mutation_acceptance =
        config.get("target_mutation_acceptance", 0.234f);
    TC_P(target_mutation_acceptance);
    mutation_strength = 0.001f;
    accepted = 1;
    mutated = 1;
  }

  virtual void render_stage() override {
    radius = initial_radius *
             (shrinking_radius
                  ? (real)pow(num_stages + 1.0_f, -(1.0_f - alpha) / 2.0f)
                  : 1);
    vm_pdf_constant = pi * radius * radius;
    hash_grid.initialize(radius, width * height * 10 + 7);
    eye_paths.clear();
    eye_paths_for_connection.clear();

    // 1. Generate eye paths (importons)
    for (int k = 0; k < n_samples_per_stage; k++) {
      auto state_sequence = RandomStateSequence(sampler, sample_count * 2 + k);
      Path eye_path = trace_eye_path(state_sequence);
      eye_paths_for_connection.push_back(eye_path);
      if (use_vm) {
        // TODO: correct??? more efficient???
        for (int num_eye_vertices = 2; num_eye_vertices <= (int)eye_path.size();
             num_eye_vertices++) {
          if (num_eye_vertices > 2)
            continue;  // NOTE:debug
          Path partial_eye_path(eye_path.begin(),
                                eye_path.begin() + num_eye_vertices);
          hash_grid.push_back_to_all_cells_in_range(
              partial_eye_path.back().pos, radius, (int)eye_paths.size());
          eye_paths.push_back(partial_eye_path);
        }
      }
    }
    hash_grid.build_grid();

    for (int i = 0; i < 2; i++) {
      normalizers[i].set_safe_value(1e-10f);
      normalizers[i].clear();
      all_pcs[i].clear();
    }
    normalizers[vis].insert(1e-6_f, 1e-5_f);

    // 2. Generate light paths (photons)

    // Initialize two chains
    long long initializing_count = 0;
    while (true) {
      initializing_count += 1;
      if (initializing_count % 100000 == 0) {
        printf("Warning: difficult initilization %lld.\n", initializing_count);
      }
      auto chain = AMCMCPPMMarkovChain();
      auto rand = MCStateSequence(chain);
      auto light_path = trace_light_path(rand);
      auto pc = vertex_merge(light_path);
      if (pc.get_total_contribution() > 0) {
        // Found... Give is to both chains because we are lazy...
        for (int i = 0; i < 2; i++) {
          states[i].chain = chain;
          states[i].pc = pc;
          states[i].sc = states[i].p_star(i);
        }
        break;
      }
    }

    // TODO: why unused?
    // int num_chains = int(use_vis_chain) + int(use_con_chain);
    RunningAverage photon_visibility;
    // TODO: deferred writting...
    for (int k = 0; k < n_samples_per_stage; k++) {
      MarkovChainTag u = (MarkovChainTag)(int(rand() * 2));
      if (!use_vis_chain && u == vis) {
        u = con;
      } else if (!use_con_chain && u == con) {
        u = vis;
      }
      MCMCState &previous_state = states[u];
      MCMCState new_state;
      bool is_large_step_done;
      if (rand() < large_step_probabilities[u]) {  // We use large step only on
                                                   // visibility chain
        // Large step
        new_state.chain = previous_state.chain.large_step();
        is_large_step_done = true;
      } else {
        // Small step (mutation)
        mutated += 1;
        new_state.chain = previous_state.chain.mutate(mutation_strength);
        is_large_step_done = false;
      }
      auto state_sequence = MCStateSequence(new_state.chain);
      Path light_path = trace_light_path(state_sequence);
      auto pc = vertex_merge(light_path);
      if (use_vc) {
        auto pc_vc = connect(eye_paths_for_connection[k], light_path, -1, -1,
                             (int)use_vm * n_samples_per_stage);
        for (auto &p : pc_vc.contributions) {
          pc.push_back(p);
        }
      }
      new_state.pc = pc;
      new_state.sc = new_state.p_star(u);

      double a = std::min(1.0, new_state.sc / max(1e-30, previous_state.sc));
      bool is_accepted = false;
      if (rand() < a) {
        if (!is_large_step_done) {
          // accepted mutation
          accepted += 1;
        }
        is_accepted = true;
      }
      MCMCState &current_state = is_accepted ? new_state : previous_state;
      if (is_large_step_done) {
        for (int i = 0; i < 2; i++) {
          normalizers[i].insert((real)new_state.p_star(i), 1);
        }
      }
      real current_state_weight =
          mutation_expectation ? real(a) : real(is_accepted);
      real last_state_weight = 1.0_f - current_state_weight;
      if (last_state_weight > 0 && previous_state.sc > 0) {
        all_pcs[u].push_back(previous_state.pc);
        real p[2] = {0.0_f};
        for (int i = 0; i < 2; i++) {
          if (markov_chain_mis) {
            p[i] =
                (real)previous_state.p_star(i) / normalizers[i].get_average();
          } else {
            p[i] = 1.0_f;
          }
        }
        auto s =
            last_state_weight / previous_state.sc * (p[u] / (p[0] + p[1])) * 2;
        assert_info(is_normal(s), "abnormal scaling");
        all_pcs[u].back().set_scaling(real(s));
      }
      if (current_state_weight > 0 && current_state.sc > 0) {
        all_pcs[u].push_back(current_state.pc);
        real p[2] = {0.0_f};
        for (int i = 0; i < 2; i++) {
          if (markov_chain_mis) {
            p[i] = real(current_state.p_star(i) / normalizers[i].get_average());
          } else {
            p[i] = 1.0_f;
          }
        }
        auto s = current_state_weight / current_state.sc *
                 (p[u] / (p[0] + p[1])) * 2;
        assert_info(is_normal(s), "abnormal scaling " + std::to_string(s));
        all_pcs[u].back().set_scaling(real(s));
      }
      if (is_accepted) {
        states[u] = new_state;
      }
      if (u == vis) {
        photon_visibility.insert((real)current_state.sc, 1);
      }
      if (chain_exchange) {
        // Replica Exchange
        double r = std::min(
            1.0, states[vis].p_star(con) / max(1e-30, states[con].p_star(con)));
        if (rand() < r) {
          std::swap(states[con], states[vis]);
          for (int i = 0; i < 2; i++) {
            states[i].sc = states[i].p_star(i);
          }
        }
      }
      // Update mutation_strength
      real ratio_accepted = (real)accepted / (real)mutated;
      mutation_strength =
          mutation_strength +
          (ratio_accepted - target_mutation_acceptance) / mutated;
      mutation_strength = std::min(10.0_f, max(1e-7_f, mutation_strength));
    }
    real ratio_accepted = (real)accepted / (real)mutated;
    TC_P(ratio_accepted);
    TC_P(mutated);
    TC_P(accepted);
    TC_P(mutation_strength);
    TC_P(photon_visibility.get_average());
    for (int u = 0; u < 2; u++) {
      real b = normalizers[u].get_average();
      TC_P(b);
      for (auto &pc : all_pcs[u]) {
        write_path_contribution(pc, b);
      }
    }
    sample_count += n_samples_per_stage;
  }
};

TC_IMPLEMENTATION(Renderer, MCMCUPSRenderer, "mcmcups");

TC_NAMESPACE_END
