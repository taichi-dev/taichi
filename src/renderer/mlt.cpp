/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#include <taichi/math/discrete_sampler.h>

#include "bidirectional_renderer.h"
#include "markov_chain.h"

TC_NAMESPACE_BEGIN

class PSSMLTMarkovChain : public MarkovChain {
 public:
  real resolution_x, resolution_y;

  PSSMLTMarkovChain() : PSSMLTMarkovChain(0, 0) {
  }

  PSSMLTMarkovChain(real resolution_x, real resolution_y)
      : resolution_x(resolution_x), resolution_y(resolution_y) {
  }

  PSSMLTMarkovChain large_step() const {
    PSSMLTMarkovChain result(resolution_x, resolution_y);
    return result;
  }

  PSSMLTMarkovChain mutate() const {
    PSSMLTMarkovChain result(*this);
    // Screen coordinates
    real delta_pixel = 2.0f / (resolution_x + resolution_y);
    result.get_state(2);
    result.states[0] = perturb(result.states[0], delta_pixel, 0.1f);
    result.states[1] = perturb(result.states[1], delta_pixel, 0.1f);
    // Events
    for (int i = 2; i < (int)result.states.size(); i++)
      result.states[i] =
          perturb(result.states[i], 1.0_f / 1024.0f, 1.0_f / 64.0f);
    return result;
  }

 protected:
  inline static real perturb(const real value, const real s1, const real s2) {
    real result;
    real r = rand();
    if (r < 0.5f) {
      r = r * 2.0f;
      result = value + s2 * exp(-log(s2 / s1) * r);
    } else {
      r = (r - 0.5f) * 2.0f;
      result = value - s2 * exp(-log(s2 / s1) * r);
    }
    result -= floor(result);
    return result;
  }
};

class PSSMLTRenderer : public BidirectionalRenderer {
 private:
  struct MCMCState {
    PSSMLTMarkovChain chain;
    PathContribution pc;
    real sc;

    MCMCState() {
    }

    MCMCState(const PSSMLTMarkovChain &chain,
              const PathContribution &pc,
              real sc)
        : chain(chain), pc(pc), sc(sc) {
    }
  };

  MCMCState current_state;
  bool first_stage_done = false;
  // Normalizer
  real b;

 protected:
  real large_step_prob;

 public:
  virtual void initialize(const Config &config) override {
    BidirectionalRenderer::initialize(config);
    large_step_prob = config.get("large_step_prob", 0.3f);
  }

  real estimate_b() {
    real sum = 0;
    int n_samples = width * height;
    for (int k = 0; k < n_samples; k++) {
      auto state_sequence = RandomStateSequence(sampler, k);
      Path eye_path = trace_eye_path(state_sequence);
      Path light_path = trace_light_path(state_sequence);
      PathContribution pc = connect(eye_path, light_path);
      sum += scalar_contribution_function(pc);
    }
    return sum / n_samples;
  }

  real scalar_contribution_function(const PathContribution &pc) {
    real ret = 0.0_f;
    for (auto &contribution : pc.contributions) {
      ret = max(ret, scalar_contribution_function(contribution));
    }
    return ret;
  }

  real scalar_contribution_function(const Contribution &contribution) {
    return contribution.c.max();  // TODO: change to luminance
  }

  virtual PathContribution get_path_contribution(MarkovChain &mc) {
    auto state_sequence = MCStateSequence(mc);
    Path eye_path = trace_eye_path(state_sequence);
    Path light_path = trace_light_path(state_sequence);
    return connect(eye_path, light_path);
  }

  virtual void render_stage() override {
    if (!first_stage_done) {
      b = estimate_b();
      TC_P(b);
      current_state.chain = PSSMLTMarkovChain((real)width, (real)height);
      current_state.pc = get_path_contribution(current_state.chain);
      current_state.sc = scalar_contribution_function(current_state.pc);
      first_stage_done = true;
    }
    MCMCState new_state;
    for (int k = 0; k < width * height / stage_frequency; k++) {
      real is_large_step;
      if (rand() <= large_step_prob) {
        new_state.chain = current_state.chain.large_step();
        is_large_step = 1.0;
      } else {
        new_state.chain = current_state.chain.mutate();
        is_large_step = 0.0;
      }
      new_state.pc = get_path_contribution(new_state.chain);
      new_state.sc = scalar_contribution_function(new_state.pc);
      double a = 1.0;
      if (current_state.sc > 0.0) {
        a = clamp(new_state.sc / current_state.sc, 0.0_f, 1.0_f);
      }
      // accumulate samples with mean value substitution and MIS
      if (new_state.sc > 0.0) {
        write_path_contribution(
            new_state.pc,
            real((a + is_large_step) / (new_state.sc / b + large_step_prob)));
      }
      if (current_state.sc > 0.0) {
        write_path_contribution(
            current_state.pc,
            real((1.0 - a) / (current_state.sc / b + large_step_prob)));
      }
      // conditionally accept the chain
      if (rand() <= a) {
        current_state = new_state;
      }
      sample_count += 1;
    }
  }
};

class MMLTRenderer : public PSSMLTRenderer {
 protected:
  class MMLTMarkovChain : public PSSMLTMarkovChain {
   private:
    real technique_state;

   public:
    MMLTMarkovChain() : MMLTMarkovChain(0, 0) {
    }

    MMLTMarkovChain(int resolution_x, int resolution_y)
        : PSSMLTMarkovChain((real)resolution_x, (real)resolution_y) {
      technique_state = rand();
    }

    MMLTMarkovChain large_step() const {
      return MMLTMarkovChain((int)resolution_x, (int)resolution_y);
    }

    MMLTMarkovChain mutate() const {
      MMLTMarkovChain result(*this);
      // Pixel location
      real delta_pixel = 2.0f / (resolution_x + resolution_y);
      result.get_state(2);
      result.states[0] = perturb(result.states[0], delta_pixel, 0.1f);
      result.states[1] = perturb(result.states[1], delta_pixel, 0.1f);
      result.technique_state =
          perturb(result.technique_state, 1.0 / 1024.0f, 1.0 / 64.0f);
      // Events
      for (int i = 2; i < (int)result.states.size(); i++)
        result.states[i] =
            perturb(result.states[i], 1.0_f / 1024.0f, 1.0_f / 64.0f);
      return result;
    }

    real get_technique_state() {
      return technique_state;
    }
  };

  struct MCMCState {
    MMLTMarkovChain chain;
    PathContribution pc;
    real sc, weight;

    MCMCState() {
      weight = 1.0_f;
    }

    MCMCState(const MMLTMarkovChain &chain, const PathContribution &pc, real sc)
        : chain(chain), pc(pc), sc(sc) {
      weight = 1.0_f;
    }
  };

  std::vector<MCMCState> current_states;
  bool first_stage_done = false;
  DiscreteSampler path_length_sampler;
  std::vector<real> normalizers;

 public:
  virtual void initialize(const Config &config) override {
    PSSMLTRenderer::initialize(config);
    current_states.resize(max_path_length + 1);
    normalizers.resize(max_path_length + 1);
  }

  void estimate_normalizers() {
    real sum = 0;
    int n_samples = width * height;
    std::vector<real> intensities(max_path_length + 1, 0.0_f);
    std::vector<long long> samples(max_path_length + 1, 0LL);
    for (int k = 0; k < n_samples; k++) {
      auto state_sequence = RandomStateSequence(sampler, k);
      Path eye_path = trace_eye_path(state_sequence);
      Path light_path = trace_light_path(state_sequence);
      PathContribution pc = connect(eye_path, light_path);
      for (auto &contribution : pc.contributions) {
        intensities[contribution.path_length] +=
            scalar_contribution_function(contribution);
        samples[contribution.path_length] += 1;
      }
    }
    for (int k = min_path_length; k <= max_path_length; k++) {
      normalizers[k] = intensities[k] / n_samples / (k + 1);
    }
  }

  PathContribution get_path_contribution(MMLTMarkovChain &mc, int path_length) {
    auto state_sequence = MCStateSequence(mc);
    Path eye_path = trace_eye_path(state_sequence);
    Path light_path = trace_light_path(state_sequence);
    int t = std::min(path_length,
                     (int)floor(mc.get_technique_state() * (path_length + 1))) +
            1,
        s = path_length - t + 1;
    assert_info(0 <= t && t <= path_length + 1,
                "Invalid eye path length: " + std::to_string(t) +
                    " state: " + std::to_string(mc.get_technique_state()));
    return connect(eye_path, light_path, t, s);
  }

  void initialize_path_length_sampler() {
    estimate_normalizers();
    for (int i = min_path_length; i <= max_path_length; i++) {
      if (normalizers[i] == 0.0_f) {
        // No path of such length
        continue;
      }
      while (true) {
        current_states[i].chain = MMLTMarkovChain(width, height);
        current_states[i].pc =
            get_path_contribution(current_states[i].chain, i);
        current_states[i].sc =
            scalar_contribution_function(current_states[i].pc);
        if (current_states[i].sc > 0) {
          break;
        }
      }
    }
    path_length_sampler.initialize(normalizers);
  }

  virtual void render_stage() override {
    if (!first_stage_done) {
      initialize_path_length_sampler();
      first_stage_done = true;
    }
    MCMCState new_state;
    for (int k = 0; k < width * height / stage_frequency; k++) {
      real path_length_pdf;
      int path_length = path_length_sampler.sample(rand(), path_length_pdf);
      real is_large_step;
      MCMCState &current_state = current_states[path_length];
      if (rand() < large_step_prob) {
        new_state.chain = current_state.chain.large_step();
        is_large_step = 1.0;
      } else {
        new_state.chain = current_state.chain.mutate();
        is_large_step = 0.0;
      }
      new_state.pc = get_path_contribution(new_state.chain, path_length);
      new_state.sc = scalar_contribution_function(new_state.pc);
      real a = 1.0;
      if (current_state.sc > 0.0) {
        a = clamp(new_state.sc / current_state.sc, 0.0_f, 1.0_f);
      }
      real factor = (path_length + 1) / path_length_pdf;
      // accumulate samples with mean value substitution and MIS
      if (new_state.sc > 0.0) {
        write_path_contribution(
            new_state.pc,
            factor * (a + is_large_step) /
                (new_state.sc / normalizers[path_length] + large_step_prob));
      }
      if (current_state.sc > 0.0) {
        write_path_contribution(
            current_state.pc,
            factor * real(1.0 - a) /
                real((current_state.sc / normalizers[path_length] +
                      large_step_prob)));
      }
      // conditionally accept the chain
      if (rand() <= a) {
        current_state = new_state;
      }
      sample_count += 1;
    }
  }
};

TC_IMPLEMENTATION(Renderer, PSSMLTRenderer, "pssmlt");

class DWMMLTRenderer : public MMLTRenderer {
 public:
  void initialize(Config &config) {
    MMLTRenderer::initialize(config);
    large_step_prob = 0.0_f;
  }

  virtual void render_stage() override {
    if (!first_stage_done) {
      initialize_path_length_sampler();
      first_stage_done = true;
    }
    MCMCState new_state;
    for (int k = 0; k < width * height / stage_frequency; k++) {
      real path_length_pdf;
      int path_length = path_length_sampler.sample(rand(), path_length_pdf);
      MCMCState &current_state = current_states[path_length];
      // TC_P(current_state.weight);
      new_state.chain = current_state.chain.mutate();
      new_state.pc = get_path_contribution(new_state.chain, path_length);
      new_state.sc = scalar_contribution_function(new_state.pc);
      bool accepted = false;
      real theta = 0.1f, r, a;
      if (current_state.sc > 0) {
        r = current_state.weight * new_state.sc / current_state.sc;
        a = r / (r + theta);
        if (rand() < r) {
          accepted = true;
        }
      } else {
        a = 0;
      }
      if (accepted) {
        current_state = new_state;
        current_state.weight = r / a;
      } else {
        current_state.weight = current_state.weight / (1 - a);
      }
      real factor = (path_length + 1) / path_length_pdf;
      if (current_state.sc > 0.0) {
        write_path_contribution(
            current_state.pc,
            current_state.weight * factor /
                (current_state.sc / normalizers[path_length]));
      }
      sample_count += 1;
    }
  }
};

TC_IMPLEMENTATION(Renderer, MMLTRenderer, "mmlt");

TC_NAMESPACE_END
