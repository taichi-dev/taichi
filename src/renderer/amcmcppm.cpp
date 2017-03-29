/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#include <taichi/visual/renderer.h>
#include <taichi/math/averager.h>

#include "sppm.h"
#include "markov_chain.h"

TC_NAMESPACE_BEGIN

class AMCMCPPMRenderer : public SPPMRenderer {
public:

    virtual void initialize(const Config &config) override {
        SPPMRenderer::initialize(config);
        uniform_count = 0;
        accepted = 0;
        mutated = 0;
        mutation_strength = 1;
        mc_initialized = false;
        russian_roulette = config.get("russian_roulette", false);
    }

    void render_stage() override;

protected:

    struct MCMCState {
        // We only need to store the visibility chain
        AMCMCPPMMarkovChain chain;
    };

    // Current state on the *visibility* chain.
    MCMCState current_state;

    MCMCState create_new_uniform_state() {
        return MCMCState();
    }

    // The ratio of visible part of the PSS hypercube
    // Also the normalizer for the visibility chain
    RunningAverage normalizer;

    int64 uniform_count;
    int64 accepted;
    int64 mutated;

    real mutation_strength;
    bool mc_initialized;
};

void AMCMCPPMRenderer::render_stage() {
    hash_grid.clear_cache();
    eye_ray_pass();
    hash_grid.build_grid();
    // TODO:....
    normalizer.clear();
    if (!mc_initialized) {
        int64 emitted = 0;
        while (true) {
            emitted += 1;
            if (emitted % 100000 == 0) {
                printf("Warning: having difficulty initializing...\n");
                std::cout << emitted << " photons emitted without any visible one." << std::endl;
            }
            current_state = create_new_uniform_state();
            auto uniform_state_sequence = MCStateSequence(current_state.chain);
            bool visible = trace_photon(uniform_state_sequence, 0.0f);
            normalizer.insert((real)visible, 1);
            if (visible) {
                accepted = 1;
                uniform_count = 1;
                mc_initialized = true;
                break;
            }
        }
    }
    real last_r = 0.0f;
    for (int64 i = 0; i < num_photons_per_stage; i++) {
        // ----------------------------------------
        // We do 3 MCMC steps here:
        //   1. Mutate the visibility chain
        //   2. Mutate the uniform chain
        //   3. Replica exchange the two chains. The exchange prob. is 1 if uniform state visible, and 0 otherwise.
        // ----------------------------------------

        // So we can do steps 2 & 3 first. If the uniform state is visible, we always do replica exchange, thus
        // there's no need for step 1. Also, note that the uniform chain is always mutated in an uncorrelated
        // manner, i.e. uniform random choice from the unit hypercube, we do not need to store the uniform
        // chain state.

        // Step 2:
        // Mutate the uniform chain, using a completely random new state
        MCMCState uniform_state = create_new_uniform_state();
        auto uniform_state_sequence = MCStateSequence(uniform_state.chain);

        real weight = normalizer.get_average();
        // The pdf of sampling this point in the PSS hypercube is 1 while
        // the normalization factor for the visibility chain is normalizer.get_average().
        // So we do the corresponding scaling of contribution.

        if (trace_photon(uniform_state_sequence, weight)) {
            // Uniform state visible
            normalizer.insert(1, 1);
            // Step 3:
            // Always do replica exchange in this case
            current_state = uniform_state;
            uniform_count += 1;
        } else {
            // Uniform state invisible
            normalizer.insert(0, 1);
            // Step 1:
            // Mutate the visibility chain
            MCMCState candidate_state;
            mutated += 1;
            candidate_state.chain = current_state.chain.mutate(mutation_strength);
            auto candidate_state_sequence = MCStateSequence(candidate_state.chain);
            if (trace_photon(candidate_state_sequence, weight)) {
                current_state = candidate_state;
                accepted += 1;
            } else {
                auto rand = MCStateSequence(current_state.chain);
                trace_photon(rand, weight);
            }
        }
        photon_counter += 1;

        // Adaptive MCMC parameter update
        real r = (real)accepted / (real)mutated;
        last_r = r;
        mutation_strength = mutation_strength + (r - 0.234f) / mutated;
        mutation_strength = std::min(0.5f, std::max(0.0001f, mutation_strength));
    }

    P(mutated);
    P(accepted);
    P(last_r);
    P(mutation_strength);
    P(normalizer.get_average());
    stages += 1;

    for (auto &ind : image.get_region()) {
        image[ind] = (1.0f / (pi * radius2[ind]) / photon_counter * flux[ind]
                      + image_direct_illum[ind] * (1.0f / stages));
    }
}

TC_IMPLEMENTATION(Renderer, AMCMCPPMRenderer, "amcmcppm")

TC_NAMESPACE_END
