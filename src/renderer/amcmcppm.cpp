#include "renderer.h"
#include "sampler.h"
#include "bsdf.h"
#include "sppm.h"
#include "markov_chain.h"
#include "averager.h"

TC_NAMESPACE_BEGIN
    class AMCMCPPMRenderer : public SPPMRenderer {
    public:
        struct MCMCState {
            AMCMCPPMMarkovChain chain;
        };
        MCMCState current_state;
        virtual void initialize(Config &config) {
            SPPMRenderer::initialize(config);
            uniform_count = 0;
            accepted = 0;
            mutated = 0;
            mutation_strength = 1;
            mc_initialized = false;
			this->russian_roulette = config.get("russian_roulette", false);
        }

        void render_stage() override;
        MCMCState create_new_uniform_state() {
            return MCMCState();
        }
    protected:
        RunningAverager averager;
        long long uniform_count;
        long long accepted;
        long long mutated;
        real mutation_strength;
        bool mc_initialized;
    };

    void AMCMCPPMRenderer::render_stage() {
        hash_grid.clear_cache();
        eye_ray_pass();
        hash_grid.build_grid();
        if (!mc_initialized) {
            int emitted = 0;
            while (true) {
                emitted += 1;
                if (emitted % 100000 == 0) {
                    printf("Warning: having difficulty initializing... (%d)\n", emitted);
                }
                current_state = create_new_uniform_state();
                auto uniform_state_sequence = MCStateSequence(current_state.chain);
                bool visible = trace_photon(uniform_state_sequence);
                averager.insert((float)visible, 1);
                if (visible) {
                    accepted = 1;
                    uniform_count = 1;
                    mc_initialized = true;
                    break;
                }
            }
        }
        real last_r;
        for (int i = 0; i < num_photons_per_stage; i++) {
            MCMCState uniform_state = create_new_uniform_state();
            auto uniform_state_sequence = MCStateSequence(uniform_state.chain);
            if (trace_photon(uniform_state_sequence)) { // Visible
                averager.insert(1, 1);
                current_state = uniform_state;
                uniform_count += 1;
            } else {
                // uniform state invisible
                averager.insert(0, 1);
                MCMCState candidate_state;
                mutated += 1;
                //current_state.chain.print_states();
                candidate_state.chain = current_state.chain.mutate(mutation_strength);
                //candidate_state.chain.print_states();
                auto candidate_state_sequence = MCStateSequence(candidate_state.chain);
                if (trace_photon(candidate_state_sequence)) {
                    current_state = candidate_state;
                    accepted += 1;
                }
            }
            real r = (real)accepted / (real)mutated;
            last_r = r;
            mutation_strength = mutation_strength + (r - 0.234f) / mutated;
            mutation_strength = min(0.5f, max(0.0001f, mutation_strength));
        }
        P(mutated);
        P(accepted);
        P(last_r);
        P(mutation_strength);
        stages += 1;
        photon_counter = accepted + uniform_count;

        real normalizer = averager.get_average();
        for (auto &ind : image.get_region()) {
            image[ind] = normalizer * (1.0f / (pi * radius2[ind]) / photon_counter * flux[ind] + image_direct_illum[ind] * (1.0f / stages));
        }
    }

    TC_IMPLEMENTATION(Renderer, AMCMCPPMRenderer, "amcmcppm")

TC_NAMESPACE_END
