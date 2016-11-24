#pragma once
#include "math/linalg.h"
#include "sampler.h"
#include <vector>
#include <memory>
#include <string>

TC_NAMESPACE_BEGIN

    class MarkovChain {
    protected:
        std::vector<real> states;
    public:
        virtual real get_state(int d) {
            while ((int)states.size() <= d) {
                states.push_back(rand());
            }
            return states[d];
        }
        virtual void print_states() {
            printf("chain = ");
            for (int i = 0; i < (int)states.size(); i++) {
                printf("%f ", states[i]);
            }
            printf("\n");
        }
    };
    class MCStateSequence : public StateSequence {
    private:
        MarkovChain &mc;
    public:
        MCStateSequence(MarkovChain &mc) : mc(mc) {}
        real sample() override {
            return mc.get_state(cursor++);
        }
    };


    template <bool starts_from_screen>
    class PSSMarkovChain : public MarkovChain {
    private:
        real resolutions[int(starts_from_screen) * 2];
    public:
        PSSMarkovChain() {
            memset(resolutions, 0, sizeof(resolutions));
        }
        PSSMarkovChain(real resolution_x, real resolution_y) {
            static_assert(starts_from_screen, "");
            resolutions[0] = resolution_x;
            resolutions[1] = resolution_y;
        }
        PSSMarkovChain<starts_from_screen> large_step() const {
            if (starts_from_screen) {
                return PSSMarkovChain<starts_from_screen>(resolutions[0], resolutions[1]);
            } else {
                return PSSMarkovChain<starts_from_screen>();
            }
        }
        PSSMarkovChain<starts_from_screen> mutate(real strength) const {
            PSSMarkovChain<starts_from_screen> result(*this);
            if (starts_from_screen) {

            }
            for (int i = 2 * (int)starts_from_screen; i < (int)states.size(); i++) {
                result.states[i] = perturb(states[i], strength);
            }
            return result;
        }

    protected:
        inline static real perturb(const real value, const real strength) {
            real result;
            real r = rand();
            if (r < 0.5f) {
                r = r * 2.0f;
                result = value + pow(r, 1.0f / strength + 1.0f);
            } else {
                r = (r - 0.5f) * 2.0f;
                result = value - pow(r, 1.0f / strength + 1.0f);
            }
            result -= floor(result);
            return result;
        }
    };

    class AMCMCPPMMarkovChain : public MarkovChain {
    public:
        AMCMCPPMMarkovChain large_step() const {
            return AMCMCPPMMarkovChain();
        }
        AMCMCPPMMarkovChain mutate(real strength) const {
            AMCMCPPMMarkovChain result(*this);
            for (int i = 0; i < (int)states.size(); i++) {
                result.states[i] = perturb(states[i], strength);
            }
            return result;
        }

    protected:
        // TODO: what's the difference between this and the one purposed in the paper?
        inline static real perturb(const real value, const real strength) {
            real result;
            real r = rand();
            if (r < 0.5f) {
                r = r * 2.0f;
                result = value + pow(r, 1.0f / strength + 1.0f);
            } else {
                r = (r - 0.5f) * 2.0f;
                result = value - pow(r, 1.0f / strength + 1.0f);
            }
            result -= floor(result);
            return result;
        }
    };
TC_NAMESPACE_END

