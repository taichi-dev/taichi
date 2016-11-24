#include "renderer.h"
#include "sampler.h"
#include "bsdf.h"
#include "markov_chain.h"

TC_NAMESPACE_BEGIN

    class Volume {
    public:
        void initialize(const Config &config) {
            printf("Info: Volumetric rendering is turned ON. Note that PT & MCMCPT are only renderers that support this.\n");
            printf("      This may lead to different output by different renderers.\n");
            this->volumetric_scattering = config.get("volumetric_scattering", 0.001f);
            this->volumetric_absorption = config.get("volumetric_absorption", 0.001f);
        }
        Vector3 calculate_volumetric_direct_lighting(const Vector3 &in_dir, const Vector3 &orig, StateSequence &rand);

        Vector3 calculate_direct_lighting(const Vector3 &in_dir, const IntersectionInfo &info, const BSDF &bsdf,
                                          StateSequence &rand, const Triangle &tri);

        real sample_volumetric_distance(StateSequence &rand) const {
            return -log(1 - rand()) / (volumetric_scattering + volumetric_absorption);
        }

        bool is_event_scattering(StateSequence &rand) const {
            return rand() < volumetric_scattering / (volumetric_scattering + volumetric_absorption);
        }

        static Vector3 sample_phase(StateSequence &rand) {
            real x = rand() * 2 - 1;
            real phi = rand() * 2 * pi;
            real yz = sqrt(1 - x * x);
            return Vector3(x, yz * cos(phi), yz * sin(phi));
        }

        real get_attenuation(real dist) const {
            return exp(-dist * (volumetric_scattering + volumetric_absorption));
        }
        real volumetric_scattering;
        real volumetric_absorption;
    };

TC_NAMESPACE_END

