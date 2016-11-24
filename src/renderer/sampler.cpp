#include "sobol.h"
#include "sampler.h"

TC_NAMESPACE_BEGIN

	TC_INTERFACE_DEF(Sampler, "sampler");

    class PseudoRandomSampler : public Sampler {
    public:
        void initialize() { }

        real sample(int d, long long i) const {
            return rand();
        }
    };

	class PrimeList {
	public:
		PrimeList() {
            for (int i = 2; i <= 10000; i++) {
                if (is_prime(i)) {
                    primes.push_back(i);
                }
            }
            assert(primes.size() == 1229);
		}

		int get_prime(int i) {
			return primes[i];
		}

		int get_num_primes() {
			return (int)primes.size();
		}
	private:
        std::vector<int> primes;
	};

    class HaltonSampler : public Sampler {
    public:
        real sample(int d, long long i) const {
            assert(d < prime_list.get_num_primes());
            real val = hal(d, i + 1); // The first one is evil...
            return val;
        }

    private:
        inline int rev(const int i, const int p) const {
            return i == 0 ? i : p - i;
        }

        real hal(const int d, long long j) const {
            const int p = prime_list.get_prime(d);
            real h = 0.0, f = 1.0f / p, fct = f;
            while (j > 0) {
                h += rev(j % p, p) * fct;
                j /= p;
                fct *= f;
            }
            return h;
        }
		static PrimeList prime_list;
    };
	PrimeList HaltonSampler::prime_list;

    class SobolSampler : public Sampler {
    public:
        real sample(int d, long long i) const {
            return sobol::sample(i, d);
        }
    };

	TC_IMPLEMENTATION(Sampler, PseudoRandomSampler, "prand")
	TC_IMPLEMENTATION(Sampler, HaltonSampler, "halton")
	TC_IMPLEMENTATION(Sampler, SobolSampler, "sobol")

TC_NAMESPACE_END

