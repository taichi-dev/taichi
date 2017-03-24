/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#include <taichi/visual/sampler.h>

#include "sobol.h"

TC_NAMESPACE_BEGIN

class PseudoRandomSampler : public Sampler {
public:
    void initialize() { }

    real sample(int d, long long i) const {
        return rand();
    }
};

TC_IMPLEMENTATION(Sampler, PseudoRandomSampler, "prand")

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

TC_IMPLEMENTATION(Sampler, HaltonSampler, "halton")

class SobolSampler : public Sampler {
public:
    real sample(int d, long long i) const {
        return sobol::sample(i, d);
    }
};

TC_IMPLEMENTATION(Sampler, SobolSampler, "sobol")

TC_NAMESPACE_END

