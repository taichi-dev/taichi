/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2017 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

//#####################################################################
// Copyright (c) 2014, the authors of submission papers_0203
// The following code is based the code provided with the SPGrid paper:
//      http://pages.cs.wisc.edu/~sifakis/papers/SPGrid.pdf
//
// Modified and merged into Taichi by Yuanming Hu
//#####################################################################

//#####################################################################
// Copyright (c) 2014  SPGrid Authors
//
// Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
//   * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
//   * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
//     other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING,
// BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
// SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//#####################################################################

// Minimalist SPGrid

#if !defined(_WIN64) && !defined(TC_DISABLE_SSE)
// let's support Unix-like (Linux and OS X) first.

#include <taichi/system/benchmark.h>
#include <taichi/math/math_simd.h>
#include <sys/mman.h>
#include <unistd.h>

namespace std {
inline std::string to_string(const void *ptr) {
    std::stringstream ss;
    ss << std::hex << reinterpret_cast<size_t >(ptr);
    return ss.str();
}
}

TC_NAMESPACE_BEGIN

#define HASWELL

template <uint v>
struct Log2;

template <>
struct Log2<0> {
    enum {
        value = 0
    };
};

template <uint v>
struct Log2 {
    enum {
        value = 1 + Log2<(v >> 1)>::value
    };
};

#ifdef HASWELL

unsigned long Bit_Spread(const unsigned long data, const unsigned long mask) { return _pdep_u64(data, mask); }

unsigned long Bit_Spread(const unsigned int data, const unsigned long mask) {
    unsigned long uldata = data;
    return _pdep_u64(uldata, mask);
}

unsigned long Bit_Spread(const int data, const unsigned long mask) {
    union {
        signed long sldata;
        unsigned long uldata;
    };
    sldata = data;
    return _pdep_u64(uldata, mask);
}

#endif

int Bit_Pack(const unsigned long data, const unsigned long mask) {
    union {
        signed long slresult;
        unsigned long ulresult;
    };
#ifdef HASWELL
    ulresult = _pext_u64(data, mask);
#else
    unsigned long uldata = data;
    int count = 0;
    ulresult = 0;
    for (unsigned long bit = 1; bit; bit <<= 1)
        if (bit & mask) ulresult |= (uldata & bit) >> count; else count++;
#endif
    return (int)slresult;
}

// Resolution
template <typename T>
class SPGrid {
private:
    uint64 masks[3];
    T *data;

public:

    enum {
        data_bits = Log2<sizeof(T)>::value,            // Bits needed for indexing individual bytes within type T
        block_bits = 12 - data_bits,                   // Bits needed for indexing data elements within a block
        block_zbits =
        block_bits / 3 + (block_bits % 3 > 0), // Bits needed for the z-coordinate of a data elements within a block
        block_ybits =
        block_bits / 3 + (block_bits % 3 > 1), // Bits needed for the y-coordinate of a data elements within a block
        block_xbits =
        block_bits / 3                   // Bits needed for the x-coordinate of a data elements within a block
    };

    enum {
        elements_per_block = 1u << block_bits
    };

    enum {
        log2_field = 4
    };

    enum { // Bit masks for the lower 12 bits of memory addresses (element indices within a page)
        element_zmask = ((1 << block_zbits) - 1) << log2_field,
        element_ymask = ((1 << block_ybits) - 1) << (log2_field + block_zbits),
        element_xmask = ((1 << block_xbits) - 1) << (log2_field + block_zbits + block_ybits)
    };

    enum { // Bit masks for the upper 52 bits of memory addresses (page indices)
        page_zmask = (0x9249249249249249UL << (3 - block_bits % 3)) & 0xfffffffffffff000UL,
        page_ymask = (0x2492492492492492UL << (3 - block_bits % 3)) & 0xfffffffffffff000UL,
        page_xmask = (0x4924924924924924UL << (3 - block_bits % 3)) & 0xfffffffffffff000UL
    };

    enum { // Bit masks for aggregate addresses
        zmask = page_zmask | (unsigned long)element_zmask,
        ymask = page_ymask | (unsigned long)element_ymask,
        xmask = page_xmask | (unsigned long)element_xmask
    };

    void initialize() {
    }

    uint64 map_coord(int i, int j, int k) const {
        return Bit_Spread(i, masks[0]) |
               Bit_Spread(j, masks[1]) |
               Bit_Spread(k, masks[2]);
    }

    T &operator()(int i, int j, int k) {
        return data[map_coord(i, j, k)];
    }

    const T &operator()(int i, int j, int k) const {
        return data[map_coord(i, j, k)];
    }


    static void *allocate(size_t size) {
        void *ptr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        if (ptr == MAP_FAILED) error("Failed to allocate " + std::to_string(size) + " bytes.");
        if (0xfffUL & (unsigned long)ptr) error(
                "Allocated pointer value " + std::to_string((size_t)ptr) + " is not page-aligned.");
        return ptr;
    }


    static void deallocate(void *data, const size_t size) {
        if (munmap(data, size) != 0) error("Failed to deallocate " + std::to_string(size) + " bytes");
    }

    static void Deactivate_Page(void *data, const size_t size) {
        if (0xfffUL & (uint64_t)data) error("Allocated pointer value " + std::to_string(data) + " is not page-aligned");
        int ret = madvise(data, size, MADV_DONTNEED);
        if (ret < 0) {
            char buffer[256];
            // char *error_message = strerror_r(errno, buffer, 256);
            //error("Failed to deallocate " + std::to_string(size) + " bytes, ERROR: " + error_message);
            error("Failed to deallocate " + std::to_string(size) + " bytes, ERROR: unknown (message disabled)");
        }
    }

    static bool Check_Address_Resident(const void *addr) {
        void *page_addr = reinterpret_cast<void *>(reinterpret_cast<unsigned long>(addr) & 0xfffffffffffff000UL);
        char status;
        if (mincore(page_addr, 4096, &status) == -1)
            switch (errno) {
                case ENOMEM: error("In Check_Address_Resident() : Input address " + std::to_string(addr) +
                                   " has not been mapped");
                default: error(" In Check_Address_Resident() : mincore() failed with errno=" + std::to_string(errno));
            }
        if (!status) return false;
        return true;
    }

    static void Check_Compliance() {
        if (sysconf(_SC_PAGESIZE) != 4096) error("Page size different than 4KB detected");
        if (sizeof(unsigned long) != 8) error("unsigned long is not 64-bit integer");
        if (sizeof(size_t) != 8) error("size_t is not 64-bit long");
        if (sizeof(void *) != 8) error("void* is not 64-bit long");
        typedef enum {
            dummy = 0xffffffffffffffffUL
        } Long_Enum;
        if (sizeof(Long_Enum) != 8) error("Missing support for 64-bit enums");
    }
};

class SPGridBenchmark : public Benchmark {
private:
    int n;
    bool brute_force;
    std::vector<Vector3> input;
public:
    void initialize(const Config &config) override {
        Benchmark::initialize(config);
        brute_force = config.get_bool("brute_force");
        input.resize(workload);
        for (int i = 0; i < workload; i++) {
            input[i] = Vector3(rand(), rand(), rand()) + Vector3(1.0f);
        }
    }

protected:
    real sum_spgrid(Vector3 p) const {
    }

    real sum_dense(Vector3 p) const {
    }

    void iterate() override {
//        real ret = 0.0f;
//        if (brute_force) {
//            for (int i = 0; i < workload; i++) {
//                ret += sum_brute_force(input[i]);
//            }
//        } else {
//            for (int i = 0; i < workload; i++) {
//                ret += sum_simd(input[i]);
//            }
//        }
//        dummy = (int)(ret);
    }

public:
    bool test() const override {
        for (int i = 0; i < workload; i++) {
//            real bf_result = sum_brute_force(input[i]);
//            real simd_result = sum_simd(input[i]);
//            if (abs(bf_result - simd_result) > 1e-6) {
//                printf("%f %f\n", bf_result, simd_result);
//                error("value mismatch");
//            }
        }
        return true;
    }
};

//TC_IMPLEMENTATION(Benchmark, KernelCalculationBenchmark, "mpm_kernel");

TC_NAMESPACE_END
#endif
