#include <taichi/math/array_3d.h>
#include <taichi/system/timer.h>
#include <taichi/dynamics/poisson_solver3d.h>

TC_NAMESPACE_BEGIN

template<typename T>
TC_FORCE_INLINE void benchmark(T &t, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                t[i][j][k] = real(i + j + k);
            }
        }
    }
    std::cout << t[int(rand() * n)][int(rand() * n)][int(rand() * n)] << std::endl;
}

const int n = 512;
real t[n][n][n];
Array3D<real> p(n, n, n);

void benchmark_array_3d() {
    benchmark(t, n);
    benchmark(t, n);
    benchmark(t, n);

    {
        Time::Timer _("Raw");
        for (int i = 0; i < 10; i++) {
            benchmark(t, n);
        }
    }
    {
        Time::Timer _("Array");
        for (int i = 0; i < 10; i++) {
            benchmark(p, n);
        }
    }
}

void test() {
    //benchmark_array_3d();
    create_instance<PoissonSolver3D>("mg")->test();
}

TC_NAMESPACE_END

