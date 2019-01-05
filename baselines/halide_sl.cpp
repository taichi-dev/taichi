// sudo apt-get install libjpeg-dev
// sudo apt-get install libpng16-dev
//

#include <Halide.h>
#include <halide_benchmark.h>
#include <iostream>

using namespace Halide;
using namespace Halide::Tools;

int main(int argc, char *argv[]) {
    constexpr int n = 1024, nchannels = 8;
    Buffer<float> motion_field(2, n, n);
    Buffer<float> input(nchannels, n, n);
    Func output("output");
    Var x("x"), y("y"), c("c");
    Expr new_x = x + motion_field(0, x, y);
    Expr new_y = y + motion_field(1, x, y);
    Expr fx = clamp(cast<int>(floor(new_x)), 0, n - 2);
    Expr fy = clamp(cast<int>(floor(new_y)), 0, n - 2);
    Expr wx = new_x - fx;
    Expr wy = new_y - fy;
    output(c, x, y) = input(c, fx    , fy    ) * (1.f - wx) * (1.f - wy) +
                      input(c, fx    , fy + 1) * (1.f - wx) * (      wy) +
                      input(c, fx + 1, fy    ) * (      wx) * (1.f - wy) +
                      input(c, fx + 1, fy + 1) * (      wx) * (      wy);
    // There's only one func here, not much to schedule
    output.compute_root()
          .reorder(x, c, y)
          .parallel(y).vectorize(x, 8);

    output.compile_jit();
    // Setup input & motion field
    for (int y = 0; y < n; y++) {
        for (int x = 0; x < n; x++) {
            for (int c = 0; c < nchannels; c++) {
                input(c, x, y) = y % 128 / 128.f;
            }
            float s = 20.f / n;
            motion_field(0, x, y) =   s * (x - n / 2);
            motion_field(1, x, y) = - s * (y - n / 2);
        }
    }

    // Benchmarking
    Buffer<float> out_buf(nchannels, n, n);
    int timing_iterations = 20;
    double best_time = benchmark(timing_iterations, 10, [&]() {
        output.realize(out_buf);
    });

    std::cout << "best_time:" << best_time << std::endl;

    return 0;
}
