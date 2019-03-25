//  88-Line 2D Moving Least Squares Material Point Method (MLS-MPM)
// [Explained Version by David Medina]

// Uncomment this line for image exporting functionality
#define TC_IMAGE_IO

// Note: You DO NOT have to install taichi or taichi_mpm.
// You only need [taichi.h] - see below for instructions.
#include "taichi.h"

using namespace taichi;
using Vec = Vector2;
using Mat = Matrix2;

// Window
const int window_size = 800;

const int q = 2;
// Grid resolution (cells)
const int n = 80 * q;

const real dt = 5e-5_f / q;
const real frame_dt = 1e-3_f;
const real dx = 1.0_f / n;
const real inv_dx = 1.0_f / dx;

// Snow material properties
const auto particle_mass = 1.0_f;
const auto vol = 1.0_f;  // Particle Volume
const auto E = 5e4_f;    // Young's Modulus
const auto nu = 0.2_f;   // Poisson ratio
const bool plastic = false;

// Initial Lamé parameters
const real mu_0 = 0.3 * E;
const real lambda_0 = 0.5 * E;  // * nu / ((1+nu) * (1 - 2 * nu));
real max_norm;

struct Particle {
  // Position and velocity
  Vec x, v;
  // Deformation gradient
  Mat F;
  // Affine momentum from APIC
  Mat C;
  // Determinant of the deformation gradient (i.e. volume)
  real Jp;
  // Color
  int c;
  real fluid;

  Particle(Vec x, int c, real fluid = 0)
      : x(x), v(Vec(0)), F(1), C(0), Jp(1), c(c), fluid(fluid) {
  }
};

std::vector<Particle> particles;

// Vector3: [velocity_x, velocity_y, mass]
Vector3 grid[n + 1][n + 1];

void advance(real dt) {
  // Reset grid
  std::memset(grid, 0, sizeof(grid));

  real max_entry_in_F(0);
  // P2G
  for (auto &p : particles) {
    // element-wise floor
    Vector2i base_coord = (p.x * inv_dx - Vec(0.5f)).cast<int>();

    Vec fx = p.x * inv_dx - base_coord.cast<real>();

    // Quadratic kernels [http://mpm.graphics Eqn. 123, with x=fx, fx-1,fx-2]
    Vec w[3] = {Vec(0.5) * sqr(Vec(1.5) - fx), Vec(0.75) - sqr(fx - Vec(1.0)),
                Vec(0.5) * sqr(fx - Vec(0.5))};

    // Compute current Lamé parameters [http://mpm.graphics Eqn. 86]
    auto e = 1;  // std::exp(hardening * (1.0f - p.Jp));
    auto mu = mu_0 * e;
    auto lambda = lambda_0 * e;

    // Current volume
    real J = p.Jp;

    // Polar decomposition for fixed corotated model
    Mat r, s;
    polar_decomp(p.F, r, s);
    p.F = lerp(p.fluid, p.F, r);

    max_entry_in_F = std::max(max_entry_in_F, p.F.frobenius_norm());

    // [http://mpm.graphics Paragraph after Eqn. 176]
    real Dinv = 4 * inv_dx * inv_dx;
    // [http://mpm.graphics Eqn. 52]
    auto PF = (2 * mu * (p.F - r) * transposed(p.F) + lambda * (J - 1) * J);

    // Cauchy stress times dt and inv_dx
    auto stress = -(dt * vol) * (Dinv * PF);
    // auto stress = -(100 * E * dt * vol) * Mat(p.Jp - 1);
    // auto stress = -(50 * E * dt * vol) * Mat(determinant(p.F) - 1);

    // Fused APIC momentum + MLS-MPM stress contribution
    // See http://taichi.graphics/wp-content/uploads/2019/03/mls-mpm-cpic.pdf
    // Eqn 29
    auto affine = stress + particle_mass * p.C;

    // P2G
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        auto dpos = (Vec(i, j) - fx) * dx;
        // Translational momentum
        Vector3 mass_x_velocity(p.v * particle_mass, particle_mass);
        grid[base_coord.x + i][base_coord.y + j] +=
            (w[i].x * w[j].y * (mass_x_velocity + Vector3(affine * dpos, 0)));
      }
    }
  }
  max_norm = max_entry_in_F;

  TC_P(max_entry_in_F);

  // For all grid nodes
  for (int i = 0; i <= n; i++) {
    for (int j = 0; j <= n; j++) {
      auto &g = grid[i][j];
      // No need for epsilon here
      if (g[2] > 0) {
        // Normalize by mass
        g /= g[2];
        // Gravity
        g += dt * Vector3(0, -200, 0);

        // boundary thickness
        real boundary = 0.05;
        // Node coordinates
        real x = (real)i / n;
        real y = real(j) / n;

        // Sticky boundary
        if (x < boundary || x > 1 - boundary || y > 1 - boundary) {
          g = Vector3(0);
        }
        // Separate boundary
        if (y < boundary) {
          // g[1] = std::max(0.0f, g[1]);
          g[1] = 0.0f;
        }
      }
    }
  }

  // G2P
  for (auto &p : particles) {
    // element-wise floor
    Vector2i base_coord = (p.x * inv_dx - Vec(0.5f)).cast<int>();
    Vec fx = p.x * inv_dx - base_coord.cast<real>();
    Vec w[3] = {Vec(0.5) * sqr(Vec(1.5) - fx), Vec(0.75) - sqr(fx - Vec(1.0)),
                Vec(0.5) * sqr(fx - Vec(0.5))};

    p.C = Mat(0);
    p.v = Vec(0);

    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        auto dpos = (Vec(i, j) - fx);
        auto grid_v = Vec(grid[base_coord.x + i][base_coord.y + j]);
        auto weight = w[i].x * w[j].y;
        // Velocity
        p.v += weight * grid_v;
        // APIC C
        p.C += 4 * inv_dx * Mat::outer_product(weight * grid_v, dpos);
      }
    }

    // Advection
    p.x += dt * p.v;

    // MLS-MPM F-update
    auto F = (Mat(1) + dt * p.C) * p.F;

    /*
    Mat svd_u, sig, svd_v;
    svd(F, svd_u, sig, svd_v);

    // Snow Plasticity
    for (int i = 0; i < 2 * int(plastic); i++) {
      sig[i][i] = clamp(sig[i][i], 1.0f - 2.5e-2f, 1.0f + 7.5e-3f);
    }

    real oldJ = determinant(F);
    F = svd_u * sig * transposed(svd_v);

     */
    p.Jp *= determinant(F) / determinant(p.F);
    p.F = F;
  }
}

// Seed particles with position and color
void add_object(Vec center, int c) {
  // Randomly sample 1000 particles in the square
  for (int i = 0; i < q * q * 1000; i++) {
    auto r = Vec::rand();
    auto f = clamp(r.x * 1.0_f - 0.0_f, 0_f, 1_f);
    f = f * f;
    particles.push_back(
        Particle((r * Vec(4, 1) - Vec(1)) * 0.1f + center, c, f));
  }
}

int main() {
  GUI gui("Real-time 2D MLS-MPM", window_size, window_size);
  auto &canvas = gui.get_canvas();

  real dy = 0.45;
  add_object(Vec(0.4, dy), 0xED553B);
  add_object(Vec(0.4, dy + 0.1), 0xF2B134);
  add_object(Vec(0.4, dy + 0.2), 0x068587);

  int frame = 0;

  // Main Loop
  for (int step = 0;; step++) {
    // Advance simulation
    advance(dt);

    // Visualize frame
    if (step % int(frame_dt / dt) == 0) {
      // Clear background
      canvas.clear(0x112F41);
      // Box
      canvas.rect(Vec(0.04), Vec(0.96)).radius(1).color(0x4FB99F).close();
      canvas.text(fmt::format("max_p ||F_p||_2 = {}", max_norm),
                  Vector2(0.05, 0.9), 20, Vector4(1));
      // Particles
      for (auto p : particles) {
        canvas.circle(p.x).radius(1.5).color(
            p.c);  // 1.0f - p.fluid, 0.2f, 0.7 * p.fluid);
      }
      // Update image
      gui.update();

      // Write to disk (optional)
      // canvas.img.write_as_image(fmt::format("tmp/{:05d}.png", frame++));
    }
  }
}

/* -----------------------------------------------------------------------------
** Reference: A Moving Least Squares Material Point Method with Displacement
              Discontinuity and Two-Way Rigid Body Coupling (SIGGRAPH 2018)

  By Yuanming Hu (who also wrote this 88-line version), Yu Fang, Ziheng Ge,
           Ziyin Qu, Yixin Zhu, Andre Pradhana, Chenfanfu Jiang


** Build Instructions:

Step 1: Download and unzip mls-mpm88.zip (Link: http://bit.ly/mls-mpm88)
        Now you should have "mls-mpm88.cpp" and "taichi.h".

Step 2: Compile and run

* Linux:
    g++ mls-mpm88-explained.cpp -std=c++14 -g -lX11 -lpthread -O3 -o mls-mpm
    ./mls-mpm


* Windows (MinGW):
    g++ mls-mpm88-explained.cpp -std=c++14 -lgdi32 -lpthread -O3 -o mls-mpm
    .\mls-mpm.exe


* Windows (Visual Studio 2017+):
  - Create an "Empty Project"
  - Use taichi.h as the only header, and mls-mpm88-explained.cpp as the only
source
  - Change configuration to "Release" and "x64"
  - Press F5 to compile and run


* OS X:
    g++ mls-mpm88-explained.cpp -std=c++14 -framework Cocoa -lpthread -O3 -o
mls-mpm
    ./mls-mpm


** FAQ:
Q1: What does "1e-4_f" mean?
A1: The same as 1e-4f, a float precision real number.

Q2: What is "real"?
A2: real = float in this file.

Q3: What are the hex numbers like 0xED553B?
A3: They are RGB color values.
    The color scheme is borrowed from
    https://color.adobe.com/Copy-of-Copy-of-Core-color-theme-11449181/

Q4: How can I get higher-quality?
A4: Change n to 320; Change dt to 1e-5; Change E to 2e4;
    Change particle per cube from 500 to 8000 (Ln 72).
    After the change the whole animation takes ~3 minutes on my computer.

Q5: How to record the animation?
A5: Uncomment Ln 2 and 85 and create a folder named "tmp".
    The frames will be saved to "tmp/XXXXX.png".

    To get a video, you can use ffmpeg. If you already have taichi installed,
    you can simply go to the "tmp" folder and execute

      ti video 60

    where 60 stands for 60 FPS. A file named "video.mp4" is what you want.

Q6: How is taichi.h generated?
A6: Please check out my #include <taichi> talk:
    http://taichi.graphics/wp-content/uploads/2018/11/include_taichi.pdf
    and the generation script:
    https://github.com/yuanming-hu/taichi/blob/master/misc/amalgamate.py
    You can regenerate it using `ti amal`, if you have taichi installed.

Questions go to yuanming _at_ mit.edu
                            or https://github.com/yuanming-hu/taichi_mpm/issues.

                                                      Last Update: March 6, 2019
                                                      Version 1.5

----------------------------------------------------------------------------- */
