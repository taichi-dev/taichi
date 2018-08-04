The Moving Least Squares Material Point Method
================================================
`Taichi` aims to accelerate computer graphics research and development.
`taichi.h` pushes this goal to another extreme: it contains the most essential
parts of `taichi` in a single, amalgamated C++ header, and can be used as a compact
"code base" for demonstrative computer graphics algorithms.

As a result, these applications are extremely portable and self-contained. For example, the Moving Least Squares Material Point Method (MLS-MPM) can be implemented
in only 88 lines of code:

.. code-block:: cpp
    :linenos:

    // The Moving Least Squares Material Point Method in 88 LoC (with comments)
    //   To compile:   g++ mpm.cpp -std=c++14 -g -lX11 -lpthread -O3 -o mpm
    #include "taichi.h" // Single header version of (a small part of) taichi
    using namespace taichi;
    const int n = 64 /*grid resolution (cells)*/, window_size = 500;
    const real dt = 1e-4_f, frame_dt = 1e-3_f, dx = 1.0_f / n, inv_dx = 1.0_f / dx;
    real mass = 1.0_f, vol = 1.0_f; // Particle mass and volume
    real hardening = 10, E = 1e4 /* Young's Modulus*/, nu = 0.2 /*Poisson's Ratio*/;
    real mu_0 = E/(2*(1+nu)), lambda_0=E*nu/((1+nu)*(1-2*nu));    // Lame parameters
    using Vec = Vector2; using Mat = Matrix2; //Handy abbriviations for lin. algebra
    struct Particle {Vec x/*position*/, v/*velocity*/; Mat B/*affine momentum*/;
      Mat F/*elastic deformation grad.*/;   real Jp /*det(plastic def. grad.)*/;
      Particle(Vec x, Vec v=Vec(0)) : x(x), v(v), B(0), F(1), Jp(1) {} };
    std::vector<Particle> particles; // Particle states
    Vector3 grid[n + 1][n + 1];// velocity with mass, note that node_res=cell_res+1

    void advance(real dt) {                // Simulation
      std::memset(grid, 0, sizeof(grid));  // Reset grid
      for (auto &p : particles) {          // P2G
        Vector2i base_coord = (p.x*inv_dx-Vec(0.5_f)).cast<int>();
        Vec fx = p.x * inv_dx - base_coord.cast<real>();
        // Quadratic kernels, see http://mpm.graphics Formula (123)
        Vec w[3]{Vec(0.5) * sqr(Vec(1.5) - fx), Vec(0.75) - sqr(fx - Vec(1.0)),
                 Vec(0.5) * sqr(fx - Vec(0.5))};
        auto e = std::exp(hardening * (1.0_f - p.Jp)), mu=mu_0*e, lambda=lambda_0*e;
        real J = determinant(p.F);         //Current volume
        Mat r, s; polar_decomp(p.F, r, s); //Polor decomp. for Fixed Corotated Model
        auto force =                   // Negative Cauchy stress times dt and inv_dx
            inv_dx*dt*vol*(2*mu * (p.F-r) * transposed(p.F) + lambda * (J-1) * J);
        for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) { // Scatter to grid
          auto dpos = fx - Vec(i, j);
          Vector3 contrib(p.v * mass, mass);
          grid[base_coord.x + i][base_coord.y + j] +=
              w[i].x*w[j].y*(contrib+Vector3(4.0_f*(force+p.B*mass)*dpos));
        }
      }
      for(int i = 0; i <= n; i++) for(int j = 0; j <= n; j++) { //For all grid nodes
        auto &g = grid[i][j];
        if (g[2] > 0) {                                  // No need for epsilon here
          g /= g[2];                                     // Normalize by mass
          g += dt * Vector3(0, -100, 0);                 // Apply gravity
          real boundary=0.05,x=(real)i/n,y=real(j)/n;//boundary thickness,node coord
          if (x < boundary||x > 1-boundary||y > 1-boundary) g=Vector3(0);//Sticky BC
          if (y < boundary) g[1]=std::max(0.0_f, g[1]);              //"Separate" BC
        }    // "BC" stands for "boundary condition", which is applied to grid nodes
      }
      for (auto &p : particles) { // Grid to particle
        Vector2i base_coord = (p.x * inv_dx - Vec(0.5_f)).cast<int>();
        Vec fx = p.x * inv_dx - base_coord.cast<real>();
        Vec w[3]{Vec(0.5) * sqr(Vec(1.5) - fx), Vec(0.75) - sqr(fx - Vec(1.0)),
                 Vec(0.5) * sqr(fx - Vec(0.5))};
        p.B = Mat(0); p.v = Vec(0);
        for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) {
          auto dpos = fx - Vec(i, j),
               grid_v = Vec(grid[base_coord.x + i][base_coord.y + j]);
          auto weight = w[i].x * w[j].y;
          p.v += weight * grid_v;                                       //  Velocity
          p.B += Mat::outer_product(weight * grid_v, dpos);             //    APIC B
        }
        p.x += dt * p.v;                                                // Advection
        auto F = (Mat(1) - (4 * inv_dx * dt) * p.B) * p.F;       // MLS-MPM F-update
        Mat svd_u, sig, svd_v; svd(F, svd_u, sig, svd_v); // SVD for snow Plasticity
        for (int i = 0; i < 2; i++)    // See SIGGRAPH 2013: MPM for Snow Simulation
          sig[i][i] = clamp(sig[i][i], 1.0_f - 2.5e-2_f, 1.0_f + 7.5e-3_f);
        real oldJ = determinant(F); F = svd_u * sig * transposed(svd_v);
        real Jp_new = clamp(p.Jp * oldJ / determinant(F), 0.6_f, 20.0_f);
        p.Jp = Jp_new; p.F = F;
      }
    }

    void add_object(Vec center) {    // Seed particles
      for (int i = 0; i < 1000; i++) // Randomly sample 1000 particles in the square
        particles.push_back(Particle((Vec::rand()*2.0_f-Vec(1))*0.08_f+center));  }

    int main() {
      GUI gui("Taichi Demo: Real-time MLS-MPM 2D ", window_size, window_size);
      add_object(Vec(0.5,0.4));add_object(Vec(0.45,0.6));add_object(Vec(0.55,0.8));
      for (int i = 0;; i++) {                              // Main Loop
        advance(dt);                                       // Advance simulation
        if (i % int(frame_dt / dt) == 0) {                 // Redraw frame
          gui.canvas->clear(Vector4(0.2, 0.4, 0.7, 1.0_f)); // Clear background
          for (auto p : particles)                         // Draw particles
            gui.buffer[(p.x * (inv_dx*window_size/n)).cast<int>()] = Vector4(0.8);
          gui.update();                                    // Update GUI
        }//Reference: A Moving Least Squares Material Point Method with Displacement
      } //             Discontinuity and Two-Way Rigid Body Coupling (SIGGRAPH 2018)
    }  //  By Yuanming Hu (who also wrote this 88-line version), Yu Fang, Ziheng Ge,
      //                        Ziyin Qu, Yixin Zhu, Andre Pradhana, Chenfanfu Jiang
