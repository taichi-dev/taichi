#include <taichi/common/util.h>
#include <taichi/math.h>
#include <taichi/system/timer.h>
#include <taichi/common/dict.h>
#include <taichi/system/profiler.h>

TC_NAMESPACE_BEGIN

static_assert(sizeof(real) == 8, "Please compile with double precision");

template <int dim>
constexpr int ke_size() {
  return pow<dim>(2) * dim;
}

class Material {
 public:
  float64 E;       // Young's modulus
  float64 nu;      // Poisson's ratio
  float64 lambda;  // 1st Lame's param
  float64 mu;      // 2nd Lame's param

  Material(float64 E, float64 nu) : E(E), nu(nu) {
    lambda = E * nu / ((1 + nu) * (1 - 2 * nu));
    mu = E / (2 * (1 + nu));
  }

  void set_lame(float64 lambda, float64 mu) {
    this->mu = mu;
    this->lambda = lambda;
    this->E = mu * (3 * lambda + 2 * mu) / (lambda + mu);
    this->nu = 0.5_f * lambda / (lambda + mu);
  }

  Material() {
    E = nu = mu = lambda = 0.0_f;
  }
};

template <int dim>
std::vector<float64> get_Ke(Material material);

template <int dim>
class HexFEMSolver {
 public:
  using Vector = VectorND<dim, real>;
  using Vectori = VectorND<dim, int>;
  template <typename T>
  using Array = ArrayND<dim, T>;
  using Region = RegionND<dim>;
  using Index = IndexND<dim>;

  struct BoundaryConditionNode {
    Vectori node;
    int axis;
    real val;
  };

  using BoundaryCondition = std::vector<BoundaryConditionNode>;

  BoundaryCondition boundary_condition;

  real cg_tolerance;
  real penalty;
  bool print_residuals;
  Material material;

  Vectori res;
  std::vector<float64> ke;
  int cg_min_iterations;
  int cg_max_iterations;

  void initialize(const Config &config) {
    res = config.get<Vectori>("res");
    cg_tolerance = config.get<real>("cg_tolerance", 1e-4);
    cg_min_iterations = config.get<int>("cg_min_iterations", 0);
    cg_max_iterations = config.get<int>("cg_max_iterations", 100000);
    assert_info(cg_min_iterations <= cg_max_iterations,
                "cg_min_iteration should <= cg_max_iterations");
    penalty = config.get<real>("penalty");
    print_residuals = config.get<bool>("print_residuals", false);

    material = *config.get_ptr<Material>("material");

    ke = get_Ke<dim>(material);

    assert_info(ke.size() == pow<2>(ke_size<dim>()), "Incorrect Ke size");
    for (int i = 0; i < ke_size<dim>(); i++) {
      for (int j = 0; j < ke_size<dim>(); j++) {
        assert_info(Ke(i, j) == Ke(j, i), "Asymmetric!");
      }
    }
  }

  // returns: Kx
  virtual Array<Vector> apply_K(const Array<real> &density,
                                Array<Vector> x) const {
    TC_NOT_IMPLEMENTED;
    return x;
  }

  TC_FORCE_INLINE float64 Ke(int row, int column) const {
    return ke[row * ke_size<dim>() + column];
  }

  constexpr int get_index(Index ind, int d) const {
    int ret = 0;
    for (int i = 0; i < dim; i++) {
      ret += (1 << i) * (ind[dim - 1 - i]);
    }
    return ret * dim + d;
  }

  constexpr int get_index(int i0, int i1, int d) const {
    int ret = 0;
    ret += (1 << 0) * i1;
    ret += (1 << 1) * i0;
    return ret * dim + d;
  }

  virtual void set_boundary_condition(
      const BoundaryCondition &boundary_condition) {
    this->boundary_condition = boundary_condition;
  }
};

using HexFEMSolver2D = HexFEMSolver<2>;

template <>
inline std::vector<float64> get_Ke<2>(Material material) {
  static constexpr int indices[8][8] = {
      {0, 1, 2, 3, 4, 5, 6, 7}, {1, 0, 7, 6, 5, 4, 3, 2},
      {2, 7, 0, 5, 6, 3, 4, 1}, {3, 6, 5, 0, 7, 2, 1, 4},
      {4, 5, 6, 7, 0, 1, 2, 3}, {5, 4, 3, 2, 1, 0, 7, 6},
      {6, 3, 4, 1, 2, 7, 0, 5}, {7, 2, 1, 4, 3, 6, 5, 0}};

  float64 E = material.E;
  float64 nu = material.nu;

  constexpr int dim = 2;

  std::vector<float64> ke_entries = {1.0f / 2.0f - nu / 6.0f,
                                     1 / 8.0f + nu / 8.0f,
                                     -1 / 4.0f - nu / 12.0f,
                                     -1 / 8.0f + 3 * nu / 8.0f,
                                     -1 / 4.0f + nu / 12.0f,
                                     -1 / 8.0f - nu / 8.0f,
                                     nu / 6.0f,
                                     1 / 8.0f - 3.0f * nu / 8.0f};

  for (int i = 0; i < ke_size<2>(); i++) {
    ke_entries[i] *= E / (1.0f - pow<2>(nu));
  }

  int map[] = {0, 3, 1, 2};

  auto M = [&](int i, int p) { return map[i] * dim + p; };

  std::vector<float64> Ke;
  for (int i = 0; i < pow<dim>(2); i++) {
    for (int p = 0; p < dim; p++) {
      for (int j = 0; j < pow<dim>(2); j++) {
        for (int q = 0; q < dim; q++) {
          Ke.push_back(ke_entries[indices[M(i, p)][M(j, q)]]);
        }
      }
    }
  }
  return Ke;
}

template <typename T>
float64 dot_product(const T &a, const T &b) {
  assert_info(a.get_size() == b.get_size(),
              "Arrays for dot product must have same shapes.");
  float64 sum(0);
  int size = a.get_size();
#pragma omp parallel for reduction(+ : sum)
  for (int i = 0; i < size; i++) {
    sum += a.data[i].dot(b.data[i]);
  }
  return sum;
}

template <typename T>
real p_abs_max(const T &a) {
  real max_val(0);
  int size = a.get_size();
#pragma omp parallel for reduction(max : max_val)
  for (int i = 0; i < size; i++) {
    max_val = std::max(max_val, a.data[i].abs_max());
  }
  return max_val;
}

// a = a + alpha * b
template <typename S, typename T>
void p_add_in_place(T &a, const S alpha, const T &b) {
  assert_info(a.get_size() == b.get_size(),
              "Arrays for add_in_place must have same shapes.");
  int size = a.get_size();
#pragma omp parallel for
  for (int i = 0; i < size; i++) {
    a.data[i] = a.data[i] + alpha * b.data[i];
  }
}

// b = a + alpha * b
template <typename S, typename T>
void p_add_in_place2(const T &a, const S alpha, T &b) {
  assert_info(a.get_size() == b.get_size(),
              "Arrays for add_in_place must have same shapes.");
  int size = a.get_size();
#pragma omp parallel for
  for (int i = 0; i < size; i++) {
    b.data[i] = a.data[i] + alpha * b.data[i];
  }
}
template <int dim>
class CPUCGHexFEMSolver : public HexFEMSolver<dim> {
 public:
  using Base = HexFEMSolver<dim>;
  using Base::Ke;
  using Base::boundary_condition;
  using Base::cg_max_iterations;
  using Base::cg_min_iterations;
  using Base::cg_tolerance;
  using Base::get_index;
  using Base::penalty;
  using Base::print_residuals;
  using Base::res;
  using typename Base::Index;
  using typename Base::Region;
  using typename Base::Vector;
  using typename Base::Vectori;

  bool use_preconditioner;

  int cg_restart;

  template <typename T>
  using Array = ArrayND<dim, T>;

  void initialize(Vectori res, int penalty) {
    Config config;
    config.set("res", res);
    config.set("penalty", penalty);
    auto material = Material(1, 0.3);
    config.set("material", &material);
    Base::initialize(config);
    cg_restart = config.get<int>("cg_restart", 0);
  }

  void enforce_boundary_condition(Array<Vector> &u) const {
    for (auto &bc : boundary_condition) {
      u[bc.node][bc.axis] = bc.val;
    }
  }

  void project(Array<Vector> &u) const {
    for (auto &bc : boundary_condition) {
      u[bc.node][bc.axis] = 0;
    }
  }

  void apply_K_impl(const Array<real> &density,
                    Array<Vector> &x,
                    Array<Vector> &Kx,
                    bool profile = false) const {
    Kx.reset(Vector(0));
    assert(Kx.get_res() == x.get_res());
    int penalty = int(this->penalty);
    for (auto ind : density.get_region()) {
      real d = density[ind];
      real scale = d;
      for (int i = 1; i < penalty; i++) {
        scale *= d;
      }
      for (int offsetKxi = 0; offsetKxi < dim; offsetKxi++) {
        for (int offsetKxj = 0; offsetKxj < dim; offsetKxj++) {
          for (int offsetxi = 0; offsetxi < dim; offsetxi++) {
            for (int offsetxj = 0; offsetxj < dim; offsetxj++) {
              real tmp[2] = {0, 0};
              const Vector x_in = x[ind + Vector2i(offsetxi, offsetxj)];
              for (int i = 0; i < dim; i++) {
                for (int j = 0; j < dim; j++) {
                  int row = get_index(offsetKxi, offsetKxj, i);
                  int column = get_index(offsetxi, offsetxj, j);
                  const real coeff = Ke(row, column);
                  tmp[i] += coeff * x_in[j];
                }
              }
              for (int i = 0; i < dim; i++)
                Kx[ind + Vector2i(offsetKxi, offsetKxj)][i] += tmp[i] * scale;
            }
          }
        }
      }
    }
  }

  // returns: project(Kx)
  Array<Vector> apply_K(const Array<real> &density,
                        Array<Vector> x) const override {
    enforce_boundary_condition(x);
    Array<Vector> Kx = x.same_shape(Vector(0.0_f));
    apply_K_impl(density, x, Kx);
    project(Kx);
    return Kx;
  }

  // returns: Kx
  Array<Vector> apply_K_no_projection(const Array<real> &density,
                                      Array<Vector> x) const {
    enforce_boundary_condition(x);
    Array<Vector> Kx = x.same_shape(Vector(0.0_f));
    apply_K_impl(density, x, Kx);
    return Kx;
  }

  Array<Vector> solve(const Array<real> &density,
                      const Array<Vector> &f_,
                      const Array<Vector> &initial_guess,
                      Array<real> &dc,
                      real &objective_out) {
    Array<Vector> x = initial_guess;
    Array<Vector> r;
    Array<Vector> p;

    Array<Vector> x0 = initial_guess.same_shape(Vector(0.0_f));
    enforce_boundary_condition(x0);
    TC_P(x0.abs_max());

    auto f = f_ - apply_K_no_projection(density, x0);
    TC_P(f.abs_max());

    real tolerance = cg_tolerance * f.abs_max();
    real restart_ratio = 0;

    int num_iterations;

    real rTr;
    real alpha, beta;
    Array<Vector> Kp = x.same_shape(Vector(0.0_f));
    Array<Vector> Kx = x.same_shape(Vector(0.0_f));

    float64 t_apply_time = 0;
    float64 t_start = Time::get_time();
    for (int k = 0; k < cg_max_iterations; k++) {
      num_iterations = k + 1;
      if (k == 0 || (cg_restart > 0 && k % cg_restart == 0)) {
        real before_restart = 0;
        if (k) {
          before_restart = r.abs_max();
        }
        apply_K_impl(density, x, Kx);
        r = f - Kx;
        project(r);
        p = r;
        if (k) {
          restart_ratio =
              std::max(restart_ratio, p_abs_max(r) / before_restart);
        }
      }

      Profiler _("cg_iteration");
      // auto t_cg = Time::get_time();
      apply_K_impl(density, p, Kp, true);
      // t_apply_time += Time::get_time() - t_cg;
      project(Kp);
      {
        Profiler __("dp1");
        rTr = dot_product(r, r);
        alpha = rTr / (dot_product(p, Kp) + 1e-100_f);
      }
      {
        Profiler __("vec_add1");
        p_add_in_place(x, alpha, p);
        p_add_in_place(r, -alpha, Kp);
      }
      if (print_residuals) {
        TC_P(r.abs_max());
      } else {
        if (k % 1000 == 0) {
          printf("iter %d, %f\n", k, r.abs_max());
        }
      }
      if (k >= cg_min_iterations && p_abs_max(r) < tolerance) {
        printf("CG converged in %d iterations\n", k);
        break;
      }
      TC_PROFILE("dp2", beta = dot_product(r, r) / (rTr + 1e-100_f));
      TC_PROFILE("vec_add2", p_add_in_place2(r, beta, p));
    }

    if (p_abs_max(r) >= tolerance) {
      printf("CG did not converge.\n");
    }
    if (restart_ratio > 1.1) {
      TC_P(restart_ratio);
    }
    apply_K_impl(density, x, Kx);
    enforce_boundary_condition(Kx);
    TC_P((f - Kx).abs_max());
    std::cout << fmt::format(
                     "Time per cg iteration: {:.4f} ms",
                     1000 * (Time::get_time() - t_start) / num_iterations)
              << std::endl;

    real objective = 0;
    for (auto &ind : density.get_region()) {
      Region offset_region(Vectori(0), Vectori(2));
      real dc_tmp = 0.0f;
      for (auto &offset_Kx : offset_region) {
        for (auto &offset_x : offset_region) {
          for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
              int row = get_index(offset_Kx, i);
              int column = get_index(offset_x, j);
              real coeff = Ke(row, column);
              dc_tmp += x[ind + offset_Kx.get_ipos()][i] * coeff *
                        x[ind + offset_x.get_ipos()][j];
            }
          }
        }
      }
      dc[ind] = std::max(
          0.0_f, dc_tmp * std::pow(density[ind], penalty - 1) * penalty);
      objective += dc_tmp * std::pow(density[ind], penalty);
    }

    objective_out = objective;
    return x;
  }
};

TC_NAMESPACE_END
