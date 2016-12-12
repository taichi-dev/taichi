#include "pressure_solver.h"
#include "system/threading.h"

TC_NAMESPACE_BEGIN

TC_INTERFACE_DEF(PressureSolver3D, "pressure_solver_3d");

const static Vector3i offsets[]{
		Vector3i(1, 0, 0), Vector3i(-1, 0, 0),
		Vector3i(0, 1, 0), Vector3i(0, -1, 0),
		Vector3i(0, 0, 1), Vector3i(0, 0, -1)
};

// Maybe we are going to need Algebraic Multigrid in the future,
// but let's have a GMG with different boundary conditions support first...
// TODO: AMG, cache

// TODO: Add one layer of Neumann padding
class MultigridPressureSolver : public PressureSolver3D {
public:
	int width, height, depth, max_level;
	std::vector<Array> pressures, residuals, tmp_residuals;
	typedef char CellType;
	typedef Array3D<CellType> BCArray;
	std::vector<BCArray> boundaries;
	const int size_threshould = 64;
	int num_threads;
	const char DIRICHLET = 1;
	const char NEUMANN = 2;

	struct SystemRow {
		float inv_numerator;
		int neighbours;
		SystemRow() : SystemRow(0) {}
		SystemRow(int i) {
			inv_numerator = 1.0f;
			neighbours = 0;
		}
		CellType get_neighbour_cell_type(int k) const {
			return (neighbours >> (k * 2)) & 3;
		}
		void set_neighbour_cell_type(int k, CellType c) {
			neighbours = (neighbours & (~(3 << (2 * k)) | (c << (2 * k))));
		}

		static void test() {

		}
	};

	std::vector<Array3D<SystemRow>> systems;
		
	void set_boundary_condition(const BCArray &boundary) {
		int width = this->width;
		int height = this->height;
		int depth = this->depth;
		boundaries.clear();
		boundaries.push_back(Array3D<char>(width, height, depth));
		for (int k = 0; k < max_level - 1; k++) {
			width /= 2;
			height /= 2;
			depth /= 2;
			bool has_dirichlet = false;
			bool all_neumann = true;
			boundaries.push_back(Array3D<char>(width, height, depth));
			for (auto &ind : boundaries.back().get_region()) {
				for (int i = 0; i < 2; i++) {
					for (int j = 0; j < 2; j++) {
						for (int k = 0; k < 2; k++) {
							char bc = boundaries.back()[ind.i * 2 + i][ind.j * 2 + j][ind.k * 2 + k];
							if (bc == DIRICHLET) {
								has_dirichlet = true;
								break;
							}
							if (bc != NEUMANN) {
								all_neumann = false;
							}
						}
					}
				}
				char bc = has_dirichlet ? DIRICHLET : (all_neumann ? NEUMANN : 0);
				boundaries.back()[ind] = bc;
			}
		}
	}

	void initialize(const Config &config) {
		this->width = config.get_int("width");
		this->height = config.get_int("height");
		this->depth = config.get_int("depth");
		this->num_threads = config.get_int("num_threads");
		this->max_level = 0;
		int width = this->width;
		int height = this->height;
		int depth = this->depth;
		do {
			pressures.push_back(Array(width, height, depth));
			residuals.push_back(Array(width, height, depth));
			tmp_residuals.push_back(Array(width, height, depth));
			assert_info(width % 2 == 0, "odd width");
			assert_info(height % 2 == 0, "odd height");
			assert_info(depth % 2 == 0, "odd depth");
			width /= 2;
			height /= 2;
			depth /= 2;
			max_level++;
		} while (width * height * depth * 8 >= size_threshould);
	}

	void parallel_for_each_cell(Array &arr, int threshold, const std::function<void(const Index3D &index)> &func) {
		int max_side = std::max(std::max(arr.get_width(), arr.get_height()), arr.get_depth());
		int num_threads;
		if (max_side >= 32) {
			num_threads = this->num_threads;
		}
		else {
			num_threads = 1;
		}
		ThreadedTaskManager::run(arr.get_width(), num_threads, [&](int x) {
			const int height = arr.get_height();
			const int depth = arr.get_depth();
			for (int y = 0; y < height; y++) {
				for (int z = 0; z < depth; z++) {
					func(Index3D(x, y, z));
				}
			}
		});
	}

	void gauss_seidel(const Array &residual, Array &pressure, int rounds) {
		for (int i = 0; i < rounds; i++) {
			for (int c = 0; c < 2; c++) {
				parallel_for_each_cell(pressure, 128, [&](const Index3D &ind) {
					int sum = ind.i + ind.j + ind.k;
					if ((sum) % 2 == c) {
						float res = residual[ind];
						float numerator = 6.0f;
						for (auto &offset : offsets) {
							if (pressure.inside(ind + offset)) {
								res += pressure[ind + offset];
							}
						}
						pressure[ind] = res / numerator;
					}
				});
			}
		}
	}

	void apply_L(const Array &pressure, Array &output) {
		for (auto &ind : pressure.get_region()) {
			float pressure_center = pressure[ind];
			float res = 0.0f;
			for (auto &offset : offsets) {
				float p = 0.0f;
				if (pressure.inside(ind + offset)) {
					p = pressure[ind + offset];
				}
				res += pressure_center - p;
			}
			output[ind] = res;
		}
	}

	void compute_residual(const Array &pressure, const Array &div, Array &residual) {
		parallel_for_each_cell(residual, 128, [&](const Index3D &ind) {
			float pressure_center = pressure[ind];
			float res = 0.0f;
			for (auto &offset : offsets) {
				float p = 0.0f;
				if (pressure.inside(ind + offset)) {
					p = pressure[ind + offset];
				}
				res += pressure_center - p;
			}
			residual[ind] = div[ind] - res;
		});
	}

	void downsample(const Array &x, Array &x_downsampled) { // Restriction
		for (auto &ind : x_downsampled.get_region()) {
			x_downsampled[ind] =
				x[ind.i * 2 + 0][ind.j * 2 + 0][ind.k * 2 + 0] +
				x[ind.i * 2 + 0][ind.j * 2 + 0][ind.k * 2 + 1] +
				x[ind.i * 2 + 0][ind.j * 2 + 1][ind.k * 2 + 0] +
				x[ind.i * 2 + 0][ind.j * 2 + 1][ind.k * 2 + 1] +
				x[ind.i * 2 + 1][ind.j * 2 + 0][ind.k * 2 + 0] +
				x[ind.i * 2 + 1][ind.j * 2 + 0][ind.k * 2 + 1] +
				x[ind.i * 2 + 1][ind.j * 2 + 1][ind.k * 2 + 0] +
				x[ind.i * 2 + 1][ind.j * 2 + 1][ind.k * 2 + 1];
		}
	}

	void prolongate(Array &x, const Array &x_delta) const {
		for (auto &ind : x.get_region()) {
			x[ind] += x_delta[ind.i / 2][ind.j / 2][ind.k / 2] * 0.5f;
		}
	}

	void run(int level) {
		if (level != 0) {
			pressures[level].reset(0.0f);
		}
		if (residuals[level].get_size() <= size_threshould) { // 4 * 4 * 4
			gauss_seidel(residuals[level], pressures[level], 100);
		}
		else {
			gauss_seidel(residuals[level], pressures[level], 4);
			{
				compute_residual(pressures[level], residuals[level], tmp_residuals[level]);
				downsample(tmp_residuals[level], residuals[level + 1]);
				run(level + 1);
				prolongate(pressures[level], pressures[level + 1]);
			}
			gauss_seidel(residuals[level], pressures[level], 4);
		}
	}

	virtual void run(const Array &residual, Array &pressure, float pressure_tolerance) {
		pressures[0] = pressure;
		residuals[0] = residual;
		int iterations = 0;
		do {
			iterations++;
			run(0);
			compute_residual(pressures[0], residuals[0], tmp_residuals[0]);
			P(iterations);
			P(tmp_residuals[0].abs_max());
		} while (tmp_residuals[0].abs_max() > pressure_tolerance);
		pressure = pressures[0];
	}
};

class MultigridPCGPressureSolver : public MultigridPressureSolver {
public:
	void initialize(const Config &config) {
		MultigridPressureSolver::initialize(config);
	}
	Array apply_preconditioner(Array &r) {
		pressures[0] = 0;
		residuals[0] = r;
		MultigridPressureSolver::run(0);
		return pressures[0];
	}
	virtual void run(const Array &residual, Array &pressure, float pressure_tolerance) {
		pressure = 0;
		Array r(width, height, depth), mu(width, height, depth);
		mu = r.get_average();
		r = residual; //TODO: r = r - Lx
		double nu = (r - mu).abs_max();
		if (nu < pressure_tolerance)
			return;
		r -= mu;
		Array p = apply_preconditioner(r);
		double rho = p.dot_double(r);
		int maximum_iterations = 20;
		Array z(width, height, depth);
		for (int count = 0; count <= maximum_iterations; count++) {
			apply_L(p, z);
			double sigma = p.dot_double(z);
			double alpha = rho / max(1e-20, sigma);
			r.add_in_place(-(float)alpha, z); mu = 0;//r.get_average();
			nu = (r - mu).abs_max();
			(r - mu).print_abs_max_pos();
			printf(" MGPCG iteration #%02d, nu=%f\n", count, nu);
			if (nu < pressure_tolerance || count == maximum_iterations) {
				pressure.add_in_place((float)alpha, p);
				return;
			}
			r -= mu;
			z = apply_preconditioner(r);
			double rho_new = z.dot_double(r);
			double beta = rho_new / rho;
			rho = rho_new;
			pressure.add_in_place((float)alpha, p);
			p = z.add((float)beta, p);
		}
	}
};

TC_IMPLEMENTATION(PressureSolver3D, MultigridPressureSolver, "mg");
TC_IMPLEMENTATION(PressureSolver3D, MultigridPCGPressureSolver, "mgpcg");

TC_NAMESPACE_END
