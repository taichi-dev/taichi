#include <taichi/common/util.h>
#include "euler_fluid.h"

TC_NAMESPACE_BEGIN

real volume_control_i = 0.0f;
real volume_control_p = 0.0f;

EulerFluid::EulerFluid() {
}

void EulerFluid::set_levelset(const LevelSet2D & boundary_levelset)
{
    this->boundary_levelset = boundary_levelset;
}

void EulerFluid::initialize(const Config &config) {
    initialize_solver(config);
    if (!config.get("initialize_particles", false)) {
        printf("initialzie_particles=false, No particles initialized.\n");
    }
    else {
        initialize_particles(config);
    }
}

void EulerFluid::initialize_solver(const Config &config)
{
    title = config.get("title", "Simulation");
    width = config.get("simulation_width", 64);
    height = config.get("simulation_height", 64);
    supersampling = config.get("supersampling", true);
    show_grid = config.get("show_grid", true);
    kernel_size = config.get("kernel_size", 1);
    cfl = config.get("cfl", 0.1f);
    u = Array(width + 1, height, 0.0f, Vector2(0.0f, 0.5f));
    u_weight = Array(width + 1, height, 0.0f);
    v = Array(width, height + 1, 0.0f, Vector2(0.5f, 0.0f));
    v_weight = Array(width, height + 1, 0.0f);
    cell_types = Array2D<CellType>(width, height, CellType::AIR);
    gravity = config.get_vec2("gravity");
    use_bridson_pcg = config.get("use_bridson_pcg", true);
    maximum_iterations = config.get("maximum_iterations", 300);
    tolerance = config.get("tolerance", 1e-4f);
    initialize_pressure_solver();
    liquid_levelset.initialize(width, height, Vector2(0.5f, 0.5f));
    t = 0;
}

void EulerFluid::initialize_particles(const Config &config)
{
    float position_noise_amp = config.get("position_noise", 0.0f);
    auto initializer = get_initializer(config.get("initializer", "collapse"));
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            cell_types[i][j] = initializer(float(i + 0.5f) / width,
                float(j + 0.5f) / height);
            if (cell_types[i][j] == CellType::WATER) {
                if (supersampling)
                    for (int k = 0; k < 4; k++) {
                        particles.push_back(Particle(Vector2(i, j) + supersample_positions[k]));
                    }
                else
                    particles.push_back(Particle(Vector2(i + 0.5f, j + 0.5f)));
            }
        }
    }
    for (auto &p : particles) {
        p.position += position_noise_amp * position_noise();
    }
}

Vector2 EulerFluid::sample_velocity(Vector2 position, const Array &u, const Array &v) {
    if (kernel_size == 1) {
        return Vector2(u.sample(position), v.sample(position));
    }
    else {
        float inv_kernel_size = 1.0f / kernel_size;
        int extent = (kernel_size + 1) / 2;
        int x, y;
        float tot_weight, tot;

        x = (int)floor(position.x);
        y = (int)floor(position.y - 0.5);
        tot_weight = 0.0f; tot = 0.0f;
        for (int dx = -extent + 1; dx <= extent; dx++) {
            for (int dy = -extent + 1; dy <= extent; dy++) {
                int nx = x + dx, ny = y + dy;
                if (!u.inside(nx, ny))
                    continue;
                float weight = kernel(inv_kernel_size * (position - Vector2(nx, ny + 0.5f)));
                tot += u[nx][ny] * weight;
                tot_weight += weight;
            }
        }
        float vx = tot / tot_weight;

        x = (int)floor(position.x - 0.5);
        y = (int)floor(position.y);
        tot_weight = 0.0f; tot = 0.0f;
        for (int dx = -extent + 1; dx <= extent; dx++) {
            for (int dy = -extent + 1; dy <= extent; dy++) {
                int nx = x + dx, ny = y + dy;
                if (!v.inside(nx, ny)) {
                    continue;
                }
                float weight = kernel(inv_kernel_size * (position - Vector2(nx + 0.5f, ny)));
                tot += v[nx][ny] * weight;
                tot_weight += weight;
            }
        }

        float vy = tot / tot_weight;
        return Vector2(vx, vy);
    }
}

Vector2 EulerFluid::sample_velocity(Vector2 position) {
    return sample_velocity(position, u, v);
}

std::function<EulerFluid::CellType(float, float)> EulerFluid::get_initializer(std::string name)
{
    if (name == "collapse") {
        return [](float i, float j) -> EulerFluid::CellType {
            if (((0.2 < i && i < 0.4 && 0.1 < j && j < 0.5)) || j <= 0.1) {
                return EulerFluid::CellType::WATER;
            }
            else {
                return EulerFluid::CellType::AIR;
            }
        };
    }
    else if (name == "single") {
        return [](float i, float j) -> EulerFluid::CellType {
            return i == 0.5f && j == 0.5f ? EulerFluid::CellType::WATER : EulerFluid::CellType::AIR;
        };
    }
    else if (name == "drop") {
        return [](float i, float j) -> EulerFluid::CellType {
            return (0.4f <= i && i <= 0.6f && 0.4f <= j && j <= 0.6f) ? EulerFluid::CellType::WATER : EulerFluid::CellType::AIR;
        };
    }
    else if (name == "still") {
        return [](float i, float j) -> EulerFluid::CellType {
            return j < 0.5f ? EulerFluid::CellType::WATER : EulerFluid::CellType::AIR;
        };
    }
    else if (name == "full") {
        return [](float i, float j) -> EulerFluid::CellType {
            return EulerFluid::CellType::WATER;
        };
    }
    else {
        error("Unknown Intiializer Name: " + name);
    }
    return [](float i, float j) -> EulerFluid::CellType {
        if (((0.2 < i && i < 0.4 && 0.1 < j && j < 0.5)) || j <= 0.1) {
            return EulerFluid::CellType::WATER;
        }
        else {
            return EulerFluid::CellType::AIR;
        }
    };
}

bool EulerFluid::check_u_activity(int i, int j) {
    if (i < 1 || j < 0 || i >= width || j >= height) return false;
    return cell_types[i - 1][j] == CellType::WATER ||
        cell_types[i][j] == CellType::WATER;
}

bool EulerFluid::check_v_activity(int i, int j) {
    if (i < 0 || j < 1 || i >= width || j >= height) return false;
    return cell_types[i][j - 1] == CellType::WATER ||
        cell_types[i][j] == CellType::WATER;
}

void EulerFluid::level_set_extrapolate() {
    //for (int i = 0; i < width + 1; i++) {
    //    for (int j = 0; j < height; j++) {
    //        Vector2 p = Vector2(i, j + 0.5);
    //        if (level_set->sample(p) > 0) {
    //            Vector2 q = level_set->sample_closest(p);
    //            u[i][j] = sample_velocity(q).x;
    //        }
    //    }
    //}
    //for (int i = 0; i < width; i++) {
    //    for (int j = 0; j < height + 1; j++) {
    //        Vector2 p = Vector2(i + 0.5f, j);
    //        if (level_set->sample(p) > 0) {
    //            Vector2 q = level_set->sample_closest(p);
    //            v[i][j] = sample_velocity(q).y;
    //        }
    //    }
    //}

}

void EulerFluid::simple_extrapolate() {
    const int dx[4]{ 1, -1, 0, 0 };
    const int dy[4]{ 0, 0, 1, -1 };
    for (int i = 1; i < width; i++) {
        for (int j = 0; j < height; j++) {
            if (check_u_activity(i, j)) continue;
            float sum = 0.0f, num = 0.0f;
            for (int k = 0; k < 4; k++) {
                int nx = i + dx[k], ny = j + dy[k];
                if (check_u_activity(nx, ny)) {
                    num += 1.0f;
                    sum += u[nx][ny];
                }
            }
            if (num == 0.0f)
                u[i][j] = 0.0f;
            else
                u[i][j] = sum / num;
        }
    }
    for (int i = 0; i < width; i++) {
        for (int j = 1; j < height; j++) {
            if (check_v_activity(i, j)) continue;
            float sum = 0.0f, num = 0.0f;
            for (int k = 0; k < 4; k++) {
                int nx = i + dx[k], ny = j + dy[k];
                if (check_v_activity(nx, ny)) {
                    num += 1.0f;
                    sum += v[nx][ny];
                }
            }
            if (num == 0.0f)
                v[i][j] = 0.0f;
            else {
                v[i][j] = sum / num;
            }
        }
    }
}

void EulerFluid::step(float delta_t)
{
    float simulation_time = 0.0f;
    while (simulation_time < delta_t - eps) {
        float purpose_dt = get_dt_with_cfl_1() * cfl;
        float thres = 0.001f;
        if (purpose_dt < delta_t * thres) {
            purpose_dt = delta_t * thres;
            printf("substep dt too small, clamp.\n");
            Particle fastest;
            float avg = 0;
            for (auto &p : particles) {
                if (length(p.velocity) > length(fastest.velocity)) {
                    fastest = p;
                }
                avg += abs(p.velocity.x) + abs(p.velocity.y);
            }
            printf("Fastest particle:\n");
            P(fastest.position);
            P(fastest.velocity);
            avg /= particles.size() * 2;
            P(avg);
        }
        float dt = std::min(delta_t - simulation_time, purpose_dt);
        substep(dt);
        simulation_time += dt;
    }
    compute_liquid_levelset();
}

void EulerFluid::compute_liquid_levelset()
{
    liquid_levelset.reset(1e7f); // Do not use INF here, otherwise interpolation will get NAN...
    for (auto &p : particles) {
        for (auto &ind : liquid_levelset.get_rasterization_region(p.position, 3)) {
            Vector2 delta_pos = ind.get_pos() - p.position;
            liquid_levelset[ind] = std::min(liquid_levelset[ind], length(delta_pos) - p.radius);
        }
    }
    for (auto &ind : liquid_levelset.get_region()) {
        if (liquid_levelset[ind] < 0.5f) {
            if (boundary_levelset.sample(ind.get_pos()) < 0)
                liquid_levelset[ind] = -0.5f;
        }
    }
}

Array EulerFluid::advect(const Array & arr, float delta_t)
{
    Array arr_out(arr.get_width(), arr.get_height(), 0, arr.get_storage_offset());
    for (auto &ind : arr.get_region()) {
        Vector2 position = ind.get_pos();
        Vector2 velocity = sample_velocity(position);
        velocity = sample_velocity(position - delta_t * 0.5f * velocity);
        arr_out[ind] = arr.sample(position - delta_t * velocity);
    }
    return arr_out;
}

bool EulerFluid::check_diag_domination()
{
    for (auto &ind : Ad.get_region()) {
        float res = Ad[ind];
        res -= abs(Ax[ind]);
        res -= abs(Ay[ind]);
        if (ind.i > 0) {
            res -= abs(Ax[ind + Vector2(-1, 0)]);
        }
        if (ind.j > 0) {
            res -= abs(Ay[ind + Vector2(0, -1)]);
        }
        if (res < -1e-7f) {
            return false;
        }
    }
    return true;
}


void EulerFluid::advect(float delta_t) {
    float total_energy = 0;
    for (auto &particle : particles) {
        Vector2 velocity = sample_velocity(particle.position);
        if (true) {
            Particle mid = particle;
            mid.move(delta_t * 0.5f * velocity);
            mid.position = clamp_particle_position(mid.position);
            velocity = sample_velocity(mid.position);
            particle.move(delta_t * velocity);
            particle.position = clamp_particle_position(particle.position);
        }
        else {
            particle.move(delta_t * velocity);
            particle.position = clamp_particle_position(particle.position);
        }
        total_energy += dot(velocity, velocity) * 0.5f;
        //total_energy -= glm::dot(particle.position, gravity);
    }
    Array new_u = advect(u, delta_t), new_v = advect(v, delta_t);
    u = new_u;
    v = new_v;
}

void EulerFluid::apply_external_forces(float delta_t) {
    for (int i = 1; i < width; i++) {
        for (int j = 0; j < height; j++) {
            u[i][j] += gravity.x * delta_t;
        }
    }
    for (int i = 0; i < width; i++) {
        for (int j = 1; j < height; j++) {
            v[i][j] += gravity.y * delta_t;
        }
    }
}

Vector2 EulerFluid::position_noise()
{
    return Vector2(rand() - 0.5f, rand() - 0.5f);
}

bool EulerFluid::inside(int x, int y) {
    return 0 <= x && x < width && 0 <= y && y < height;
}

void EulerFluid::prepare_for_pressure_solve() {
    for (auto &ind : u.get_region()) {
        u_weight[ind] = LevelSet2D::fraction_outside(boundary_levelset[ind], boundary_levelset[ind.neighbour(Vector2i(0, 1))]);
    }
    for (auto &ind : v.get_region()) {
        v_weight[ind] = LevelSet2D::fraction_outside(boundary_levelset[ind], boundary_levelset[ind.neighbour(Vector2i(1, 0))]);
    }
    Ax = 0;
    Ay = 0;
    Ad = 0;
    E = 0;
    const float theta_threshold = 0.01f;
    for (auto &ind : cell_types.get_region()) {
        int i = ind.i, j = ind.j;

        float phi = liquid_levelset.sample(ind);
        if (phi >= 0) continue;
        float lhs = 0;
        float neighbour_phi;
        float vel_weight;

        neighbour_phi = liquid_levelset.sample(ind.get_pos() - Vector2(1, 0));
        if (neighbour_phi < 0) {
            vel_weight = u_weight[ind];
            lhs += vel_weight;
        }
        else {
            float theta = max(theta_threshold, LevelSet2D::fraction_inside(phi, neighbour_phi));
            lhs += 1.0f / theta;
        }

        neighbour_phi = liquid_levelset.sample(ind.get_pos() + Vector2(1, 0));
        if (neighbour_phi < 0) {
            vel_weight = u_weight[ind.neighbour(Vector2i(1, 0))];
            lhs += vel_weight;
            Ax[i][j] -= vel_weight;
        }
        else {
            float theta = max(theta_threshold, LevelSet2D::fraction_inside(phi, neighbour_phi));
            lhs += 1.0f / theta;
        }

        neighbour_phi = liquid_levelset.sample(ind.get_pos() - Vector2(0, 1));
        if (neighbour_phi < 0) {
            vel_weight = v_weight[ind];
            lhs += vel_weight;
        }
        else {
            float theta = max(theta_threshold, LevelSet2D::fraction_inside(phi, neighbour_phi));
            lhs += 1.0f / theta;
        }


        neighbour_phi = liquid_levelset.sample(ind.get_pos() + Vector2(0, 1));
        if (neighbour_phi < 0) {
            vel_weight = v_weight[ind.neighbour(Vector2i(0, 1))];
            lhs += vel_weight;
            Ay[i][j] -= vel_weight;
        }
        else {
            float theta = max(theta_threshold, LevelSet2D::fraction_inside(phi, neighbour_phi));
            lhs += 1.0f / theta;
        }

        Ad[ind] = lhs;
    }

    if (!check_diag_domination()) {
        printf("Warning: Non diagonally dominant matrix found!\n");
    }

    float tao = 0.97f, sigma = 0.25f;

    for (auto &ind : cell_types.get_region()) {
        if (Ad[ind] > 0) {
            float e = Ad[ind];
            float e_tao = 0.0f;
            Index2D nei;
            nei = ind.neighbour(Vector2i(-1, 0));
            if (cell_types.inside(nei) && Ad[nei] > 0) {
                e -= sqr(Ax[nei] * E[nei]);
                e_tao -= Ax[nei] * Ay[nei] * sqr(E[nei]);
            }
            nei = ind.neighbour(Vector2i(0, -1));
            if (cell_types.inside(nei) && Ad[nei] > 0) {
                e -= sqr(Ay[nei] * E[nei]);
                e_tao -= Ay[nei] * Ax[nei] * sqr(E[nei]);
            }
            e += e_tao * tao;
            if (e < sigma * Ad[ind]) e = Ad[ind];
            assert_info(e >= 0, "Negative e!");
            E[ind] = 1.0f / sqrtf(e);
            if (!is_normal(E[ind])) {
                P(E[ind]);
                P(e);
                printf("Bad E.\n");
            }
        }
    }
}

Array EulerFluid::apply_A(const Array &x) {
    Array y(width, height);
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            if (Ad[i][j] > 0) {
                float t = 0;
                if (0 < i)
                    t += Ax[i - 1][j] * x[i - 1][j];
                if (i < width - 1)
                    t += Ax[i][j] * x[i + 1][j];
                if (0 < j)
                    t += Ay[i][j - 1] * x[i][j - 1];
                if (j < height - 1)
                    t += Ay[i][j] * x[i][j + 1];
                t += Ad[i][j] * x[i][j];
                y[i][j] = t;
            }
            else {
                y[i][j] = 0;
            }
        }
    }
    return y;
}


Array EulerFluid::solve_pressure_naive() {
    static int total_count = 0;
    int count = 0;
    Array r = get_rhs(), z, s;
    z = apply_preconditioner(r);
    s = z;
#define CH(v) if (!v.is_normal()) printf("Abnormal value doring CG: %s [Ln %d]\n", #v, __LINE__);
    pressure = 0;
    double sigma = z.dot_double(r);
    double zs;

    for (count = 0; count < maximum_iterations; count++){
        z = apply_A(s);
        zs = z.dot_double(s);
        double alpha = sigma / max(1e-6, zs);
        pressure = pressure.add((float)alpha, s);
        r = r.add(-(float)alpha, z);
        if (r.abs_max() < tolerance) break;
        z = apply_preconditioner(r);
        double sigma_new = z.dot_double(r);
        double beta = sigma_new / sigma;
        s = z.add((float)beta, s);
        sigma = sigma_new;
    }
    total_count += count;
    printf("t = %f, iterated %d times, avg = %f\n", t, count, total_count / t);
    return pressure;
}

void EulerFluid::project(float delta_t) {
    update_volume_controller();
    // P(count_water_cells());

    apply_boundary_condition();
    if (use_bridson_pcg) {
        assert_info(false, "Not implemented");
    }
    else {
        prepare_for_pressure_solve();
        p = solve_pressure_naive();
    }
    apply_boundary_condition();

    if (!(p.is_normal())) {
        printf("Abnormal pressure!!!!!\n");
    }
    apply_pressure(p);
}

void EulerFluid::mark_cells() {
    cell_types = CellType::AIR;
    for (auto &particle : particles) {
        int x = (int)particle.position.x, y = (int)particle.position.y;
        cell_types[x][y] = CellType::WATER;
    }
}

void EulerFluid::substep(float delta_t) {
    mark_cells();
    apply_external_forces(delta_t);
    compute_liquid_levelset();
    project(delta_t);
    simple_extrapolate();
    advect(delta_t);
    t += delta_t;
}

void EulerFluid::apply_pressure(const Array &p) {
    for (int i = 0; i < width - 1; i++) {
        for (int j = 0; j < height; j++) {
            u[i + 1][j] += p[i][j] - p[i + 1][j];
        }
    }
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height - 1; j++) {
            v[i][j + 1] += p[i][j] - p[i][j + 1];
        }
    }
}

void EulerFluid::apply_boundary_condition() {
    for (auto &ind : u.get_region()) {
        if (boundary_levelset.sample(ind.get_pos()) <= 0.0f) {
            u[ind] = 0.0f;
        }
    }
    for (auto &ind : v.get_region()) {
        if (boundary_levelset.sample(ind.get_pos()) <= 0.0f) {
            v[ind] = 0.0f;
        }
    }
}

void EulerFluid::show(ImageBuffer<Vector3> &buffer) {
    if (show_grid)
        for (int i = 0; i < width + 1; i++) {
            for (int j = 0; j < height + 1; j++) {
                buffer.set_pixel((float(i) / width), (float(j) / height), Vector3(0, 0, 0.9));
                buffer.set_pixel((float(i + 0.5f) / width), (float(j) / height), Vector3(0.3, 0, 0));
                buffer.set_pixel((float(i) / width), (float(j + 0.5f) / height), Vector3(0, 0.3, 0));
            }
        }
    for (auto &particle : particles) {
        float x = particle.position.x / width;
        float y = particle.position.y / height;
        buffer.set_pixel(x, y, Vector3(1));
    }
}

float EulerFluid::get_current_time()
{
    return t;
}

void EulerFluid::add_particle(Fluid::Particle & particle)
{
    particles.push_back(particle);
}

std::vector<Fluid::Particle> EulerFluid::get_particles()
{
    return particles;
}

LevelSet2D EulerFluid::get_liquid_levelset()
{
    return liquid_levelset;
}


Array EulerFluid::apply_preconditioner(const Array &r) {
    q = 0;
    z = 0;
    assert_info(E.is_normal(), "Abnormal E!\n");
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            if (Ad[i][j] > 0) {
                float t = r[i][j];
                if (i > 0)
                    t -= Ax[i - 1][j] * E[i - 1][j] * q[i - 1][j];
                if (j > 0)
                    t -= Ay[i][j - 1] * E[i][j - 1] * q[i][j - 1];
                q[i][j] = t * E[i][j];
            }
        }
    }
    for (int i = width - 1; i >= 0; i--) {
        for (int j = height - 1; j >= 0; j--) {
            if (Ad[i][j] > 0) {
                float t = q[i][j];
                if (i < width - 1) {
                    t -= Ax[i][j] * E[i][j] * z[i + 1][j];
                }
                if (j < height - 1) {
                    t -= Ay[i][j] * E[i][j] * z[i][j + 1];
                }
                z[i][j] = t * E[i][j];
            }
        }
    }
    return z;
}

Array EulerFluid::get_rhs() {
    Array r(width, height, 0);
    float correction = get_volume_correction();
    for (auto &ind : cell_types.get_region()) {
        if (Ad[ind] > 0) {
            float rhs =
                - u[ind.neighbour(Vector2i(1, 0))] * u_weight[ind.neighbour(Vector2i(1, 0))]
                + u[ind] * u_weight[ind]
                - v[ind.neighbour(Vector2i(0, 1))] * v_weight[ind.neighbour(Vector2i(0, 1))]
                + v[ind] * v_weight[ind];
            r[ind] = rhs + correction;
        }
    }
    return r;
}

void EulerFluid::apply_viscosity(float delta_t) {
    //static float tmp[2048][2048];
    //for (int i = 1; i < width; i++) {
    //    for (int j = 1; j < height - 1; j++) {
    //        float sum = u[i][j] * 4;
    //        sum -= u[i - 1][j];
    //        sum -= u[i + 1][j];
    //        sum -= u[i][j - 1];
    //        sum -= u[i][j + 1];
    //        tmp[i][j] = sum;
    //    }
    //}
    //for (int i = 1; i < width; i++) {
    //    for (int j = 1; j < height - 1; j++) {
    //        u[i][j] -= viscosity * delta_t * tmp[i][j];
    //    }
    //}
    //for (int i = 1; i < width - 1; i++) {
    //    for (int j = 1; j < height; j++) {
    //        float sum = v[i][j] * 4;
    //        sum -= v[i - 1][j];
    //        sum -= v[i + 1][j];
    //        sum -= v[i][j - 1];
    //        sum -= v[i][j + 1];
    //        tmp[i][j] = sum;
    //    }
    //}
    //for (int i = 1; i < width - 1; i++) {
    //    for (int j = 1; j < height; j++) {
    //        v[i][j] -= viscosity * delta_t * tmp[i][j];
    //    }
    //}

}

int EulerFluid::count_water_cells() {
    int ret = 0;
    for (auto &cell : cell_types)
        ret += cell == CellType::WATER;
    return ret;
}

void EulerFluid::initialize_volume_controller() {
    integrate_water_cells_difference = 0.0f;
    target_water_cells = (float)count_water_cells();
    last_water_cells = (float)count_water_cells();
}

void EulerFluid::update_volume_controller() {
    int current_water_cells = count_water_cells();
    integrate_water_cells_difference +=
        current_water_cells - target_water_cells;
    last_water_cells = (float)current_water_cells;
    float factor = 1.0f / width / height;
    volume_correction_factor =
        -factor * (integrate_water_cells_difference * volume_control_i +
            (current_water_cells - target_water_cells) *
            volume_control_p);
}

float EulerFluid::get_volume_correction() {
    //if (enable_volume_control)
    if (false)
        return volume_correction_factor;
    else
        return 0.0f;
}

void EulerFluid::advect_level_set(float delta_t) {
    //float tmp[width][height];
    //for (int i = 0; i < width; i++) {
    //    for (int j = 0; j < height; j++) {
    //        Vector2 position = sl_position(level_set->get_location(i, j), delta_t);
    //        tmp[i][j] = level_set->sample(position);
    //    }
    //}
    //memcpy(level_set->signed_distance, tmp, sizeof(tmp));
}

Vector2 EulerFluid::sl_position(Vector2 position, float delta_t) {
    Vector2 velocity = sample_velocity(position, u, v);
    Vector2 mid = clamp(position - velocity * 0.5f * delta_t);
    velocity = sample_velocity(mid, u, v);
    position = clamp(position - velocity * delta_t);
    return position;
}

void EulerFluid::print_u() {
    printf("u:\n");
    for (int j = height - 1; j >= 0; j--) {
        for (int i = 0; i <= width; i++) {
            printf("%.4f ", u[i][j]);
        }
        printf("\n");
    }
}

void EulerFluid::print_v() {
    printf("v:\n");
    for (int j = height; j >= 0; j--) {
        for (int i = 0; i < width; i++) {
            printf("%+6.4f ", v[i][j]);
        }
        printf("\n");
    }

}

void EulerFluid::show_surface() {
}

void EulerFluid::initialize_pressure_solver() {
    pressure = Array(width, height, 0.0);
    Ad = Array(width, height);
    Ax = Array(width, height);
    Ay = Array(width, height);
    E = Array(width, height);
    p = Array(width, height, 0.0f);
    q = Array(width, height);
    z = Array(width, height);
    water_cell_index = Array2D<int>(width, height);
}

Vector2 EulerFluid::clamp_particle_position(Vector2 pos) {
    pos = Vector2(clamp(pos.x, 0.0f, (float)width), clamp(pos.y, 0.0f, (float)height));
    float phi = boundary_levelset.sample(pos);
    if (phi < 0) {
        pos -= boundary_levelset.get_normalized_gradient(pos) * phi;
    }
    return pos;
}

float EulerFluid::get_dt_with_cfl_1()
{
    return 1 / max(get_max_grid_speed(), 1e-5f);
}

float EulerFluid::get_max_grid_speed()
{
    float maximum_speed = 0;
    for (auto &vel : u)
        maximum_speed = max(abs(vel), maximum_speed);
    for (auto &vel : v)
        maximum_speed = max(abs(vel), maximum_speed);
    return maximum_speed;
}

Array EulerFluid::get_density()
{
    return density;
}

void EulerFluid::add_source(const Config & config)
{
    sources.push_back(config);
}

Array EulerFluid::get_pressure()
{
    return pressure;
}


TC_NAMESPACE_END
