#include "point_level_set.h"

TC_NAMESPACE_BEGIN

/*
void PointLevelSet::load(vector<vec3> points) {
    this->points = points;
    calculate();
}

void PointLevelSet::initialize_surface_points() {
    for (auto p : points) {
        float r = p.z;
        int sx = (int)ceil(max(0.0f, p.x - r - 1.0f)), sy = (int)ceil(max(0.0f, p.y - r - 1.0f));
        int ex = (int)min(N - 1.0f, p.x + r), ey = (int)max(N - 1.0f, p.y + r);
        for (int i = sx; i <= ex; i++) {
            for (int j = sy; j <= ey; j++) {
                update_using(i, j, p);
            }
        }
    }
}

bool PointLevelSet::update_signed_distance(int x, int y) {
    int dx[] = {0, 0, 1, -1};
    int dy[] = {1, -1, 0, 0};
    bool modified = false;
    for (int k = 0; k < 4; k++) {
        int nx = x + dx[k], ny = y + dy[k];
        if (!inside(nx, ny) || !updated[nx][ny]) continue;
        if (update_using(x, y, closest_surface_point[nx][ny])) {
            modified = true;
        }
    }
    return modified;

}

bool PointLevelSet::update_using(int x, int y, vec3 p) {
    vec2 pos = get_location(x, y);
    float distance = glm::length(pos - vec2(p.x, p.y));
    float &sd = signed_distance[x][y];
    if (!((distance >= 0 && sd <= 0) || abs(distance) >= abs(sd))) {
        sd = distance;
        closest_surface_point[x][y] = p;
        updated[x][y] = true;
        return true;
    } else {
        return false;
    }
}

void PointLevelSet::load_normalized(vector<vec3> points) {
    for (auto &p : points) {
        p *= N;
    }
    load(points);
}

vec2 PointLevelSet::sample_closest(vec2 pos) {
    vec3 c = closest_surface_point[int(pos.x - 0.5f)][int(pos.y - 0.5f)];
    vec2 d = pos - vec2(c.x, c.y);
    float l = max(1e-3f, glm::length(d));
    vec2 n = d / l;
    return vec2(c.x, c.y) + n * 0.6f;
}

void PointLevelSet::post_processing() {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            signed_distance[i][j] -= 0.5f;
        }
    }
}
*/

TC_NAMESPACE_END

