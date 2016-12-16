#include <taichi/common/util.h>
#include "level_set.h"
#include <cmath>

TC_NAMESPACE_BEGIN

/*
void LevelSet::load_bool(function<bool(float, float)> f) {
    auto t = [&](float i, float j) -> float {
        return 2.0f * (!f(i, j)) - 1.0f;
    };
    load(t);
}

void LevelSet::calculate() {
    memset(updated, 0, sizeof(updated));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            signed_distance[i][j] = 1e10;
        }
    }
    initialize_surface_points();
    fast_sweep();
    post_processing();
}

void LevelSet::render(TextureRenderer &texture_renderer) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float gray = tanhf(signed_distance[i][j] * 20 / N) / 2 + 0.5f;
            // float gray = signed_distance[i][j] * 0.3 + 0.5;
            texture_renderer.set_pixel(i, j, vec4(gray, gray, gray, 0));
        }
    }
    texture_renderer.render();
}

LevelSet::LevelSet() {

}

void LevelSet::get_cloest_points(int x, int y) {
    float &s = signed_distance[x][y];
    int dx[] = {0, 0, 1, -1};
    int dy[] = {1, -1, 0, 0};
    for (int k = 0; k < 4; k++) {
        int nx = x + dx[k], ny = y + dy[k];
        if (!inside(nx, ny)) continue;
        vec2 a = get_location(x, y), b = get_location(nx, ny);
        if (sgn(indicator(a.x, a.y)) == sgn(indicator(b.x, b.y))) continue;
        float q = find_surface(a, b);
        float d = q;
        if (d < s) {
            s = d;
            closest_surface_point[x][y] = (1.0f - q) * a + q * b;
            updated[x][y] = true;
        }
    }
}

bool LevelSet::update_signed_distance(int x, int y) {
    int dx[] = {0, 0, 1, -1};
    int dy[] = {1, -1, 0, 0};
    bool modified = false;
    vec2 v = get_location(x, y);
    for (int k = 0; k < 4; k++) {
        int nx = x + dx[k], ny = y + dy[k];
        if (!inside(nx, ny) || updated[nx][ny] == false) continue;
        vec2 csp = closest_surface_point[nx][ny];
        float d = glm::length(csp - v);
        if (d < signed_distance[x][y]) {
            signed_distance[x][y] = d;
            closest_surface_point[x][y] = csp;
            updated[x][y] = true;
            modified = true;
        }
    }
    return modified;
}

float LevelSet::find_surface(vec2 a, vec2 b) {
    float sa = indicator(a.x, a.y);
    float sb = indicator(b.x, b.y);
    return abs(sa / (sa - sb));
}

vec2 LevelSet::get_location(int x, int y) {
    return vec2(x + 0.5f, y + 0.5f);
}

bool LevelSet::inside(int x, int y) {
    return 0 <= x && x < N && 0 <= y && y < N;
}

float LevelSet::sample(vec2 pos) {
    pos -= vec2(0.5f, 0.5f);
    pos.x = clamp(pos.x, 0.0f, N - 1.0f);
    pos.y = clamp(pos.y, 0.0f, N - 1.0f);
    int x = (int) floor(pos.x);
    int y = (int) floor(pos.y);
    float a = pos.x - x;
    float b = pos.y - y;
    if (false && 1 <= x && x < N - 2 && 1 <= y && y < N - 2) {
        float tx[4];
        for (int i = 0; i < 4; i++) {
            tx[i] = catmull_rom(&signed_distance[x + i - 1][y - 1], b);
        }
        return catmull_rom(tx, a);
    }
    else
        return (1 - a) *
               ((1 - b) * signed_distance[x][y] +
                b * signed_distance[x][y + 1]) +
               a * ((1 - b) * signed_distance[x + 1][y] +
                    b * signed_distance[x + 1][y + 1]);
}

vec2 LevelSet::sample_closest(vec2 pos) {
    pos -= vec2(0.5f, 0.5f);
    pos.x = clamp(pos.x, 0.0f, N - 1.0f);
    pos.y = clamp(pos.y, 0.0f, N - 1.0f);
    int x = (int) floor(pos.x);
    int y = (int) floor(pos.y);
    float a = pos.x - x;
    float b = pos.y - y;
    return (1 - a) * ((1 - b) * closest_surface_point[x][y] +
                      b * closest_surface_point[x][y + 1]) +
           a * ((1 - b) * closest_surface_point[x + 1][y] +
                b * closest_surface_point[x + 1][y + 1]);
}

void LevelSet::adjust_from(LevelSet &l) {
    auto f = [&](float i, float j) {
        return l.sample((float) N * vec2(i, j));
    };
    load(f);
    calculate();
}

LevelSet::LevelSet(LevelSet &l) {
    adjust_from(l);
}

void LevelSet::print() {
    printf("Level set:\n");
    for (int j = N - 1; j >= 0; j--) {
        for (int i = 0; i < N; i++) {
            printf("%+f ", signed_distance[i][j]);
        }
        printf("\n");
    }

}


void LevelSet::load(function<float(float, float)> f) {
    raw_indicator = f;
    calculate();
}

float LevelSet::indicator(vec2 position) {
    return raw_indicator(position.x / N, position.y / N);
}

float LevelSet::indicator(float x, float y) {
    return raw_indicator(x / N, y / N);
}

void LevelSet::fast_sweep() {
    bool modified = true;
    int count = 0;
    while (modified) {
        modified = false;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                if (update_signed_distance(i, j)) {
                    modified = true;
                }
            }
        }
        for (int i = N - 1; i >= 0; i--) {
            for (int j = 0; j < N; j++) {
                if (update_signed_distance(i, j)) {
                    modified = true;
                }
            }
        }
        for (int i = 0; i < N; i++) {
            for (int j = N - 1; j >= 0; j--) {
                if (update_signed_distance(i, j)) {
                    modified = true;
                }
            }
        }
        for (int i = N - 1; i >= 0; i--) {
            for (int j = N - 1; j >= 0; j--) {
                if (update_signed_distance(i, j)) {
                    modified = true;
                }
            }
        }
        count++;
    }
    assert(count < 8);

}

void LevelSet::initialize_surface_points() {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            signed_distance[i][j] = 1e10;
            get_cloest_points(i, j);
        }
    }
}

void LevelSet::post_processing() {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            vec2 v = get_location(i, j);
            if (indicator(v.x, v.y) < 0) {
                signed_distance[i][j] *= -1;
            }
        }
    }
}
*/

TC_NAMESPACE_END
