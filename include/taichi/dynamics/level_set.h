#pragma once

#include <taichi/math/linalg.h>

TC_NAMESPACE_BEGIN

/*
class LevelSet {
public:
    function<float(float, float)> raw_indicator;
    float signed_distance[N][N];


    vec2 closest_surface_point[N][N];

private:
    void get_cloest_points(int x, int y);

    virtual bool update_signed_distance(int x, int y);

    float find_surface(vec2 a, vec2 b);

    float indicator(vec2 position);

    float indicator(float x, float y);


protected:
    bool updated[N][N];

    bool inside(int x, int y);

    virtual void initialize_surface_points();

    virtual void fast_sweep();

public:

    LevelSet();

    LevelSet(LevelSet &l);

    void load_bool(function<bool(float, float)> f);

    void load(function<float(float, float)> f);

    void calculate();

    void render(TextureRenderer &texture_renderer);

    float sample(vec2 pos);

    virtual vec2 sample_closest(vec2 pos);

    void adjust_from(LevelSet &l);

    void print();

    vec2 get_location(int x, int y);

    virtual void post_processing();

    virtual void load(vector<vec3> points) {assert(false);};

    virtual void load_normalized(vector<vec3> points) {assert(false);};
};


*/

TC_NAMESPACE_END
