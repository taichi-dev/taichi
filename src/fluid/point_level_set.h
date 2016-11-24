#pragma once
#include "level_set.h"

TC_NAMESPACE_BEGIN

/*
class PointLevelSet: public LevelSet {
private:
    vector<vec3> points;
    vec3 closest_surface_point[N][N];

public:
    void load(vector<vec3> points);

    virtual vec2 sample_closest(vec2 pos);

    void load_normalized(vector<vec3> points);

    void initialize_surface_points();

    bool update_signed_distance(int x, int y);

    bool update_using(int x, int y, vec3 p);

    void post_processing();

};

*/

TC_NAMESPACE_END
