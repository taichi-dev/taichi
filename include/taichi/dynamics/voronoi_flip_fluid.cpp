#include <taichi/point_cloud/point_cloud.h>

#include "voronoi_flip_fluid.h"

TC_NAMESPACE_BEGIN

class NearesetNeighbour2D {

};

void VoronoiFLIP::rasterize()
{
    error("Not implemented");
    /*
    NearestNeighbour2D voronoi[2];
    vector<float> v_input[2];
    for (int k = 0; k < 2; k++) {
        v_input[k].reserve(particles.size() * 3);
        for (int i = 0; i < (int)particles.size(); i++) {
            v_input[k].push_back(particles[i].position.x);
            v_input[k].push_back(particles[i].position.y);
            v_input[k].push_back(particles[i].velocity[k]);
        }
        voronoi[k].insert_data_points(v_input[k], (int)particles.size());
    }
    for (int i = 0; i < width + 1; i++) {
        for (int j = 0; j < height; j++) {
            if (check_u_activity_loose(i, j))
                u[i][j] = voronoi[0].query((float)i, j + 0.5f, 1.0f, 4);
        }
    }
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height + 1; j++) {
            if (check_v_activity_loose(i, j)) {
                v[i][j] = voronoi[1].query(i + 0.5f, (float)j, 1.0f, 4);
            }
        }
    }
    */
}

TC_NAMESPACE_END

