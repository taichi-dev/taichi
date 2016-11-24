#pragma once
#include <vector>
#include "system/opengl.h"

TC_NAMESPACE_BEGIN

class PointRenderer {
    GLuint program, vao, vbo;
    int max_size;
	vec2 lower_left, upper_right;
public:
    PointRenderer(int max_size = 1048576);
    void render(vector<vec2> points, float point_size=1.0f);
	void setViewport(vec2 lower_left, vec2 upper_right);
};



TC_NAMESPACE_END

