#include <taichi/geometry/factory.h>

TC_NAMESPACE_BEGIN

std::vector<Triangle> Mesh3D::generate(const Vector2i res,
    const Function23 *surf, const Function23 *norm, const Function22 *uv,
    bool smooth_normal) {
    const Vector2 dp = Vector2(1.0) / Vector2(res);

    assert_info(surf != nullptr, "Surface function can not be null");

    auto get_normal_at = [&](const Vector2 &p) -> Vector3 {
        if (norm) {
            return normalized((*norm)(p));
        }
        else if (smooth_normal) {
            Vector3 u = normalized((*surf)(p + dp * Vector2(1, 0)) - (*surf)(p + dp * Vector2(-1, 0)));
            Vector3 v = normalized((*surf)(p + dp * Vector2(0, 1)) - (*surf)(p + dp * Vector2(0, -1)));
            return normalized(cross(u, v));
        }
        else {
            // Useless value
            return Vector3(1, 0, 0);
        }
    };
    auto get_uv_at = [&](const Vector2 &p) -> Vector2 {
        if (uv) {
            return (*uv)(p);
        }
        else {
            return p;
        }
    };
    Array2D<Vector3> vertices(res + Vector2i(1));
    Array2D<Vector3> normals(res + Vector2i(1));
    Array2D<Vector2> uvs(res + Vector2i(1));
    for (int i = 0; i < res[0] + 1; i++) {
        for (int j = 0; j < res[1] + 1; j++) {
            Vector2 p = Vector2(i, j) / Vector2(res);
            vertices[i][j] = (*surf)(p);
            normals[i][j] = get_normal_at(p);
            uvs[i][j] = get_uv_at(p);
        }
    }
    std::vector<Triangle> triangles;
    int counter = 0;
    if (smooth_normal) {
        for (int i = 0; i < res[0]; i++) {
            for (int j = 0; j < res[1]; j++) {
                triangles.push_back(Triangle(
                    vertices[i][j],
                    vertices[i + 1][j],
                    vertices[i + 1][j + 1],
                    normals[i][j],
                    normals[i + 1][j],
                    normals[i + 1][j + 1],
                    uvs[i][j],
                    uvs[i + 1][j],
                    uvs[i + 1][j + 1],
                    counter++
                ));
                triangles.push_back(Triangle(
                    vertices[i][j],
                    vertices[i + 1][j + 1],
                    vertices[i][j + 1],
                    normals[i][j],
                    normals[i + 1][j + 1],
                    normals[i][j + 1],
                    uvs[i][j],
                    uvs[i + 1][j + 1],
                    uvs[i][j + 1],
                    counter++
                ));
            }
        }
    }
    else {
        for (int i = 0; i < res[0]; i++) {
            for (int j = 0; j < res[1]; j++) {
                Vector3 normal;
                normal = normalized(cross(vertices[i + 1][j] - vertices[i][j],
                    vertices[i + 1][j + 1] - vertices[i][j]));
                triangles.push_back(Triangle(
                    vertices[i][j],
                    vertices[i + 1][j],
                    vertices[i + 1][j + 1],
                    normal, normal, normal,
                    uvs[i][j],
                    uvs[i + 1][j],
                    uvs[i + 1][j + 1],
                    counter++
                ));
                normal = normalized(cross(vertices[i + 1][j + 1] - vertices[i][j],
                    vertices[i][j + 1] - vertices[i][j]));
                triangles.push_back(Triangle(
                    vertices[i][j],
                    vertices[i + 1][j + 1],
                    vertices[i][j + 1],
                    normal, normal, normal,
                    uvs[i][j],
                    uvs[i + 1][j + 1],
                    uvs[i][j + 1],
                    counter++
                ));
            }
        }
    }
    return triangles;
}

TC_NAMESPACE_END