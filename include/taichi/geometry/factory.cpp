#include <taichi/geometry/factory.h>

TC_NAMESPACE_BEGIN

    std::vector<Triangle> Mesh3D::generate(const SurfaceGenerator &func_, Vector2i res) {
        Vector2 dp = Vector2(1.0) / Vector2(res);
        auto func = [&](const Vector2 &p) -> Vector3 {
            return func_(p.x, p.y);
        };
        auto get_normal_at = [&](const Vector2 &p) -> Vector3 {
            Vector3 u = normalized(func(p + dp * Vector2(1, 0)) - func(p + dp * Vector2(-1, 0)));
            Vector3 v = normalized(func(p + dp * Vector2(0, 1)) - func(p + dp * Vector2(0, -1)));
            return normalized(cross(u, v));
        };
        Array2D<Vector3> vertices(res + Vector2i(1));
        Array2D<Vector3> normals(res + Vector2i(1));
        Array2D<Vector2> uvs(res + Vector2i(1));
        for (int i = 0; i < res[0] + 1; i++) {
            for (int j = 0; j < res[1] + 1; j++) {
                Vector2 p = Vector2(i, j) / Vector2(res);
                vertices[i][j] = func(p);
                normals[i][j] = get_normal_at(p);
                uvs[i][j] = p;
            }
        }
        std::vector<Triangle> triangles;
        int counter = 0;
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
        return triangles;
    }

TC_NAMESPACE_END