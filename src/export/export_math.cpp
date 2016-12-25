#include <taichi/python/export.h>

#include <taichi/common/config.h>
#include <taichi/levelset/levelset2d.h>
#include <taichi/visualization/rgb.h>

using namespace boost::python;
namespace py = boost::python;

TC_NAMESPACE_BEGIN
    std::vector<real> make_range(real start, real end, real delta) {
        return std::vector<real> {start, end, delta};
    }

    std::string rasterize_levelset(const LevelSet2D &levelset, int width, int height) {
        std::string ret;
        for (auto &ind : Region2D(0, width, 0, height)) {
            real c = -levelset.sample((ind.i + 0.5f) / width * levelset.get_width(),
                                      (ind.j + 0.5f) / height * levelset.get_height());
            RGB rgb(c, c, c);
            rgb.append_to_string(ret);
        }
        return ret;
    }

    Matrix4 matrix4_translate(Matrix4 *transform, const Vector3 &offset) {
        return glm::translate(Matrix4(1.0f), offset) * *transform;
    }

    Matrix4 matrix4_scale(Matrix4 *transform, const Vector3 &scales) {
        return glm::scale(Matrix4(1.0f), scales) * *transform;
    }

    Matrix4 matrix4_scale_s(Matrix4 *transform, real s) {
        return matrix4_scale(transform, Vector3(s));
    }

    Matrix4 matrix4_rotate_angle_axis(Matrix4 *transform, real angle, const Vector3 &axis) {
        return glm::rotate(Matrix4(1.0f), angle * pi / 180.0f, axis) * *transform;
    }

    Matrix4 matrix4_rotate_euler(Matrix4 *transform, const Vector3 &euler_angles) {
        Matrix4 ret = *transform;
        ret = matrix4_rotate_angle_axis(&ret, euler_angles.x, Vector3(1.0f, 0.0f, 0.0f));
        ret = matrix4_rotate_angle_axis(&ret, euler_angles.y, Vector3(0.0f, 1.0f, 0.0f));
        ret = matrix4_rotate_angle_axis(&ret, euler_angles.z, Vector3(0.0f, 0.0f, 1.0f));
        return ret;
    }

    void export_math() {
        def("rasterize_levelset", rasterize_levelset);

        class_<Config>("Config");
        numeric::array::set_module_and_type("numpy", "ndarray");
        class_<Array>("Array2DFloat")
                .def("to_ndarray", &array2d_to_ndarray<Array2D<real >>)
                .def("rasterize", &Array2D<real>::rasterize);

        class_<LevelSet2D>("LevelSet2D", init<int, int, Vector2>())
                .def("get", &LevelSet2D::get_copy)
                .def("set", static_cast<void (LevelSet2D::*)(int, int, const real &)>(&LevelSet2D::set))
                .def("add_sphere", &LevelSet2D::add_sphere)
                .def("add_polygon", &LevelSet2D::add_polygon)
                .def("get_gradient", &LevelSet2D::get_gradient)
                .def("rasterize", &LevelSet2D::rasterize)
                .def("sample", static_cast<real (LevelSet2D::*)(real, real) const>(&LevelSet2D::sample))
                .def("get_normalized_gradient", &LevelSet2D::get_normalized_gradient)
                .def("to_ndarray", &array2d_to_ndarray<LevelSet2D>)
                .def_readwrite("friction", &LevelSet2D::friction);
        def("points_inside_polygon", points_inside_polygon);
        def("points_inside_sphere", points_inside_sphere);
        def("make_range", make_range);
        class_<Matrix4>("Matrix4", init<real>())
                .def(real() * self)
                .def(self + self)
                .def(self - self)
                .def(self * self)
                .def(self / self)
                .def("translate", &matrix4_translate)
                .def("scale", &matrix4_scale)
                .def("scale_s", &matrix4_scale_s)
                .def("rotate_euler", &matrix4_rotate_euler)
                .def("rotate_angle_axis", &matrix4_rotate_angle_axis)
                .def("get_ptr_string", &Config::get_ptr_string<Matrix4>);

        class_<Vector2i>("Vector2i", init<int, int>())
                .def_readwrite("x", &Vector2i::x)
                .def_readwrite("y", &Vector2i::y)
                .def(self * int())
                .def(int() * self)
                .def(self / int())
                .def(self + self)
                .def(self - self)
                .def(self * self)
                .def(self / self);

        class_<Vector2>("Vector2", init<real, real>())
                .def_readwrite("x", &Vector2::x)
                .def_readwrite("y", &Vector2::y)
                .def(self * real())
                .def(real() * self)
                .def(self / real())
                .def(self + self)
                .def(self - self)
                .def(self * self)
                .def(self / self);

        class_<Vector3i>("Vector3i", init<int, int, int>())
                .def_readwrite("x", &Vector3i::x)
                .def_readwrite("y", &Vector3i::y)
                .def_readwrite("z", &Vector3i::z)
                .def(self * int())
                .def(int() * self)
                .def(self / int())
                .def(self + self)
                .def(self - self)
                .def(self * self)
                .def(self / self);

        class_<Vector3>("Vector3", init<real, real, real>())
                .def_readwrite("x", &Vector3::x)
                .def_readwrite("y", &Vector3::y)
                .def_readwrite("z", &Vector3::z)
                .def(self * real())
                .def(real() * self)
                .def(self / real())
                .def(self + self)
                .def(self - self)
                .def(self * self)
                .def(self / self);

        class_<Vector4>("Vector4", init<real, real, real, real>())
                .def_readwrite("x", &Vector4::x)
                .def_readwrite("y", &Vector4::y)
                .def_readwrite("z", &Vector4::z)
                .def_readwrite("w", &Vector4::w)
                .def(self * real())
                .def(real() * self)
                .def(self / real())
                .def(self + self)
                .def(self - self)
                .def(self * self)
                .def(self / self);

        DEFINE_VECTOR_OF(real);
        DEFINE_VECTOR_OF(int);
        DEFINE_VECTOR_OF(Vector2);
        DEFINE_VECTOR_OF(Vector3);
        DEFINE_VECTOR_OF(Vector4);
        DEFINE_VECTOR_OF(Vector2i);
        DEFINE_VECTOR_OF(Vector3i);
        DEFINE_VECTOR_OF(Vector4i);
    }

TC_NAMESPACE_END
