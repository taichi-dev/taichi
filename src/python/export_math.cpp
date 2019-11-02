/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#include <taichi/python/export.h>
#include <taichi/common/dict.h>
#include <taichi/visualization/rgb.h>
#include <taichi/image/operations.h>

// TODO: these "make_opaque" macros disables pybind11 from resolving [1] as
// std::vector<int>. Why?
/*
PYBIND11_MAKE_OPAQUE(std::vector<int>);
PYBIND11_MAKE_OPAQUE(std::vector<taichi::float32>);
PYBIND11_MAKE_OPAQUE(std::vector<taichi::float64>);
PYBIND11_MAKE_OPAQUE(std::vector<taichi::Vector2>);
PYBIND11_MAKE_OPAQUE(std::vector<taichi::Vector3>);
PYBIND11_MAKE_OPAQUE(std::vector<taichi::Vector4>);
PYBIND11_MAKE_OPAQUE(std::vector<taichi::Vector2i>);
PYBIND11_MAKE_OPAQUE(std::vector<taichi::Vector3i>);
PYBIND11_MAKE_OPAQUE(std::vector<taichi::Vector4i>);
 */

TC_NAMESPACE_BEGIN
std::vector<real> make_range(real start, real end, real delta) {
  return std::vector<real>{start, end, delta};
}

template <typename T, int ret>
int return_constant(T *) {
  return ret;
}

template <typename T, int channels>
void ndarray_to_image_buffer(T *arr,
                             uint64 input,
                             int width,
                             int height)  // 'input' is actually a pointer...
{
  arr->initialize(Vector2i(width, height));
  for (auto &ind : arr->get_region()) {
    for (int i = 0; i < channels; i++) {
      (*arr)[ind][i] = reinterpret_cast<real *>(
          input)[ind.i * channels * height + ind.j * channels + i];
    }
  }
}

template <typename T, int channels>
void ndarray_to_array2d(T *arr,
                        long long input,
                        int width,
                        int height)  // 'input' is actually a pointer...
{
  arr->initialize(Vector2i(width, height));
  for (auto &ind : arr->get_region()) {
    for (int i = 0; i < channels; i++) {
      (*arr)[ind][i] = reinterpret_cast<float *>(
          input)[ind.i * height * channels + ind.j * channels + i];
    }
  }
}

void ndarray_to_array2d_real(Array2D<real> *arr,
                             long long input,
                             int width,
                             int height)  // 'input' is actually a pointer...
{
  arr->initialize(Vector2i(width, height));
  for (auto &ind : arr->get_region()) {
    (*arr)[ind] = reinterpret_cast<float *>(input)[ind.i * height + ind.j];
  }
}

template <typename T, int channels>
void array2d_to_ndarray(T *arr,
                        uint64 output)  // 'output' is actually a pointer...
{
  int width = arr->get_width(), height = arr->get_height();
  TC_ASSERT(width > 0);
  TC_ASSERT(height > 0);
  for (auto &ind : arr->get_region()) {
    for (int k = 0; k < channels; k++) {
      reinterpret_cast<real *>(
          output)[ind.i * height * channels + ind.j * channels + k] =
          *(((float *)(&(*arr)[ind])) + k);
    }
  }
}

// Specialize for Vector3 for the padding float32
template <int channels>
void array2d_to_ndarray(Array2D<Vector3> *arr,
                        uint64 output)  // 'output' is actually a pointer...
{
  int width = arr->get_width(), height = arr->get_height();
  assert_info(width > 0, "");
  assert_info(height > 0, "");
  for (auto &ind : arr->get_region()) {
    for (int k = 0; k < channels; k++) {
      const Vector3 entry = (*arr)[ind];
      reinterpret_cast<real *>(
          output)[ind.i * height * channels + ind.j * channels + k] = entry[k];
    }
  }
}

template <typename T>
constexpr std::string get_type_short_name();

template <>
std::string get_type_short_name<float32>() {
  return "f";
}

template <>
std::string get_type_short_name<float64>() {
  return "d";
}

template <>
std::string get_type_short_name<int>() {
  return "i";
}

template <>
std::string get_type_short_name<int64>() {
  return "I";
}

template <>
std::string get_type_short_name<uint64>() {
  return "U";
}

template <typename T>
struct get_dim {};

template <int dim, typename T, InstSetExt ISE>
struct get_dim<VectorND<dim, T, ISE>> {
  constexpr static int value = dim;
};

template <int dim, typename T>
struct VectorInitializer {};

template <typename T>
struct VectorInitializer<1, T> {
  static auto get() {
    return py::init<T>();
  }
};

template <typename T>
struct VectorInitializer<2, T> {
  static auto get() {
    return py::init<T, T>();
  }
};

template <typename T>
struct VectorInitializer<3, T> {
  static auto get() {
    return py::init<T, T, T>();
  }
};

template <typename T>
struct VectorInitializer<4, T> {
  static auto get() {
    return py::init<T, T, T, T>();
  }
};

template <int i, typename VEC>
struct get_vec_field {};

template <typename VEC>
struct get_vec_field<0, VEC> {
  static auto get() {
    return &VEC::x;
  }
};

template <typename VEC>
struct get_vec_field<1, VEC> {
  static auto get() {
    return &VEC::y;
  }
};

template <typename VEC>
struct get_vec_field<2, VEC> {
  static auto get() {
    return &VEC::z;
  }
};

template <typename VEC>
struct get_vec_field<3, VEC> {
  static auto get() {
    return &VEC::w;
  }
};

template <int i,
          typename VEC,
          typename Class,
          std::enable_if_t<get_dim<VEC>::value<i + 1, int> = 0> void
              register_vec_field(Class &cls) {
}

template <int i,
          typename VEC,
          typename Class,
          std::enable_if_t<get_dim<VEC>::value >= i + 1, int> = 0>
void register_vec_field(Class &cls) {
  static const char *names[4] = {"x", "y", "z", "w"};
  cls.def_readwrite(names[i], get_vec_field<i, VEC>::get());
}

template <typename T>
struct VectorRegistration {};

template <int dim, typename T, InstSetExt ISE>
struct VectorRegistration<VectorND<dim, T, ISE>> {
  static void run(py::module &m) {
    using VectorBase = VectorNDBase<dim, T, ISE>;
    using Vector = VectorND<dim, T, ISE>;

    // e.g. Vector4f
    std::string vector_name =
        std::string("Vector") + std::to_string(dim) + get_type_short_name<T>();
    // e.g. Vector4fBase
    std::string base_name = vector_name + "Base";

    py::class_<VectorBase>(m, base_name.c_str());

    auto cls = py::class_<Vector, VectorBase>(m, vector_name.c_str());
    cls.def(VectorInitializer<dim, T>::get())
        .def(py::init<T>())
        .def(T() * py::self)
        .def(py::self * T())
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def(py::self * py::self)
        .def(py::self / py::self)
        .def(py::self += py::self)
        .def(py::self -= py::self)
        .def(py::self *= py::self)
        .def(py::self /= py::self)
        .def(py::self /= py::self)
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def(-py::self)
        .def("cast_real", &Vector::template cast<real>)
        .def("cast_float32", &Vector::template cast<float32>)
        .def("cast_float64", &Vector::template cast<float64>)
        .def("cast_int", &Vector::template cast<int>)
        .def("D", [](Vector *) { return Vector::dim; })
        .def("__len__", [](Vector *) { return Vector::dim; })
        //.def("__getitem__", static_cast<T
        //(Vector::*)(int)>(&Vector::operator[]))
        .def("__getitem__", [](Vector *vec, int i) { return (*vec)[i]; })
        .def("ise", [](Vector *) { return Vector::ise; })
        .def("simd", [](Vector *) { return Vector::simd; })
        .def("abs", &Vector::abs)
        .def("floor", &Vector::floor)
        .def("sin", &Vector::sin)
        .def("cos", &Vector::cos)
        .def("min", &Vector::min)
        .def("max", &Vector::max);

    register_vec_field<0, Vector>(cls);
    register_vec_field<1, Vector>(cls);
    register_vec_field<2, Vector>(cls);
    register_vec_field<3, Vector>(cls);

    DEFINE_VECTOR_OF_NAMED(Vector, (vector_name + "List").c_str());
  }
};

void export_math(py::module &m) {
  py::class_<Config>(m, "Config");

#define EXPORT_ARRAY_2D_OF(T, C)                             \
  py::class_<Array2D<real>> PyArray2D##T(m, "Array2D" #T);   \
  PyArray2D##T.def(py::init<Vector2i>())                     \
      .def("to_ndarray", &array2d_to_ndarray<Array2D<T>, C>) \
      .def("get_width", &Array2D<T>::get_width)              \
      .def("get_height", &Array2D<T>::get_height)            \
      .def("rasterize", &Array2D<T>::rasterize)              \
      .def("rasterize_scale", &Array2D<T>::rasterize_scale)  \
      .def("from_ndarray", &ndarray_to_array2d_real);

  EXPORT_ARRAY_2D_OF(real, 1);

#define EXPORT_ARRAY_3D_OF(T, C)                           \
  py::class_<Array3D<real>> PyArray3D##T(m, "Array3D" #T); \
  PyArray3D##T.def(py::init<Vector3i>())                   \
      .def("get_width", &Array3D<T>::get_width)            \
      .def("get_height", &Array3D<T>::get_height);

  EXPORT_ARRAY_3D_OF(real, 1);

  py::class_<Array2D<Vector3>>(m, "Array2DVector3")
      .def(py::init<Vector2i, Vector3>())
      .def("get_width", &Array2D<Vector3>::get_width)
      .def("get_height", &Array2D<Vector3>::get_height)
      .def("get_channels", &return_constant<Array2D<Vector3>, 3>)
      .def("from_ndarray", &ndarray_to_image_buffer<Array2D<Vector3>, 3>)
      .def("read", &Array2D<Vector3>::load_image)
      .def("write", &Array2D<Vector3>::write_as_image)
      .def("write_to_disk", &Array2D<Vector3>::write_to_disk)
      .def("read_from_disk", &Array2D<Vector3>::read_from_disk)
      .def("rasterize", &Array2D<Vector3>::rasterize)
      .def("rasterize_scale", &Array2D<Vector3>::rasterize_scale)
      .def("to_ndarray", &array2d_to_ndarray<Array2D<Vector3>, 3>);

  py::class_<Array2D<Vector4>>(m, "Array2DVector4")
      .def(py::init<Vector2i, Vector4>())
      .def("get_width", &Array2D<Vector4>::get_width)
      .def("get_height", &Array2D<Vector4>::get_height)
      .def("get_channels", &return_constant<Array2D<Vector4>, 4>)
      .def("write", &Array2D<Vector4>::write_as_image)
      .def("from_ndarray", &ndarray_to_image_buffer<Array2D<Vector4>, 4>)
      .def("write_to_disk", &Array2D<Vector4>::write_to_disk)
      .def("read_from_disk", &Array2D<Vector4>::read_from_disk)
      .def("rasterize", &Array2D<Vector4>::rasterize)
      .def("rasterize_scale", &Array2D<Vector4>::rasterize_scale)
      .def("to_ndarray", &array2d_to_ndarray<Array2D<Vector4>, 4>);

  m.def("points_inside_polygon", points_inside_polygon);
  m.def("points_inside_sphere", points_inside_sphere);
  m.def("make_range", make_range);

  py::class_<Matrix4>(m, "Matrix4")
      .def(py::init<real>())
      .def(real() * py::self)
      .def(py::self + py::self)
      .def(py::self - py::self)
      .def(py::self * py::self)
      .def("translate", &matrix4_translate)
      .def("scale", &matrix4_scale)
      .def("scale_s", &matrix4_scale_s)
      .def("rotate_euler", &matrix4_rotate_euler)
      .def("rotate_angle_axis", &matrix4_rotate_angle_axis)
      .def("get_ptr_string", &Config::get_ptr_string<Matrix4>);

  // VectorRegistration<Vector1>::run(m);
  VectorRegistration<Vector2f>::run(m);
  VectorRegistration<Vector3f>::run(m);
  VectorRegistration<Vector4f>::run(m);

  // VectorRegistration<Vector1d>::run(m);
  VectorRegistration<Vector2d>::run(m);
  VectorRegistration<Vector3d>::run(m);
  VectorRegistration<Vector4d>::run(m);

  // VectorRegistration<Vector1i>::run(m);
  VectorRegistration<Vector2i>::run(m);
  VectorRegistration<Vector3i>::run(m);
  VectorRegistration<Vector4i>::run(m);

  DEFINE_VECTOR_OF(real);
  DEFINE_VECTOR_OF(int);
}

TC_NAMESPACE_END
