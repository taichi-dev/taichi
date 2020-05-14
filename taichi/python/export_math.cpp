/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#include "taichi/python/export.h"
#include "taichi/common/dict.h"

TI_NAMESPACE_BEGIN

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
    using Vector = VectorND<dim, T, ISE>;

    // e.g. Vector4f
    std::string vector_name =
        std::string("Vector") + std::to_string(dim) + get_type_short_name<T>();

    auto cls = py::class_<Vector>(m, vector_name.c_str());
    cls.def(VectorInitializer<dim, T>::get())
        .def(py::init<T>())
        .def("__len__", [](Vector *) { return Vector::dim; })
        .def("__getitem__", [](Vector *vec, int i) { return (*vec)[i]; });

    register_vec_field<0, Vector>(cls);
    register_vec_field<1, Vector>(cls);
    register_vec_field<2, Vector>(cls);
    register_vec_field<3, Vector>(cls);
  }
};

void export_math(py::module &m) {
  VectorRegistration<Vector2f>::run(m);
  VectorRegistration<Vector3f>::run(m);
  VectorRegistration<Vector4f>::run(m);

  VectorRegistration<Vector2d>::run(m);
  VectorRegistration<Vector3d>::run(m);
  VectorRegistration<Vector4d>::run(m);

  VectorRegistration<Vector2i>::run(m);
  VectorRegistration<Vector3i>::run(m);
  VectorRegistration<Vector4i>::run(m);
}

TI_NAMESPACE_END
