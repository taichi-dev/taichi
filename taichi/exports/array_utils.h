#pragma once

#define TIE_DECL_ARRAY_FUNCTIONS_IMPL(ArrayType, ArrayHandle, ArrayRef,    \
                                      ObjectType, ObjectHandle, ObjectRef) \
  TI_DLL_EXPORT int TI_API_CALL tie_##ArrayType##_destroy(ArrayRef self);  \
  TI_DLL_EXPORT int TI_API_CALL tie_##ArrayType##_size(                    \
      ArrayRef self, size_t *ret_num_elements);                            \
  TI_DLL_EXPORT int TI_API_CALL tie_##ArrayType##_get_element(             \
      ArrayRef self, size_t index, ObjectRef *ret_element)

#define TIE_IMPL_ARRAY_FUNCTIONS_IMPL(ArrayType, ArrayHandle, ArrayRef,    \
                                      ObjectType, ObjectHandle, ObjectRef) \
  int tie_##ArrayType##_destroy(ArrayRef self) {                           \
    TIE_FUNCTION_BODY_BEGIN();                                             \
    TIE_CHECK_HANDLE(self);                                                \
    ArrayType##Impl *impl = (ArrayType##Impl *)self;                       \
    delete impl;                                                           \
    TIE_FUNCTION_BODY_END();                                               \
  }                                                                        \
  int tie_##ArrayType##_size(ArrayRef self, size_t *ret_num_elements) {    \
    TIE_FUNCTION_BODY_BEGIN();                                             \
    TIE_CHECK_HANDLE(self);                                                \
    TIE_CHECK_RETURN_ARG(ret_num_elements);                                \
    ArrayType##Impl *impl = (ArrayType##Impl *)self;                       \
    *ret_num_elements = impl->data.size();                                 \
    TIE_FUNCTION_BODY_END();                                               \
  }                                                                        \
  int tie_##ArrayType##_get_element(ArrayRef self, size_t index,           \
                                    ObjectRef *ret_element) {              \
    TIE_FUNCTION_BODY_BEGIN();                                             \
    TIE_CHECK_HANDLE(self);                                                \
    TIE_CHECK_RETURN_ARG(ret_element);                                     \
    ArrayType##Impl *impl = (ArrayType##Impl *)self;                       \
    TIE_CHECK_INDEX_ARG(index, impl->data.size());                         \
    *ret_element = (ObjectRef)&impl->data[index];                          \
    TIE_FUNCTION_BODY_END();                                               \
  }

#define TIE_IMPL_ARRAY_UTILS_IMPL(ArrayType, ArrayHandle, ArrayRef,    \
                                  ObjectType, ObjectHandle, ObjectRef, \
                                  ObjectTypeFullName)                  \
  namespace {                                                          \
  struct ArrayType##Impl {                                             \
    std::vector<ObjectTypeFullName> data;                              \
    template <typename... Args>                                        \
    explicit ArrayType##Impl(Args &&...args)                           \
        : data(std::forward<Args>(args)...) {}                         \
  };                                                                   \
  template <typename... Args>                                          \
  ArrayType##Impl *tie_api_create_##ArrayType##_impl(Args &&...args) { \
    return new ArrayType##Impl(std::forward<Args>(args)...);           \
  }                                                                    \
  }

#define TIE_TYPEDEF_ARRAY_HANDLE_AND_REF(Type) \
  typedef TieHandle Tie##Type##ArrayHandle;    \
  typedef TieRef Tie##Type##ArrayRef

#define TIE_DECL_ARRAY_FUNCTIONS(Type)                                        \
  TIE_DECL_ARRAY_FUNCTIONS_IMPL(Type##Array, Tie##Type##ArrayHandle,          \
                                Tie##Type##ArrayRef, Type, Tie##Type##Handle, \
                                Tie##Type##Ref)

#define TIE_IMPL_ARRAY_FUNCTIONS(Type)                                        \
  TIE_IMPL_ARRAY_FUNCTIONS_IMPL(Type##Array, Tie##Type##ArrayHandle,          \
                                Tie##Type##ArrayRef, Type, Tie##Type##Handle, \
                                Tie##Type##Ref)

#define TIE_IMPL_ARRAY_UTILS(Type, TypeFullName)                          \
  TIE_IMPL_ARRAY_UTILS_IMPL(Type##Array, Tie##Type##ArrayHandle,          \
                            Tie##Type##ArrayRef, Type, Tie##Type##Handle, \
                            Tie##Type##Ref, TypeFullName)
