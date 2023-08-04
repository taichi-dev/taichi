#include "exports.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <exception>

#include "taichi/common/exceptions.h"
#include "taichi/system/timeline.h"
#include "taichi/ir/frontend_ir.h"
#include "taichi/program/extension.h"
#include "taichi/program/function.h"
#include "taichi/program/kernel.h"
#include "taichi/program/kernel_profiler.h"
#include "taichi/program/sparse_matrix.h"
#include "taichi/program/program.h"
#include "taichi/program/launch_context_builder.h"

#define TIE_WRAP_ERROR_MSG(msg)                         \
  std::string("Tie-API error (")                        \
      .append(__func__)                                 \
      .append("): " msg " (triggered at " __FILE__ ":") \
      .append(std::to_string(__LINE__))                 \
      .append(")")

#define TIE_INVALID_ARGUMENT(arg)                              \
  do {                                                         \
    tie_api_set_last_error_impl(                               \
        TIE_ERROR_INVALID_ARGUMENT,                            \
        TIE_WRAP_ERROR_MSG("Argument '" #arg "' is nullptr")); \
    return TIE_ERROR_INVALID_ARGUMENT;                         \
  } while (0)

#define TIE_CHECK_PTR_NOT_NULL(ptr) \
  if (ptr == nullptr)               \
    TIE_INVALID_ARGUMENT(ptr);

#define TIE_CHECK_RETURN_ARG(ret_param)                                     \
  if (ret_param == nullptr) {                                               \
    tie_api_set_last_error_impl(                                            \
        TIE_ERROR_INVALID_RETURN_ARG,                                       \
        TIE_WRAP_ERROR_MSG("Return argument '" #ret_param "' is nullptr")); \
    return TIE_ERROR_INVALID_RETURN_ARG;                                    \
  }

#define TIE_CHECK_HANDLE(handle)                                \
  if (handle == nullptr) {                                      \
    tie_api_set_last_error_impl(                                \
        TIE_ERROR_INVALID_HANDLE,                               \
        TIE_WRAP_ERROR_MSG("Handle '" #handle "' is nullptr")); \
    return TIE_ERROR_INVALID_HANDLE;                            \
  }

#define TIE_CHECK_INDEX_ARG(index, size)                           \
  if (index >= size) {                                             \
    tie_api_set_last_error_impl(                                   \
        TIE_ERROR_INVALID_INDEX,                                   \
        TIE_WRAP_ERROR_MSG("Index '" #index "' is out of range")); \
    return TIE_ERROR_INVALID_INDEX;                                \
  }

#define TIE_MAKE_CALLBACK(function_argument) \
  tie_api_make_callback_impl(                \
      function_argument,                     \
      TIE_WRAP_ERROR_MSG("Call " #function_argument " failed"))

#define TIE_FUNCTION_BODY_BEGIN() try {
#define TIE_FUNCTION_BODY_END()                                              \
  }                                                                          \
  catch (const taichi::lang::TaichiTypeError &e) {                           \
    tie_api_set_last_error_impl(TIE_ERROR_TAICHI_TYPE_ERROR, e.what());      \
    return TIE_ERROR_TAICHI_TYPE_ERROR;                                      \
  }                                                                          \
  catch (const taichi::lang::TaichiSyntaxError &e) {                         \
    tie_api_set_last_error_impl(TIE_ERROR_TAICHI_SYNTAX_ERROR, e.what());    \
    return TIE_ERROR_TAICHI_SYNTAX_ERROR;                                    \
  }                                                                          \
  catch (const taichi::lang::TaichiIndexError &e) {                          \
    tie_api_set_last_error_impl(TIE_ERROR_TAICHI_INDEX_ERROR, e.what());     \
    return TIE_ERROR_TAICHI_INDEX_ERROR;                                     \
  }                                                                          \
  catch (const taichi::lang::TaichiRuntimeError &e) {                        \
    tie_api_set_last_error_impl(TIE_ERROR_TAICHI_RUNTIME_ERROR, e.what());   \
    return TIE_ERROR_TAICHI_RUNTIME_ERROR;                                   \
  }                                                                          \
  catch (const taichi::lang::TaichiAssertionError &e) {                      \
    tie_api_set_last_error_impl(TIE_ERROR_TAICHI_ASSERTION_ERROR, e.what()); \
    return TIE_ERROR_TAICHI_ASSERTION_ERROR;                                 \
  }                                                                          \
  catch (const TieCallbackFailedException &e) {                              \
    tie_api_set_last_error_impl(TIE_ERROR_CALLBACK_FAILED, e.what());        \
    return TIE_ERROR_CALLBACK_FAILED;                                        \
  }                                                                          \
  catch (const std::bad_alloc &e) {                                          \
    tie_api_set_last_error_impl(TIE_ERROR_OUT_OF_MEMORY, e.what());          \
    return TIE_ERROR_OUT_OF_MEMORY;                                          \
  }                                                                          \
  catch (const std::exception &e) {                                          \
    tie_api_set_last_error_impl(TIE_ERROR_UNKNOWN_CXX_EXCEPTION, e.what());  \
    return TIE_ERROR_UNKNOWN_CXX_EXCEPTION;                                  \
  }                                                                          \
  catch (const std::string &e) {                                             \
    tie_api_set_last_error_impl(TIE_ERROR_UNKNOWN_CXX_EXCEPTION, e);         \
    return TIE_ERROR_UNKNOWN_CXX_EXCEPTION;                                  \
  }                                                                          \
  catch (...) {                                                              \
    tie_api_set_last_error_impl(TIE_ERROR_UNKNOWN_CXX_EXCEPTION,             \
                                "C++ Exception");                            \
    return TIE_ERROR_UNKNOWN_CXX_EXCEPTION;                                  \
  }                                                                          \
  return TIE_ERROR_SUCCESS

namespace {

class TieApiException : std::exception {
 public:
  explicit TieApiException(std::string msg) : msg_(std::move(msg)) {
  }

  const char *what() const noexcept override {
    return msg_.c_str();
  }

 private:
  std::string msg_;
};

class TieCallbackFailedException : public TieApiException {
 public:
  using TieApiException::TieApiException;
};

struct TieErrorCache {
  int error{TIE_ERROR_SUCCESS};
  std::string msg;
};

template <typename Dest, typename Src>
struct TieAttrAssign {
  static void assign(Dest &dest, const Src &src) {
    dest = static_cast<Dest>(src);
  }
};

template <>
struct TieAttrAssign<std::string, const char *> {
  static void assign(std::string &dest, const char *src) {
    dest = src;
  }
};

template <>
struct TieAttrAssign<const char *, std::string> {
  static void assign(const char *&dest, const std::string &src) {
    dest = src.c_str();
  }
};

template <>
struct TieAttrAssign<taichi::lang::DataType, TieDataTypeRef> {
  static void assign(taichi::lang::DataType &dest, TieDataTypeRef src) {
    dest = *reinterpret_cast<taichi::lang::DataType *>(src);
  }
};

template <>
struct TieAttrAssign<TieDataTypeRef, taichi::lang::DataType> {
  static void assign(TieDataTypeRef &dest,
                     const taichi::lang::DataType &src) {
    dest = reinterpret_cast<TieDataTypeRef>(
        const_cast<taichi::lang::DataType *>(&src));
  }
};

static thread_local TieErrorCache tie_last_error;

void tie_api_set_last_error_impl(int error, std::string msg) {
  tie_last_error.error = error;
  tie_last_error.msg = std::move(msg);
}

std::function<void()> tie_api_make_callback_impl(TieCallback func,
                                                 std::string msg) {
  return [func, _msg = std::move(msg)]() {
    if (func() != 0) {
      throw TieCallbackFailedException(std::move(_msg));
    }
  };
}
}  // namespace

// std::string

int tie_String_create(const char *str, TieStringHandle *ret_handle) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_PTR_NOT_NULL(str);
  TIE_CHECK_RETURN_ARG(ret_handle);
  *ret_handle = reinterpret_cast<TieStringHandle>(new std::string(str));
  TIE_FUNCTION_BODY_END();
}

int tie_String_destroy(TieStringHandle self) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_HANDLE(self);
  delete reinterpret_cast<std::string *>(self);
  TIE_FUNCTION_BODY_END();
}

int tie_String_c_str(TieStringHandle self, const char **ret_c_str) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_HANDLE(self);
  TIE_CHECK_RETURN_ARG(ret_c_str);
  *ret_c_str = reinterpret_cast<std::string *>(self)->c_str();
  TIE_FUNCTION_BODY_END();
}

int tie_String_size(TieStringHandle self, size_t *ret_size) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_HANDLE(self);
  TIE_CHECK_RETURN_ARG(ret_size);
  *ret_size = reinterpret_cast<std::string *>(self)->size();
  TIE_FUNCTION_BODY_END();
}

// Error proessing

int tie_G_set_last_error(int error, const char *msg) {
  tie_api_set_last_error_impl(error, msg ? msg : "");
  return TIE_ERROR_SUCCESS;
}

int tie_G_get_last_error(int *ret_error, const char **ret_msg) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_RETURN_ARG(ret_error);
  TIE_CHECK_RETURN_ARG(ret_msg);
  *ret_error = tie_last_error.error;
  *ret_msg = tie_last_error.msg.c_str();
  TIE_FUNCTION_BODY_END();
}

// Arch handling
int tie_G_arch_name(int arch, const char **ret_name) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_RETURN_ARG(ret_name);

  switch (static_cast<taichi::Arch>(arch)) {
#define PER_ARCH(x)     \
  case taichi::Arch::x: \
    *ret_name = #x;     \
    break;
#include "taichi/inc/archs.inc.h"
#undef PER_ARCH
    default:
      TIE_INVALID_ARGUMENT(arch);
  }

  TIE_FUNCTION_BODY_END();
}

int tie_G_arch_from_name(const char *name, int *ret_arch) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_PTR_NOT_NULL(name);
  TIE_CHECK_RETURN_ARG(ret_arch);
  *ret_arch = (TieArch)taichi::arch_from_name(name);
  TIE_FUNCTION_BODY_END();
}

int tie_G_host_arch(int *ret_arch) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_RETURN_ARG(ret_arch);
  *ret_arch = static_cast<TieArch>(taichi::host_arch());
  TIE_FUNCTION_BODY_END();
}

int tie_G_is_extension_supported(int arch, int extension, bool *ret_supported) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_RETURN_ARG(ret_supported);
  *ret_supported = taichi::lang::is_extension_supported(
      static_cast<taichi::Arch>(arch),
      static_cast<taichi::lang::Extension>(extension));
  TIE_FUNCTION_BODY_END();
}

// default_compile_config handling

int tie_G_default_compile_config(TieCompileConfigRef *ret_handle) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_RETURN_ARG(ret_handle);
  *ret_handle = reinterpret_cast<TieCompileConfigRef>(
      &taichi::lang::default_compile_config);
  TIE_FUNCTION_BODY_END();
}

int tie_G_reset_default_compile_config() {
  TIE_FUNCTION_BODY_BEGIN();
  taichi::lang::default_compile_config = taichi::lang::CompileConfig();
  TIE_FUNCTION_BODY_END();
}

// class CompileConfig
int tie_CompileConfig_create(TieCompileConfigHandle *ret_handle) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_RETURN_ARG(ret_handle);
  auto *config = new taichi::lang::CompileConfig();
  *ret_handle = reinterpret_cast<TieCompileConfigHandle>(config);
  TIE_FUNCTION_BODY_END();
}

int tie_CompileConfig_destroy(TieCompileConfigRef self) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_HANDLE(self);
  auto *config = reinterpret_cast<taichi::lang::CompileConfig *>(self);
  delete config;
  TIE_FUNCTION_BODY_END();
}

#define TIE_IMPL_COMPILE_CONFIG_GET_SET_ATTR(TaichiStruct, attr_name,         \
                                             attr_type, get_set_type)         \
  int tie_CompileConfig_get_##attr_name(TieCompileConfigRef self,          \
                                        get_set_type *ret_value) {            \
    TIE_FUNCTION_BODY_BEGIN();                                                \
    TIE_CHECK_HANDLE(self);                                                   \
    TIE_CHECK_RETURN_ARG(ret_value);                                          \
    auto *config = reinterpret_cast<taichi::lang::CompileConfig *>(self);     \
    TieAttrAssign<get_set_type, attr_type>::assign(*ret_value,                \
                                                   config->attr_name);        \
    TIE_FUNCTION_BODY_END();                                                  \
  }                                                                           \
  int tie_CompileConfig_set_##attr_name(TieCompileConfigRef self,          \
                                        get_set_type value) {                 \
    TIE_FUNCTION_BODY_BEGIN();                                                \
    TIE_CHECK_HANDLE(self);                                                   \
    auto *config = reinterpret_cast<taichi::lang::CompileConfig *>(self);     \
    TieAttrAssign<attr_type, get_set_type>::assign(config->attr_name, value); \
    TIE_FUNCTION_BODY_END();                                                  \
  }

#define TIE_PER_COMPILE_CONFIG_ATTR TIE_IMPL_COMPILE_CONFIG_GET_SET_ATTR
#include "taichi/exports/inc/compile_config.inc.h"
#undef TIE_PER_COMPILE_CONFIG_ATTR
#undef TIE_DECL_COMPILE_CONFIG_GET_SET_ATTR

// class Kernel

int tie_Kernel_insert_scalar_param(TieKernelRef self,
                                   TieDataTypeRef dt,
                                   const char *name,
                                   int *ret_param_index) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_HANDLE(self);
  TIE_CHECK_HANDLE(dt);
  TIE_CHECK_PTR_NOT_NULL(name);
  TIE_CHECK_RETURN_ARG(ret_param_index);
  auto *kernel = reinterpret_cast<taichi::lang::Kernel *>(self);
  auto *data_type = reinterpret_cast<taichi::lang::DataType *>(dt);
  *ret_param_index = kernel->insert_scalar_param(*data_type, name);
  TIE_FUNCTION_BODY_END();
}

int tie_Kernel_insert_arr_param(TieKernelRef self,
                                TieDataTypeRef dt,
                                int total_dim,
                                int *ap_element_shape,
                                size_t element_shape_dim,
                                const char *name,
                                int *ret_param_index) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_HANDLE(self);
  TIE_CHECK_HANDLE(dt);
  TIE_CHECK_PTR_NOT_NULL(ap_element_shape);
  TIE_CHECK_PTR_NOT_NULL(name);
  TIE_CHECK_RETURN_ARG(ret_param_index);
  auto *kernel = reinterpret_cast<taichi::lang::Kernel *>(self);
  auto *data_type = reinterpret_cast<taichi::lang::DataType *>(dt);
  std::vector<int> element_shape(ap_element_shape,
                                 ap_element_shape + element_shape_dim);
  *ret_param_index =
      kernel->insert_arr_param(*data_type, total_dim, element_shape, name);
  TIE_FUNCTION_BODY_END();
}

int tie_Kernel_insert_ndarray_param(TieKernelRef self,
                                    TieDataTypeRef dt,
                                    int ndim,
                                    const char *name,
                                    int needs_grad,
                                    int *ret_param_index) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_HANDLE(self);
  TIE_CHECK_HANDLE(dt);
  TIE_CHECK_PTR_NOT_NULL(name);
  TIE_CHECK_RETURN_ARG(ret_param_index);
  auto *kernel = reinterpret_cast<taichi::lang::Kernel *>(self);
  auto *data_type = reinterpret_cast<taichi::lang::DataType *>(dt);
  *ret_param_index =
      kernel->insert_ndarray_param(*data_type, ndim, name, (bool)needs_grad);
  TIE_FUNCTION_BODY_END();
}

int tie_Kernel_insert_texture_param(TieKernelRef self,
                                    int total_dim,
                                    const char *name,
                                    int *ret_param_index) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_HANDLE(self);
  TIE_CHECK_PTR_NOT_NULL(name);
  TIE_CHECK_RETURN_ARG(ret_param_index);
  auto *kernel = reinterpret_cast<taichi::lang::Kernel *>(self);
  *ret_param_index = kernel->insert_texture_param(total_dim, name);
  TIE_FUNCTION_BODY_END();
}

int tie_Kernel_insert_pointer_param(TieKernelRef self,
                                    TieDataTypeRef dt,
                                    const char *name,
                                    int *ret_param_index) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_HANDLE(self);
  TIE_CHECK_HANDLE(dt);
  TIE_CHECK_PTR_NOT_NULL(name);
  TIE_CHECK_RETURN_ARG(ret_param_index);
  auto *kernel = reinterpret_cast<taichi::lang::Kernel *>(self);
  auto *data_type = reinterpret_cast<taichi::lang::DataType *>(dt);
  *ret_param_index = kernel->insert_pointer_param(*data_type, name);
  TIE_FUNCTION_BODY_END();
}

int tie_Kernel_insert_rw_texture_param(TieKernelRef self,
                                       int total_dim,
                                       int format,
                                       const char *name,
                                       int *ret_param_index) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_HANDLE(self);
  TIE_CHECK_PTR_NOT_NULL(name);
  TIE_CHECK_RETURN_ARG(ret_param_index);
  auto *kernel = reinterpret_cast<taichi::lang::Kernel *>(self);
  *ret_param_index = kernel->insert_rw_texture_param(
      total_dim, static_cast<taichi::lang::BufferFormat>(format), name);
  TIE_FUNCTION_BODY_END();
}

int tie_Kernel_insert_ret(TieKernelRef self,
                          TieDataTypeRef dt,
                          int *ret_ret_index) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_HANDLE(self);
  TIE_CHECK_HANDLE(dt);
  TIE_CHECK_RETURN_ARG(ret_ret_index);
  auto *kernel = reinterpret_cast<taichi::lang::Kernel *>(self);
  auto *data_type = reinterpret_cast<taichi::lang::DataType *>(dt);
  *ret_ret_index = kernel->insert_ret(*data_type);
  TIE_FUNCTION_BODY_END();
}

int tie_Kernel_finalize_rets(TieKernelRef self) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_HANDLE(self);
  auto *kernel = reinterpret_cast<taichi::lang::Kernel *>(self);
  kernel->finalize_rets();
  TIE_FUNCTION_BODY_END();
}

int tie_Kernel_finalize_params(TieKernelRef self) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_HANDLE(self);
  auto *kernel = reinterpret_cast<taichi::lang::Kernel *>(self);
  kernel->finalize_params();
  TIE_FUNCTION_BODY_END();
}

int tie_Kernel_ast_builder(TieKernelRef self,
                           TieASTBuilderRef *ret_ast_builder) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_HANDLE(self);
  TIE_CHECK_RETURN_ARG(ret_ast_builder);
  auto *kernel = reinterpret_cast<taichi::lang::Kernel *>(self);
  *ret_ast_builder =
      reinterpret_cast<TieASTBuilderRef>(&kernel->context->builder());
  TIE_FUNCTION_BODY_END();
}

int tie_Kernel_no_activate(TieKernelRef self, TieSNodeRef snode) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_HANDLE(self);
  TIE_CHECK_HANDLE(snode);
  auto *kernel = reinterpret_cast<taichi::lang::Kernel *>(self);
  auto *t_snode = reinterpret_cast<taichi::lang::SNode *>(snode);
  kernel->no_activate.push_back(t_snode);
  TIE_FUNCTION_BODY_END();
}

// class Function

int tie_Function_insert_scalar_param(TieFunctionRef self,
                                     TieDataTypeRef dt,
                                     const char *name,
                                     int *ret_param_index) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_HANDLE(self);
  TIE_CHECK_HANDLE(dt);
  TIE_CHECK_PTR_NOT_NULL(name);
  TIE_CHECK_RETURN_ARG(ret_param_index);
  auto *function = reinterpret_cast<taichi::lang::Function *>(self);
  auto *data_type = reinterpret_cast<taichi::lang::DataType *>(dt);
  *ret_param_index = function->insert_scalar_param(*data_type, name);
  TIE_FUNCTION_BODY_END();
}

int tie_Function_insert_arr_param(TieFunctionRef self,
                                  TieDataTypeRef dt,
                                  int total_dim,
                                  int *ap_element_shape,
                                  size_t element_shape_dim,
                                  const char *name,
                                  int *ret_param_index) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_HANDLE(self);
  TIE_CHECK_HANDLE(dt);
  TIE_CHECK_PTR_NOT_NULL(ap_element_shape);
  TIE_CHECK_PTR_NOT_NULL(name);
  TIE_CHECK_RETURN_ARG(ret_param_index);
  auto *function = reinterpret_cast<taichi::lang::Function *>(self);
  auto *data_type = reinterpret_cast<taichi::lang::DataType *>(dt);
  std::vector<int> element_shape(ap_element_shape,
                                 ap_element_shape + element_shape_dim);
  *ret_param_index =
      function->insert_arr_param(*data_type, total_dim, element_shape, name);
  TIE_FUNCTION_BODY_END();
}

int tie_Function_insert_ndarray_param(TieFunctionRef self,
                                      TieDataTypeRef dt,
                                      int ndim,
                                      const char *name,
                                      int needs_grad,
                                      int *ret_param_index) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_HANDLE(self);
  TIE_CHECK_HANDLE(dt);
  TIE_CHECK_PTR_NOT_NULL(name);
  TIE_CHECK_RETURN_ARG(ret_param_index);
  auto *function = reinterpret_cast<taichi::lang::Function *>(self);
  auto *data_type = reinterpret_cast<taichi::lang::DataType *>(dt);
  *ret_param_index =
      function->insert_ndarray_param(*data_type, ndim, name, (bool)needs_grad);
  TIE_FUNCTION_BODY_END();
}

int tie_Function_insert_texture_param(TieFunctionRef self,
                                      int total_dim,
                                      const char *name,
                                      int *ret_param_index) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_HANDLE(self);
  TIE_CHECK_PTR_NOT_NULL(name);
  TIE_CHECK_RETURN_ARG(ret_param_index);
  auto *function = reinterpret_cast<taichi::lang::Function *>(self);
  *ret_param_index = function->insert_texture_param(total_dim, name);
  TIE_FUNCTION_BODY_END();
}

int tie_Function_insert_pointer_param(TieFunctionRef self,
                                      TieDataTypeRef dt,
                                      const char *name,
                                      int *ret_param_index) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_HANDLE(self);
  TIE_CHECK_HANDLE(dt);
  TIE_CHECK_PTR_NOT_NULL(name);
  TIE_CHECK_RETURN_ARG(ret_param_index);
  auto *function = reinterpret_cast<taichi::lang::Function *>(self);
  auto *data_type = reinterpret_cast<taichi::lang::DataType *>(dt);
  *ret_param_index = function->insert_pointer_param(*data_type, name);
  TIE_FUNCTION_BODY_END();
}

int tie_Function_insert_rw_texture_param(TieFunctionRef self,
                                         int total_dim,
                                         int format,
                                         const char *name,
                                         int *ret_param_index) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_HANDLE(self);
  TIE_CHECK_PTR_NOT_NULL(name);
  TIE_CHECK_RETURN_ARG(ret_param_index);
  auto *function = reinterpret_cast<taichi::lang::Function *>(self);
  *ret_param_index = function->insert_rw_texture_param(
      total_dim, static_cast<taichi::lang::BufferFormat>(format), name);
  TIE_FUNCTION_BODY_END();
}

int tie_Function_set_function_body(TieFunctionRef self, TieCallback func) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_HANDLE(self);
  TIE_CHECK_PTR_NOT_NULL(func);
  auto *function = reinterpret_cast<taichi::lang::Function *>(self);
  function->set_function_body(TIE_MAKE_CALLBACK(func));
  TIE_FUNCTION_BODY_END();
}

int tie_Function_insert_ret(TieFunctionRef self,
                            TieDataTypeRef dt,
                            int *ret_ret_index) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_HANDLE(self);
  TIE_CHECK_HANDLE(dt);
  TIE_CHECK_RETURN_ARG(ret_ret_index);
  auto *function = reinterpret_cast<taichi::lang::Function *>(self);
  auto *data_type = reinterpret_cast<taichi::lang::DataType *>(dt);
  *ret_ret_index = function->insert_ret(*data_type);
  TIE_FUNCTION_BODY_END();
}

int tie_Function_finalize_rets(TieFunctionRef self) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_HANDLE(self);
  auto *function = reinterpret_cast<taichi::lang::Function *>(self);
  function->finalize_rets();
  TIE_FUNCTION_BODY_END();
}

int tie_Function_finalize_params(TieFunctionRef self) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_HANDLE(self);
  auto *function = reinterpret_cast<taichi::lang::Function *>(self);
  function->finalize_params();
  TIE_FUNCTION_BODY_END();
}

int tie_Function_ast_builder(TieFunctionRef self,
                             TieASTBuilderRef *ret_ast_builder) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_HANDLE(self);
  TIE_CHECK_RETURN_ARG(ret_ast_builder);
  auto *function = reinterpret_cast<taichi::lang::Function *>(self);
  *ret_ast_builder =
      reinterpret_cast<TieASTBuilderRef>(&function->context->builder());
  TIE_FUNCTION_BODY_END();
}

// class LaunchContextBuilder

int tie_LaunchContextBuilder_create(TieKernelRef kernel_handle,
                                    TieLaunchContextBuilderHandle *ret_handle) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_RETURN_ARG(ret_handle);
  TIE_CHECK_HANDLE(kernel_handle);
  auto *kernel = reinterpret_cast<taichi::lang::Kernel *>(kernel_handle);
  auto *builder = new taichi::lang::LaunchContextBuilder(kernel);
  *ret_handle = reinterpret_cast<TieLaunchContextBuilderHandle>(builder);
  TIE_FUNCTION_BODY_END();
}

int tie_LaunchContextBuilder_destroy(TieLaunchContextBuilderRef handle) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_HANDLE(handle);
  auto *builder =
      reinterpret_cast<taichi::lang::LaunchContextBuilder *>(handle);
  delete builder;
  TIE_FUNCTION_BODY_END();
}

int tie_LaunchContextBuilder_set_arg_int(TieLaunchContextBuilderRef handle,
                                         int arg_id,
                                         int64_t i64) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_HANDLE(handle);
  auto *builder =
      reinterpret_cast<taichi::lang::LaunchContextBuilder *>(handle);
  builder->set_arg_int(arg_id, i64);
  TIE_FUNCTION_BODY_END();
}

int tie_LaunchContextBuilder_set_arg_uint(TieLaunchContextBuilderRef handle,
                                          int arg_id,
                                          uint64_t u64) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_HANDLE(handle);
  auto *builder =
      reinterpret_cast<taichi::lang::LaunchContextBuilder *>(handle);
  builder->set_arg_uint(arg_id, u64);
  TIE_FUNCTION_BODY_END();
}

int tie_LaunchContextBuilder_set_arg_float(TieLaunchContextBuilderRef handle,
                                           int arg_id,
                                           double d) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_HANDLE(handle);
  auto *builder =
      reinterpret_cast<taichi::lang::LaunchContextBuilder *>(handle);
  builder->set_arg_float(arg_id, d);
  TIE_FUNCTION_BODY_END();
}

int tie_LaunchContextBuilder_set_struct_arg_int(
    TieLaunchContextBuilderRef handle,
    int *arg_indices,
    size_t arg_indices_dim,
    int64_t i64) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_HANDLE(handle);
  TIE_CHECK_PTR_NOT_NULL(arg_indices);
  auto *builder =
      reinterpret_cast<taichi::lang::LaunchContextBuilder *>(handle);
  std::vector<int> indices(arg_indices, arg_indices + arg_indices_dim);
  builder->set_struct_arg(indices, i64);
  TIE_FUNCTION_BODY_END();
}

int tie_LaunchContextBuilder_set_struct_arg_uint(
    TieLaunchContextBuilderRef handle,
    int *arg_indices,
    size_t arg_indices_dim,
    uint64_t u64) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_HANDLE(handle);
  TIE_CHECK_PTR_NOT_NULL(arg_indices);
  auto *builder =
      reinterpret_cast<taichi::lang::LaunchContextBuilder *>(handle);
  std::vector<int> indices(arg_indices, arg_indices + arg_indices_dim);
  builder->set_struct_arg(indices, u64);
  TIE_FUNCTION_BODY_END();
}

int tie_LaunchContextBuilder_set_struct_arg_float(
    TieLaunchContextBuilderRef handle,
    int *arg_indices,
    size_t arg_indices_dim,
    double d) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_HANDLE(handle);
  TIE_CHECK_PTR_NOT_NULL(arg_indices);
  auto *builder =
      reinterpret_cast<taichi::lang::LaunchContextBuilder *>(handle);
  std::vector<int> indices(arg_indices, arg_indices + arg_indices_dim);
  builder->set_struct_arg(indices, d);
  TIE_FUNCTION_BODY_END();
}

int tie_LaunchContextBuilder_set_arg_external_array_with_shape(
    TieLaunchContextBuilderRef handle,
    int arg_id,
    uintptr_t ptr,
    uint64_t size,
    int64_t *shape,
    size_t shape_dim,
    uintptr_t grad_ptr) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_HANDLE(handle);
  TIE_CHECK_PTR_NOT_NULL(shape);
  auto *builder =
      reinterpret_cast<taichi::lang::LaunchContextBuilder *>(handle);
  std::vector<int64_t> shape_vec(shape, shape + shape_dim);
  builder->set_arg_external_array_with_shape(arg_id, ptr, size, shape_vec,
                                             grad_ptr);
  TIE_FUNCTION_BODY_END();
}

int tie_LaunchContextBuilder_set_arg_ndarray(
    TieLaunchContextBuilderRef handle,
    int arg_id,
    TieNdarrayRef arr) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_HANDLE(handle);
  TIE_CHECK_HANDLE(arr);
  auto *builder =
      reinterpret_cast<taichi::lang::LaunchContextBuilder *>(handle);
  auto *ndarray = reinterpret_cast<const taichi::lang::Ndarray *>(arr);
  builder->set_arg_ndarray(arg_id, *ndarray);
  TIE_FUNCTION_BODY_END();
}

int tie_LaunchContextBuilder_set_arg_ndarray_with_grad(
    TieLaunchContextBuilderRef handle,
    int arg_id,
    TieNdarrayRef arr,
    TieNdarrayRef arr_grad) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_HANDLE(handle);
  TIE_CHECK_HANDLE(arr);
  TIE_CHECK_HANDLE(arr_grad);
  auto *builder =
      reinterpret_cast<taichi::lang::LaunchContextBuilder *>(handle);
  auto *ndarray = reinterpret_cast<const taichi::lang::Ndarray *>(arr);
  auto *ndarray_grad =
      reinterpret_cast<const taichi::lang::Ndarray *>(arr_grad);
  builder->set_arg_ndarray_with_grad(arg_id, *ndarray, *ndarray_grad);
  TIE_FUNCTION_BODY_END();
}

int tie_LaunchContextBuilder_set_arg_texture(
    TieLaunchContextBuilderRef handle,
    int arg_id,
    TieTextureRef tex) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_HANDLE(handle);
  TIE_CHECK_HANDLE(tex);
  auto *builder =
      reinterpret_cast<taichi::lang::LaunchContextBuilder *>(handle);
  auto *texture = reinterpret_cast<const taichi::lang::Texture *>(tex);
  builder->set_arg_texture(arg_id, *texture);
  TIE_FUNCTION_BODY_END();
}

int tie_LaunchContextBuilder_set_arg_rw_texture(
    TieLaunchContextBuilderRef handle,
    int arg_id,
    TieTextureRef tex) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_HANDLE(handle);
  TIE_CHECK_HANDLE(tex);
  auto *builder =
      reinterpret_cast<taichi::lang::LaunchContextBuilder *>(handle);
  auto *texture = reinterpret_cast<const taichi::lang::Texture *>(tex);
  builder->set_arg_rw_texture(arg_id, *texture);
  TIE_FUNCTION_BODY_END();
}

int tie_LaunchContextBuilder_get_struct_ret_int(
    TieLaunchContextBuilderRef handle,
    int *index,
    size_t index_dim,
    int64_t *ret_i64) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_HANDLE(handle);
  TIE_CHECK_PTR_NOT_NULL(index);
  TIE_CHECK_RETURN_ARG(ret_i64);
  auto *builder =
      reinterpret_cast<taichi::lang::LaunchContextBuilder *>(handle);
  std::vector<int> indices(index, index + index_dim);
  *ret_i64 = builder->get_struct_ret_int(indices);
  TIE_FUNCTION_BODY_END();
}

int tie_LaunchContextBuilder_get_struct_ret_uint(
    TieLaunchContextBuilderRef handle,
    int *index,
    size_t index_dim,
    uint64_t *ret_u64) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_HANDLE(handle);
  TIE_CHECK_PTR_NOT_NULL(index);
  TIE_CHECK_RETURN_ARG(ret_u64);
  auto *builder =
      reinterpret_cast<taichi::lang::LaunchContextBuilder *>(handle);
  std::vector<int> indices(index, index + index_dim);
  *ret_u64 = builder->get_struct_ret_uint(indices);
  TIE_FUNCTION_BODY_END();
}

int tie_LaunchContextBuilder_get_struct_ret_float(
    TieLaunchContextBuilderRef handle,
    int *index,
    size_t index_dim,
    double *ret_d) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_HANDLE(handle);
  TIE_CHECK_PTR_NOT_NULL(index);
  TIE_CHECK_RETURN_ARG(ret_d);
  auto *builder =
      reinterpret_cast<taichi::lang::LaunchContextBuilder *>(handle);
  std::vector<int> indices(index, index + index_dim);
  *ret_d = builder->get_struct_ret_float(indices);
  TIE_FUNCTION_BODY_END();
}

// class Program

int tie_Program_create(TieProgramHandle *ret_handle) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_RETURN_ARG(ret_handle);
  auto *program = new taichi::lang::Program();
  *ret_handle = reinterpret_cast<TieProgramHandle>(program);
  TIE_FUNCTION_BODY_END();
}

int tie_Program_destroy(TieProgramRef self) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_HANDLE(self);
  auto *program = reinterpret_cast<taichi::lang::Program *>(self);
  
  delete program;
  TIE_FUNCTION_BODY_END();
}

int tie_Program_finalize(TieProgramRef self) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_HANDLE(self);
  auto *program = reinterpret_cast<taichi::lang::Program *>(self);
  program->finalize();
  TIE_FUNCTION_BODY_END();
}

int tie_Program_synchronize(TieProgramRef self) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_HANDLE(self);
  auto *program = reinterpret_cast<taichi::lang::Program *>(self);
  program->synchronize();
  TIE_FUNCTION_BODY_END();
}

int tie_Program_config(TieProgramRef self, TieCompileConfigRef *ret_config) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_HANDLE(self);
  TIE_CHECK_RETURN_ARG(ret_config);
  auto *program = reinterpret_cast<taichi::lang::Program *>(self);
  *ret_config = (TieCompileConfigRef)&program->compile_config();
  TIE_FUNCTION_BODY_END();
}

int tie_Program_sync_kernel_profiler(TieProgramRef self) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_HANDLE(self);
  auto *program = reinterpret_cast<taichi::lang::Program *>(self);
  program->profiler->sync();
  TIE_FUNCTION_BODY_END();
}

int tie_Program_update_kernel_profiler(TieProgramRef self) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_HANDLE(self);
  auto *program = reinterpret_cast<taichi::lang::Program *>(self);
  program->profiler->update();
  TIE_FUNCTION_BODY_END();
}

int tie_Program_clear_kernel_profiler(TieProgramRef self) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_HANDLE(self);
  auto *program = reinterpret_cast<taichi::lang::Program *>(self);
  program->profiler->clear();
  TIE_FUNCTION_BODY_END();
}

int tie_Program_query_kernel_profile_info(TieProgramRef self,
                                          const char *name,
                                          int *ret_counter,
                                          double *ret_min,
                                          double *ret_max,
                                          double *ret_avg) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_HANDLE(self);
  TIE_CHECK_RETURN_ARG(ret_counter);
  TIE_CHECK_RETURN_ARG(ret_min);
  TIE_CHECK_RETURN_ARG(ret_max);
  TIE_CHECK_RETURN_ARG(ret_avg);
  auto *program = reinterpret_cast<taichi::lang::Program *>(self);
  auto result = program->query_kernel_profile_info(name);
  *ret_counter = result.counter;
  *ret_min = result.min;
  *ret_max = result.max;
  *ret_avg = result.avg;
  TIE_FUNCTION_BODY_END();
}

int tie_Program_get_num_kernel_profiler_records(TieProgramRef self,
                                                size_t *ret_size) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_HANDLE(self);
  TIE_CHECK_RETURN_ARG(ret_size);
  auto *program = reinterpret_cast<taichi::lang::Program *>(self);
  *ret_size = program->profiler->get_traced_records().size();
  TIE_FUNCTION_BODY_END();
}

int tie_Program_get_kernel_profiler_record(
    TieProgramRef self,
    size_t index,
    TieKernelProfileTracedRecordRef *ret_record) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_HANDLE(self);
  TIE_CHECK_RETURN_ARG(ret_record);
  auto *program = reinterpret_cast<taichi::lang::Program *>(self);
  const auto &records = program->profiler->get_traced_records();
  TIE_CHECK_INDEX_ARG(index, records.size());
  *ret_record = reinterpret_cast<TieKernelProfileTracedRecordRef>(
      (void *)&records[index]);
  TIE_FUNCTION_BODY_END();
}

int tie_Program_get_kernel_profiler_device_name(TieProgramRef self,
                                                TieStringHandle *ret_name) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_HANDLE(self);
  TIE_CHECK_RETURN_ARG(ret_name);
  auto *program = reinterpret_cast<taichi::lang::Program *>(self);
  return tie_String_create(program->profiler->get_device_name().c_str(),
                           ret_name);
  TIE_FUNCTION_BODY_END();
}

int tie_Program_reinit_kernel_profiler_with_metrics(TieProgramRef self,
                                                    const char **ap_metrics,
                                                    size_t metrics_dim,
                                                    bool *ret_b) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_HANDLE(self);
  TIE_CHECK_PTR_NOT_NULL(ap_metrics);
  TIE_CHECK_RETURN_ARG(ret_b);
  auto *program = reinterpret_cast<taichi::lang::Program *>(self);
  std::vector<std::string> metrics(ap_metrics, ap_metrics + metrics_dim);
  *ret_b = program->profiler->reinit_with_metrics(metrics);
  TIE_FUNCTION_BODY_END();
}

int tie_Program_kernel_profiler_total_time(TieProgramRef self,
                                           double *ret_time) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_HANDLE(self);
  TIE_CHECK_RETURN_ARG(ret_time);
  auto *program = reinterpret_cast<taichi::lang::Program *>(self);
  *ret_time = program->profiler->get_total_time();
  TIE_FUNCTION_BODY_END();
}

int tie_Program_set_kernel_profiler_toolkit(TieProgramRef self,
                                            const char *toolkit_name,
                                            bool *ret_b) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_HANDLE(self);
  TIE_CHECK_PTR_NOT_NULL(toolkit_name);
  TIE_CHECK_RETURN_ARG(ret_b);
  auto *program = reinterpret_cast<taichi::lang::Program *>(self);
  *ret_b = program->profiler->set_profiler_toolkit(toolkit_name);
  TIE_FUNCTION_BODY_END();
}

int tie_Program_timeline_clear(TieProgramRef self) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_HANDLE(self);
  taichi::Timelines::get_instance().clear();
  TIE_FUNCTION_BODY_END();
}

int tie_Program_timeline_save(TieProgramRef self, const char *fn) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_HANDLE(self);
  TIE_CHECK_PTR_NOT_NULL(fn);
  taichi::Timelines::get_instance().save(fn);
  TIE_FUNCTION_BODY_END();
}

int tie_Program_print_memory_profiler_info(TieProgramRef self) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_HANDLE(self);
  auto *program = reinterpret_cast<taichi::lang::Program *>(self);
  program->print_memory_profiler_info();
  TIE_FUNCTION_BODY_END();
}

int tie_Program_get_total_compilation_time(TieProgramRef self,
                                           double *ret_time) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_HANDLE(self);
  TIE_CHECK_RETURN_ARG(ret_time);
  auto *program = reinterpret_cast<taichi::lang::Program *>(self);
  *ret_time = program->get_total_compilation_time();
  TIE_FUNCTION_BODY_END();
}

int tie_Program_get_snode_num_dynamically_allocated(TieProgramRef self,
                                                    TieSNodeRef snode,
                                                    size_t *ret_size) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_HANDLE(self);
  TIE_CHECK_HANDLE(snode);
  TIE_CHECK_RETURN_ARG(ret_size);
  auto *program = reinterpret_cast<taichi::lang::Program *>(self);
  auto *t_snode = reinterpret_cast<taichi::lang::SNode *>(snode);
  *ret_size = program->get_snode_num_dynamically_allocated(t_snode);
  TIE_FUNCTION_BODY_END();
}

int tie_Program_materialize_runtime(TieProgramRef self) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_HANDLE(self);
  auto *program = reinterpret_cast<taichi::lang::Program *>(self);
  program->materialize_runtime();
  TIE_FUNCTION_BODY_END();
}

int tie_Program_make_aot_module_builder(TieProgramRef self,
                                        int arch,
                                        const char **ap_caps,
                                        size_t caps_count,
                                        TieAotModuleBuilderHandle *ret_handle) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_HANDLE(self);
  TIE_CHECK_PTR_NOT_NULL(ap_caps);
  TIE_CHECK_RETURN_ARG(ret_handle);
  auto *program = reinterpret_cast<taichi::lang::Program *>(self);
  std::vector<std::string> caps(ap_caps, ap_caps + caps_count);
  auto builder = program->make_aot_module_builder((taichi::Arch)arch, caps);
  *ret_handle = reinterpret_cast<TieAotModuleBuilderHandle>(builder.release());
  TIE_FUNCTION_BODY_END();
}

int tie_Program_get_snode_tree_size(TieProgramRef self, int *ret_size) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_HANDLE(self);
  TIE_CHECK_RETURN_ARG(ret_size);
  auto *program = reinterpret_cast<taichi::lang::Program *>(self);
  *ret_size = program->get_snode_tree_size();
  TIE_FUNCTION_BODY_END();
}

int tie_Program_get_snode_root(TieProgramRef self,
                               int tree_id,
                               TieSNodeRef *ret_snode) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_HANDLE(self);
  TIE_CHECK_RETURN_ARG(ret_snode);
  auto *program = reinterpret_cast<taichi::lang::Program *>(self);
  auto *snode = program->get_snode_root(tree_id);
  *ret_snode = reinterpret_cast<TieSNodeRef>(snode);
  TIE_FUNCTION_BODY_END();
}

int tie_Program_create_kernel(TieProgramRef self,
                              const char *name,
                              int autodiff_mode,
                              TieKernelRef *ret_kernel) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_HANDLE(self);
  TIE_CHECK_PTR_NOT_NULL(name);
  TIE_CHECK_RETURN_ARG(ret_kernel);
  auto *program = reinterpret_cast<taichi::lang::Program *>(self);
  auto *kernel = &program->kernel([](taichi::lang::Kernel *) {}, name,
                                  (AutodiffMode)autodiff_mode);
  *ret_kernel = reinterpret_cast<TieKernelRef>(kernel);
  TIE_FUNCTION_BODY_END();
}

int tie_Program_create_function(TieProgramRef self,
                                const char *func_name,
                                int func_id,
                                int instance_id,
                                TieFunctionRef *ret_func) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_HANDLE(self);
  TIE_CHECK_PTR_NOT_NULL(func_name);
  TIE_CHECK_RETURN_ARG(ret_func);
  auto *program = reinterpret_cast<taichi::lang::Program *>(self);
  auto *func = program->create_function(
      taichi::lang::FunctionKey{func_name, func_id, instance_id});
  *ret_func = reinterpret_cast<TieFunctionRef>(func);
  TIE_FUNCTION_BODY_END();
}

int tie_Program_create_sparse_matrix(TieProgramRef self,
                                     int n,
                                     int m,
                                     TieDataTypeRef dtype,
                                     const char *storage_format,
                                     TieSparseMatrixHandle *ret_handle) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_HANDLE(self);
  TIE_CHECK_HANDLE(dtype);
  TIE_CHECK_PTR_NOT_NULL(storage_format);
  TIE_CHECK_RETURN_ARG(ret_handle);
  auto *program = reinterpret_cast<taichi::lang::Program *>(self);
  auto *data_type = reinterpret_cast<taichi::lang::DataType *>(dtype);
  TI_ERROR_IF(!arch_is_cpu(program->compile_config().arch) &&
                  !arch_is_cuda(program->compile_config().arch),
              "SparseMatrix only supports CPU and CUDA for now.");
  if (arch_is_cpu(program->compile_config().arch)) {
    auto res =
        taichi::lang::make_sparse_matrix(n, m, *data_type, storage_format);
    *ret_handle = reinterpret_cast<TieSparseMatrixHandle>(res.release());
  } else {
    auto res = taichi::lang::make_cu_sparse_matrix(n, m, *data_type);
    *ret_handle = reinterpret_cast<TieSparseMatrixHandle>(res.release());
  }
  TIE_FUNCTION_BODY_END();
}

int tie_Program_make_sparse_matrix_from_ndarray(TieProgramRef self,
                                                TieSparseMatrixRef sm,
                                                TieNdarrayRef ndarray) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_HANDLE(self);
  TIE_CHECK_HANDLE(sm);
  TIE_CHECK_HANDLE(ndarray);
  auto *program = reinterpret_cast<taichi::lang::Program *>(self);
  auto *sparse_matrix = reinterpret_cast<taichi::lang::SparseMatrix *>(sm);
  auto *t_ndarray = reinterpret_cast<taichi::lang::Ndarray *>(ndarray);
  TI_ERROR_IF(!arch_is_cpu(program->compile_config().arch) &&
                  !arch_is_cuda(program->compile_config().arch),
              "SparseMatrix only supports CPU and CUDA for now.");
  taichi::lang::make_sparse_matrix_from_ndarray(program, *sparse_matrix,
                                                *t_ndarray);
  TIE_FUNCTION_BODY_END();
}

int tie_Program_create_ndarray(TieProgramRef self,
                               TieDataTypeRef dt,
                               int *ap_shape,
                               size_t shape_dim,
                               int external_array_layout,
                               bool zero_fill,
                               TieNdarrayRef *ret_handle) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_HANDLE(self);
  TIE_CHECK_HANDLE(dt);
  TIE_CHECK_PTR_NOT_NULL(ap_shape);
  TIE_CHECK_RETURN_ARG(ret_handle);
  auto *program = reinterpret_cast<taichi::lang::Program *>(self);
  auto *data_type = reinterpret_cast<taichi::lang::DataType *>(dt);
  std::vector<int> shape(ap_shape, ap_shape + shape_dim);
  auto ptr = program->create_ndarray(
      *data_type, shape, (ExternalArrayLayout)external_array_layout, zero_fill);
  *ret_handle = reinterpret_cast<TieNdarrayRef>(ptr);
  TIE_FUNCTION_BODY_END();
}

int tie_Program_delete_ndarray(TieProgramRef self, TieNdarrayRef ndarray) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_HANDLE(self);
  TIE_CHECK_HANDLE(ndarray);
  auto *program = reinterpret_cast<taichi::lang::Program *>(self);
  auto *t_ndarray = reinterpret_cast<taichi::lang::Ndarray *>(ndarray);
  program->delete_ndarray(t_ndarray);
  TIE_FUNCTION_BODY_END();
}

int tie_Program_create_texture(TieProgramRef self,
                               int fmt,
                               int *ap_shape,
                               size_t shape_dim,
                               TieTextureRef *ret_handle) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_HANDLE(self);
  TIE_CHECK_PTR_NOT_NULL(ap_shape);
  TIE_CHECK_RETURN_ARG(ret_handle);
  auto *program = reinterpret_cast<taichi::lang::Program *>(self);
  std::vector<int> shape(ap_shape, ap_shape + shape_dim);
  auto ptr = program->create_texture((taichi::lang::BufferFormat)fmt, shape);
  *ret_handle = reinterpret_cast<TieTextureRef>(ptr);
  TIE_FUNCTION_BODY_END();
}

int tie_Program_fill_ndarray_float(TieProgramRef self,
                                   TieNdarrayRef ndarray,
                                   float f) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_HANDLE(self);
  TIE_CHECK_HANDLE(ndarray);
  auto *program = reinterpret_cast<taichi::lang::Program *>(self);
  auto *t_ndarray = reinterpret_cast<taichi::lang::Ndarray *>(ndarray);
  program->fill_ndarray_fast_u32(t_ndarray, reinterpret_cast<uint32_t &>(f));
  TIE_FUNCTION_BODY_END();
}

int tie_Program_fill_ndarray_int(TieProgramRef self,
                                 TieNdarrayRef ndarray,
                                 int32_t i) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_HANDLE(self);
  TIE_CHECK_HANDLE(ndarray);
  auto *program = reinterpret_cast<taichi::lang::Program *>(self);
  auto *t_ndarray = reinterpret_cast<taichi::lang::Ndarray *>(ndarray);
  program->fill_ndarray_fast_u32(t_ndarray, reinterpret_cast<int32_t &>(i));
  TIE_FUNCTION_BODY_END();
}

int tie_Program_fill_ndarray_uint(TieProgramRef self,
                                  TieNdarrayRef ndarray,
                                  uint32_t u) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_HANDLE(self);
  TIE_CHECK_HANDLE(ndarray);
  auto *program = reinterpret_cast<taichi::lang::Program *>(self);
  auto *t_ndarray = reinterpret_cast<taichi::lang::Ndarray *>(ndarray);
  program->fill_ndarray_fast_u32(t_ndarray, u);
  TIE_FUNCTION_BODY_END();
}

int tie_Program_compile_kernel(TieProgramRef self,
                               TieCompileConfigRef compile_config,
                               TieKernelRef kernel,
                               TieCompiledKernelDataRef *ret_ckd) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_HANDLE(self);
  TIE_CHECK_HANDLE(compile_config);
  TIE_CHECK_HANDLE(kernel);
  TIE_CHECK_RETURN_ARG(ret_ckd);
  auto *program = reinterpret_cast<taichi::lang::Program *>(self);
  auto *t_compile_config =
      reinterpret_cast<taichi::lang::CompileConfig *>(compile_config);
  auto *t_kernel = reinterpret_cast<taichi::lang::Kernel *>(kernel);
  const auto *ckd = &program->compile_kernel(
      *t_compile_config, program->get_device_caps(), *t_kernel);
  *ret_ckd = reinterpret_cast<TieCompiledKernelDataRef>((void *)ckd);
  TIE_FUNCTION_BODY_END();
}

int tie_Program_launch_kernel(TieProgramRef self,
                              TieCompiledKernelDataRef kernel_data,
                              TieLaunchContextBuilderRef ctx) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_HANDLE(self);
  TIE_CHECK_HANDLE(kernel_data);
  TIE_CHECK_HANDLE(ctx);
  auto *program = reinterpret_cast<taichi::lang::Program *>(self);
  auto *t_kernel_data =
      reinterpret_cast<taichi::lang::CompiledKernelData *>(kernel_data);
  auto *t_ctx = reinterpret_cast<taichi::lang::LaunchContextBuilder *>(ctx);
  program->launch_kernel(*t_kernel_data, *t_ctx);
  TIE_FUNCTION_BODY_END();
}

// struct KernelProfileTracedRecord
int tie_KernelProfileTracedRecord_get_register_per_thread(
    TieKernelProfileTracedRecordRef self,
    int *ret_register_per_thread) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_HANDLE(self);
  TIE_CHECK_RETURN_ARG(ret_register_per_thread);
  auto *record =
      reinterpret_cast<taichi::lang::KernelProfileTracedRecord *>(self);
  *ret_register_per_thread = record->register_per_thread;
  TIE_FUNCTION_BODY_END();
}

int tie_KernelProfileTracedRecord_get_shared_mem_per_block(
    TieKernelProfileTracedRecordRef self,
    int *ret_shared_mem_per_block) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_HANDLE(self);
  TIE_CHECK_RETURN_ARG(ret_shared_mem_per_block);
  auto *record =
      reinterpret_cast<taichi::lang::KernelProfileTracedRecord *>(self);
  *ret_shared_mem_per_block = record->shared_mem_per_block;
  TIE_FUNCTION_BODY_END();
}

int tie_KernelProfileTracedRecord_get_grid_size(
    TieKernelProfileTracedRecordRef self,
    int *ret_grid_size) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_HANDLE(self);
  TIE_CHECK_RETURN_ARG(ret_grid_size);
  auto *record =
      reinterpret_cast<taichi::lang::KernelProfileTracedRecord *>(self);
  *ret_grid_size = record->grid_size;
  TIE_FUNCTION_BODY_END();
}

int tie_KernelProfileTracedRecord_get_block_size(
    TieKernelProfileTracedRecordRef self,
    int *ret_block_size) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_HANDLE(self);
  TIE_CHECK_RETURN_ARG(ret_block_size);
  auto *record =
      reinterpret_cast<taichi::lang::KernelProfileTracedRecord *>(self);
  *ret_block_size = record->block_size;
  TIE_FUNCTION_BODY_END();
}

int tie_KernelProfileTracedRecord_get_active_blocks_per_multiprocessor(
    TieKernelProfileTracedRecordRef self,
    int *ret_active_blocks_per_multiprocessor) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_HANDLE(self);
  TIE_CHECK_RETURN_ARG(ret_active_blocks_per_multiprocessor);
  auto *record =
      reinterpret_cast<taichi::lang::KernelProfileTracedRecord *>(self);
  *ret_active_blocks_per_multiprocessor =
      record->active_blocks_per_multiprocessor;
  TIE_FUNCTION_BODY_END();
}

int tie_KernelProfileTracedRecord_get_kernel_elapsed_time_in_ms(
    TieKernelProfileTracedRecordRef self,
    float *ret_kernel_elapsed_time_in_ms) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_HANDLE(self);
  TIE_CHECK_RETURN_ARG(ret_kernel_elapsed_time_in_ms);
  auto *record =
      reinterpret_cast<taichi::lang::KernelProfileTracedRecord *>(self);
  *ret_kernel_elapsed_time_in_ms = record->kernel_elapsed_time_in_ms;
  TIE_FUNCTION_BODY_END();
}

int tie_KernelProfileTracedRecord_get_time_since_base(
    TieKernelProfileTracedRecordRef self,
    float *ret_time_since_base) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_HANDLE(self);
  TIE_CHECK_RETURN_ARG(ret_time_since_base);
  auto *record =
      reinterpret_cast<taichi::lang::KernelProfileTracedRecord *>(self);
  *ret_time_since_base = record->time_since_base;
  TIE_FUNCTION_BODY_END();
}

int tie_KernelProfileTracedRecord_get_name(TieKernelProfileTracedRecordRef self,
                                           const char **ret_name) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_HANDLE(self);
  TIE_CHECK_RETURN_ARG(ret_name);
  auto *record =
      reinterpret_cast<taichi::lang::KernelProfileTracedRecord *>(self);
  *ret_name = record->name.c_str();
  TIE_FUNCTION_BODY_END();
}

int tie_KernelProfileTracedRecord_get_num_metric_values(
    TieKernelProfileTracedRecordRef self,
    size_t *ret_size) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_HANDLE(self);
  TIE_CHECK_RETURN_ARG(ret_size);
  auto *record =
      reinterpret_cast<taichi::lang::KernelProfileTracedRecord *>(self);
  *ret_size = record->metric_values.size();
  TIE_FUNCTION_BODY_END();
}

int tie_KernelProfileTracedRecord_get_metric_value(
    TieKernelProfileTracedRecordRef self,
    size_t index,
    float *ret_value) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_HANDLE(self);
  TIE_CHECK_RETURN_ARG(ret_value);
  auto *record =
      reinterpret_cast<taichi::lang::KernelProfileTracedRecord *>(self);
  TIE_CHECK_INDEX_ARG(index, record->metric_values.size());
  *ret_value = record->metric_values[index];
  TIE_FUNCTION_BODY_END();
}

// util functions (for Python)
#if defined(TI_WITH_PYTHON)

#include <Python.h>

namespace {

constexpr const char kTiePyTpFinalAttrName[] = "_tie_api_tp_finalize";
using TieDestroyFunc = void (*)(TieHandle);

static void tie_api_pytype_tp_finalize(PyObject *self) {
  PyObject *error_type, *error_value, *error_traceback;

  // Save the current exception, if any.
  PyErr_Fetch(&error_type, &error_value, &error_traceback);

  // Call tie_XXX_destroy(self)
  /// if self._manage_handle and self._handle:
  ///     self.__class__._tie_api_tp_finalize(self)
  PyObject *manage_handle_ = PyObject_GetAttrString(self, "_manage_handle");
  if (manage_handle_ && PyObject_IsTrue(manage_handle_)) {
    PyObject *handle_ = PyObject_GetAttrString(self, "_handle");
    PyObject *destroy_ = PyObject_GetAttrString((PyObject *)Py_TYPE(self),
                                                kTiePyTpFinalAttrName);
    if (handle_ && destroy_) {
      TieHandle handle = PyLong_AsVoidPtr(handle_);
      TieDestroyFunc destroy = (TieDestroyFunc)PyLong_AsVoidPtr(destroy_);
      if (handle && destroy) {
        destroy(handle);
      }
    }
    Py_XDECREF(handle_);
    Py_XDECREF(destroy_);
  }
  Py_XDECREF(manage_handle_);

  // Restore the saved exception.
  PyErr_Restore(error_type, error_value, error_traceback);
}

}  // namespace

// See https://docs.python.org/3/c-api/typeobj.html#c.PyTypeObject.tp_finalize
int tie_G_set_pytype_tp_finalize(void *py_type_object) {
  TIE_FUNCTION_BODY_BEGIN();
  TIE_CHECK_PTR_NOT_NULL(py_type_object);
  PyTypeObject *type_object = (PyTypeObject *)py_type_object;
  type_object->tp_flags |= Py_TPFLAGS_HAVE_FINALIZE;
  type_object->tp_finalize = tie_api_pytype_tp_finalize;
  TIE_FUNCTION_BODY_END();
}

#endif  // TI_WITH_PYTHON
