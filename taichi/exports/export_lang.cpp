#include "exports.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <exception>

#include "taichi/common/exceptions.h"
#include "taichi/ir/frontend_ir.h"
#include "taichi/program/extension.h"
#include "taichi/program/function.h"
#include "taichi/program/kernel.h"
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
  catch (...) {                                                              \
    tie_api_set_last_error_impl(TIE_ERROR_UNKNOWN_CXX_EXCEPTION,             \
                                "C++ Exception");                            \
    return TIE_ERROR_UNKNOWN_CXX_EXCEPTION;                                  \
  }                                                                          \
  return TIE_ERROR_SUCCESS

namespace {

class TieCallbackFailedException : public std::exception {
 public:
  explicit TieCallbackFailedException(std::string msg) : msg_(std::move(msg)) {
  }

  const char *what() const noexcept override {
    return msg_.c_str();
  }

 private:
  std::string msg_;
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
