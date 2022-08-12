#include "taichi/taichi_core.h"

// handle.gui
typedef struct TiGui_t *TiGui;

TI_DLL_EXPORT TiGui TI_API_CALL ti_create_gui(TiArch arch,
                                              const char *shader_path,
                                              int window_h,
                                              int window_w,
                                              bool is_packed_mode);

TI_DLL_EXPORT void TI_API_CALL ti_destroy_gui(TiGui gui);

TI_DLL_EXPORT int TI_API_CALL ti_set_image_info(TiGui gui,
                                                TiMemory memory,
                                                TiDataType dtype,
                                                TiArch source_arch,
                                                int *shape,
                                                int num_axes);

TI_DLL_EXPORT int TI_API_CALL ti_set_circle_info(TiGui gui,
                                                 TiMemory memory,
                                                 TiDataType dtype,
                                                 TiArch source_arch,
                                                 int *shape,
                                                 int num_axes);

TI_DLL_EXPORT int TI_API_CALL ti_render_circle(TiGui gui,
                                               int circle_renderable_handle);

TI_DLL_EXPORT int TI_API_CALL ti_render_image(TiGui gui,
                                              int image_renderable_handle);
