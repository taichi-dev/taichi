#include "taichi/taichi_core.h"

// handle.gui
typedef struct TiGui_t *TiGui;

TiGui ti_create_gui(TiArch arch,
                    const char *shader_path,
                    int window_h,
                    int window_w,
                    bool is_packed_mode);

void ti_destroy_gui(TiGui gui);

int ti_set_image_info(TiGui gui,
                      TiMemory memory,
                      TiDataType dtype,
                      TiArch source_arch,
                      int *shape,
                      int num_axes);

int ti_set_circle_info(TiGui gui,
                       TiMemory memory,
                       TiDataType dtype,
                       TiArch source_arch,
                       int *shape,
                       int num_axes);

int ti_render_circle(TiGui gui, int circle_renderable_handle);

int ti_render_image(TiGui gui, int image_renderable_handle);
