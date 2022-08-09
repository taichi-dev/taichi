#include "taichi/common/logging.h"

#include "c_api/src/gui_utils/gui_utils.h"
#include "c_api/src/gui_utils/gui_helper.h"

TiGui ti_create_gui(TiArch arch,
                    const char *shader_path,
                    int window_h,
                    int window_w,
                    bool is_packed_mode) {
  TI_NOT_IMPLEMENTED;
}

void ti_destroy_gui(TiGui gui) {
  TI_NOT_IMPLEMENTED;
}

int ti_set_image_info(TiGui gui,
                      TiMemory memory,
                      TiDataType dtype,
                      TiArch source_arch,
                      int *shape,
                      int num_axes) {
  TI_NOT_IMPLEMENTED;
}

int ti_set_circle_info(TiGui gui,
                       TiMemory memory,
                       TiDataType dtype,
                       TiArch source_arch,
                       int *shape,
                       int num_axes) {
  TI_NOT_IMPLEMENTED;
}

int ti_render_circle(TiGui gui, int circle_renderable_handle) {
  TI_NOT_IMPLEMENTED;
}

int ti_render_image(TiGui gui, int image_renderable_handle) {
  TI_NOT_IMPLEMENTED;
}
