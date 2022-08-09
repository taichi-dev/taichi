#include "taichi/common/logging.h"

#include "c_api/src/gui_utils/gui_utils.h"

TiGui ti_create_gui() {
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
  // return the key to img_info as handle
}

int ti_set_circle_info(TiGui gui,
                       TiMemory memory,
                       TiDataType dtype,
                       TiArch source_arch,
                       int *shape,
                       int num_axes) {
  TI_NOT_IMPLEMENTED;

  // return the key to circle_info as handle
}

int ti_render_circle(TiGui gui, int circle_renderable_handle) {
  TI_NOT_IMPLEMENTED;
}

int ti_render_image(TiGui gui, int image_renderable_handle) {
  TI_NOT_IMPLEMENTED;
}
