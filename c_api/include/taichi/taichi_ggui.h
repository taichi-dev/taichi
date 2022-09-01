#pragma once
#include <taichi/taichi_core.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// handle.window
typedef struct TiWindowExt_t *TiWindowExt;

// handle.canvas
typedef struct TiCanvasExt_t *TiCanvasExt;

// enumeration.orientation
typedef enum TiOrientationExt {
  TI_ORIENTATION_PORTRAIT_EXT = 0,
  TI_ORIENTATION_LANDSCAPE_EXT = 1,
  TI_ORIENTATION_MAX_ENUM_EXT = 0xffffffff,
} TiOrientationExt;

// structure.window_create_info
typedef struct TiWindowCreateInfoExt {
  const char *title;
  uint32_t width;
  uint32_t height;
  TiOrientationExt orientation;
} TiWindowCreateInfoExt;

// structure.window_state
typedef struct TiWindowStateExt {
  const float *cursor_position;
  const float *wheel_delta;
  const uint32_t *pressed_keys;
  const uint32_t *released_keys;
  const uint32_t *down_keys;
} TiWindowStateExt;

// structure.matrix4x4
typedef struct TiMatrix4X4Ext {
  float components[16];
} TiMatrix4X4Ext;

// enumeration.polygon_mode
typedef enum TiPolygonModeExt {
  TI_POLYGON_MODE_TRIANGLES_EXT = 0,
  TI_POLYGON_MODE_WIREFRAME_EXT = 1,
  TI_POLYGON_MODE_LINES_EXT = 2,
  TI_POLYGON_MODE_POINTS_EXT = 3,
  TI_POLYGON_MODE_MAX_ENUM_EXT = 0xffffffff,
} TiPolygonModeExt;

// structure.point_light
typedef struct TiPointLightExt {
  float color[3];
  float radius;
} TiPointLightExt;

// structure.scene_light
typedef struct TiSceneLightExt {
  float ambient_color[3];
  uint32_t point_light_count;
  TiPointLightExt point_lights[3];
} TiSceneLightExt;

// structure.scene_object
typedef struct TiSceneObjectExt {
  TiPolygonModeExt polygon_mode;
  TiNdArray indices;
  TiNdArray positions;
  TiNdArray texcoords;
  TiNdArray normals;
} TiSceneObjectExt;

// function.create_window
TI_DLL_EXPORT TiWindowExt TI_API_CALL
ti_create_window_ext(TiRuntime runtime, TiWindowCreateInfoExt create_info);

// function.destroy_window
TI_DLL_EXPORT void TI_API_CALL ti_destroy_window_ext(TiWindowExt window);

// function.get_window_state
TI_DLL_EXPORT void TI_API_CALL
ti_get_window_state_ext(TiWindowExt window, TiWindowStateExt *window_state);

// function.begin_frame
TI_DLL_EXPORT TiCanvasExt TI_API_CALL ti_begin_frame_ext(TiWindowExt window);

// function.end_frame
TI_DLL_EXPORT TiCanvasExt TI_API_CALL ti_end_frame_ext();

// function.draw_scene
TI_DLL_EXPORT void TI_API_CALL
ti_draw_scene_ext(TiCanvasExt canvas,
                  TiMatrix4X4Ext camera_matrix,
                  uint32_t scene_light_count,
                  const TiSceneLightExt *scene_light,
                  uint32_t scene_object_count,
                  const TiSceneObjectExt *scene_objects);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
