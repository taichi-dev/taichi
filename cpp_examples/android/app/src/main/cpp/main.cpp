/*
 * Copyright (C) 2010 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

// BEGIN_INCLUDE(all)
#include <initializer_list>
#include <memory>
#include <cstdlib>
#include <cstring>
#include <jni.h>
#include <errno.h>
#include <cassert>

#include "taichi/backends/vulkan/vulkan_program.h"
#include "taichi/backends/vulkan/vulkan_common.h"
#include "taichi/backends/vulkan/vulkan_loader.h"
#include "taichi/backends/vulkan/runtime.h"
#include "taichi/backends/vulkan/aot_module_loader_impl.h"

#include "taichi/ui/backends/vulkan/app_context.h"
#include "taichi/ui/backends/vulkan/gui.h"
#include "taichi/ui/backends/vulkan/renderer.h"

#include <android/log.h>
#include <android_native_app_glue.h>

#define LOGI(...) \
  ((void)__android_log_print(ANDROID_LOG_INFO, "native-activity", __VA_ARGS__))
#define LOGW(...) \
  ((void)__android_log_print(ANDROID_LOG_WARN, "native-activity", __VA_ARGS__))
#define LOGE(...) \
  ((void)__android_log_print(ANDROID_LOG_ERROR, "native-activity", __VA_ARGS__))

#define NR_PARTICLES 8192

struct engine {
  struct android_app *app;

  std::unique_ptr<taichi::ui::vulkan::Renderer> renderer;
  std::unique_ptr<taichi::lang::MemoryPool> memory_pool;
  std::unique_ptr<taichi::lang::vulkan::VkRuntime> vulkan_runtime;
  std::unique_ptr<taichi::lang::aot::Module> module;
  std::shared_ptr<taichi::ui::vulkan::Gui> gui;
  taichi::lang::RuntimeContext host_ctx;

  taichi::lang::aot::Kernel *init_kernel;
  taichi::lang::aot::Kernel *substep_kernel;
  taichi::lang::DeviceAllocation dev_alloc_pos;
  taichi::ui::CirclesInfo circles;

  bool init{false};
};

static jobject getGlobalContext(JNIEnv *env) {
  // Get the instance object of Activity Thread
  jclass activityThread = env->FindClass("android/app/ActivityThread");
  jmethodID currentActivityThread =
      env->GetStaticMethodID(activityThread, "currentActivityThread",
                             "()Landroid/app/ActivityThread;");
  jobject at =
      env->CallStaticObjectMethod(activityThread, currentActivityThread);
  // Get Application, which is the global Context
  jmethodID getApplication = env->GetMethodID(activityThread, "getApplication",
                                              "()Landroid/app/Application;");
  jobject context = env->CallObjectMethod(at, getApplication);
  return context;
}

static std::string GetExternalFilesDir(struct engine *engine) {
  std::string ret;

  engine->app->activity->vm->AttachCurrentThread(&engine->app->activity->env,
                                                 NULL);
  JNIEnv *env = engine->app->activity->env;

  // getExternalFilesDir() - java
  jclass cls_Env = env->FindClass("android/app/NativeActivity");
  jmethodID mid = env->GetMethodID(cls_Env, "getExternalFilesDir",
                                   "(Ljava/lang/String;)Ljava/io/File;");
  jobject obj_File = env->CallObjectMethod(getGlobalContext(env), mid, NULL);
  jclass cls_File = env->FindClass("java/io/File");
  jmethodID mid_getPath =
      env->GetMethodID(cls_File, "getPath", "()Ljava/lang/String;");
  jstring obj_Path = (jstring)env->CallObjectMethod(obj_File, mid_getPath);

  ret = env->GetStringUTFChars(obj_Path, NULL);

  engine->app->activity->vm->DetachCurrentThread();

  return ret;
}

static void copyAssetsToDataDir(struct engine *engine, const char *folder) {
  const char *filename;
  auto dir = AAssetManager_openDir(engine->app->activity->assetManager, folder);
  std::string out_dir = GetExternalFilesDir(engine) + "/" + folder;
  std::filesystem::create_directories(out_dir);

  while ((filename = AAssetDir_getNextFileName(dir))) {
    std::ofstream out_file(out_dir + filename, std::ios::binary);
    std::string in_filepath = std::string(folder) + filename;
    AAsset *asset =
        AAssetManager_open(engine->app->activity->assetManager,
                           in_filepath.c_str(), AASSET_MODE_UNKNOWN);
    auto in_buffer = AAsset_getBuffer(asset);
    auto size = AAsset_getLength(asset);
    out_file.write((const char *)in_buffer, size);
  }
}

static int engine_init_display(struct engine *engine) {
  // Copy the assets from the AssetManager to internal storage so we can use a
  // file system path inside Taichi.
  copyAssetsToDataDir(engine, "mpm88/");
  copyAssetsToDataDir(engine, "shaders/");

  // Create the configuration for the renderer
  taichi::ui::AppConfig app_config;
  app_config.name = "AOT Loader";
  app_config.vsync = true;
  app_config.show_window = false;
  app_config.ti_arch = taichi::Arch::vulkan;
  app_config.is_packed_mode = true;
  app_config.width = ANativeWindow_getWidth(engine->app->window);
  app_config.height = ANativeWindow_getHeight(engine->app->window);
  app_config.package_path = GetExternalFilesDir(engine);

  // Create the renderer
  engine->renderer = std::make_unique<taichi::ui::vulkan::Renderer>();
  engine->renderer->init(
      nullptr, (taichi::ui::TaichiWindow *)engine->app->window, app_config);

  taichi::uint64 *result_buffer{nullptr};
  engine->memory_pool =
      std::make_unique<taichi::lang::MemoryPool>(taichi::Arch::vulkan, nullptr);
  result_buffer = (taichi::uint64 *)engine->memory_pool->allocate(
      sizeof(taichi::uint64) * taichi_result_buffer_entries, 8);

  // Create the Runtime
  taichi::lang::vulkan::VkRuntime::Params params;
  params.host_result_buffer = result_buffer;
  params.device = &(engine->renderer->app_context().device());
  engine->vulkan_runtime =
      std::make_unique<taichi::lang::vulkan::VkRuntime>(std::move(params));

  // @FIXME: On some Phones (MTK GPU for example),
  // VkExternalMemoryImageCreateInfo doesn't seem to support external memory...
  // it returns VK_ERROR_INVALID_EXTERNAL_HANDLE
  params.device->set_cap(taichi::lang::DeviceCapability::vk_has_external_memory,
                         false);

  // Create the GUI and initialize a default background color
  engine->gui = std::make_shared<taichi::ui::vulkan::Gui>(
      &engine->renderer->app_context(), &engine->renderer->swap_chain(),
      (taichi::ui::TaichiWindow *)engine->app->window);

  engine->renderer->set_background_color({0.6, 0.6, 0.6});

  // Load the AOT module using the previously created Runtime
  taichi::lang::vulkan::AotModuleParams aotParams{
      app_config.package_path + "/mpm88/", engine->vulkan_runtime.get()};
  engine->module =
      taichi::lang::aot::Module::load(taichi::Arch::vulkan, aotParams);
  auto rootSize = engine->module->get_root_size();
  engine->vulkan_runtime->add_root_buffer(rootSize);

  // Retrieve kernels/fields/etc from AOT module so we can initialize our
  // runtime
  engine->init_kernel = engine->module->get_kernel("init");
  engine->substep_kernel = engine->module->get_kernel("substep");

  // Create a NdArray allocation that could be used by the AOT module kernel
  taichi::lang::Device::AllocParams alloc_params;
  alloc_params.size = NR_PARTICLES * 2 * sizeof(float);
  engine->dev_alloc_pos =
      engine->vulkan_runtime->get_ti_device()->allocate_memory(
          std::move(alloc_params));

  engine->host_ctx.set_arg(0, &engine->dev_alloc_pos);
  engine->host_ctx.set_device_allocation(0, true);
  engine->host_ctx.extra_args[0][0] = 1;
  // Size in term of number of float
  engine->host_ctx.extra_args[0][1] =
      taichi::ui::VboHelpers::size(taichi::ui::VboHelpers::all()) / 4;
  engine->host_ctx.extra_args[0][2] = 1;

  engine->init_kernel->launch(&engine->host_ctx);
  engine->vulkan_runtime->synchronize();

  // Create the circles configuration for rendering with Taichi GGUI
  taichi::ui::FieldInfo field_info;
  field_info.valid = true;
  field_info.field_type = taichi::ui::FieldType::Scalar;
  field_info.matrix_rows = 1;
  field_info.matrix_cols = 1;
  field_info.shape = {NR_PARTICLES};
  field_info.field_source = taichi::ui::FieldSource::TaichiVulkan;
  field_info.dtype = taichi::lang::PrimitiveType::f32;
  field_info.snode = nullptr;
  field_info.dev_alloc = engine->dev_alloc_pos;

  engine->circles.renderable_info.has_per_vertex_color = false;
  engine->circles.renderable_info.vbo = field_info;
  engine->circles.renderable_info.vbo_attrs = taichi::ui::VboHelpers::all();
  engine->circles.color = {0.8, 0.4, 0.1};
  engine->circles.radius = 0.002f;

  engine->init = true;

  return 0;
}

/**
 * Just the current frame in the display.
 */
static void engine_draw_frame(struct engine *engine) {
  if (!engine->init) {
    // No display.
    return;
  }

  // Run 'substep' 50 times
  for (int i = 0; i < 50; i++) {
    engine->substep_kernel->launch(&engine->host_ctx);
  }
  engine->vulkan_runtime->synchronize();

  // Render elements
  engine->renderer->circles(engine->circles);
  engine->renderer->draw_frame(engine->gui.get());
  engine->renderer->swap_chain().surface().present_image();
  engine->renderer->prepare_for_next_frame();
}

static void engine_term_display(struct engine *engine) {
  // @TODO: to implement
}

static int32_t engine_handle_input(struct android_app *app,
                                   AInputEvent *event) {
  // Implement input with Taichi Kernel
  return 0;
}

static void engine_handle_cmd(struct android_app *app, int32_t cmd) {
  struct engine *engine = (struct engine *)app->userData;
  switch (cmd) {
    case APP_CMD_INIT_WINDOW:
      // The window is being shown, get it ready.
      if (engine->app->window != NULL) {
        engine_init_display(engine);
        engine_draw_frame(engine);
      }
      break;
    case APP_CMD_TERM_WINDOW:
      // The window is being hidden or closed, clean it up.
      engine_term_display(engine);
      break;
  }
}

void android_main(struct android_app *state) {
  struct engine engine;

  memset(&engine, 0, sizeof(engine));
  state->userData = &engine;
  state->onAppCmd = engine_handle_cmd;
  state->onInputEvent = engine_handle_input;
  engine.app = state;

  while (1) {
    // Read all pending events.
    int ident;
    int events;
    struct android_poll_source *source;

    // If not animating, we will block forever waiting for events.
    // If animating, we loop until all events are read, then continue
    // to draw the next frame of animation.
    while ((ident = ALooper_pollAll(0, NULL, &events, (void **)&source)) >= 0) {
      // Process this event.
      if (source != NULL) {
        source->process(state, source);
      }

      // Check if we are exiting.
      if (state->destroyRequested != 0) {
        engine_term_display(&engine);
        return;
      }
    }

    engine_draw_frame(&engine);
  }
}
// END_INCLUDE(all)
