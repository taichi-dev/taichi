/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#pragma once

#include <cmath>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>
#include <ctime>
#include <thread>
#include <map>
#ifndef _WIN64
#include <unistd.h>
#endif
#include <mutex>
#include <taichi/math/math_util.h>
#ifndef _WIN64
#include <sys/time.h>
#endif
#include <functional>

#ifdef __linux__
#include <GL/glew.h>
#elif defined(__APPLE__)
#include <GL/glew.h>
#include <OpenGL/gl3.h>
#elif defined(_WIN64)
#include <GL/glew.h>
#endif

#pragma warning(push)
#pragma warning(disable:4005)
#include <GLFW/glfw3.h>
#pragma warning(pop)

#include <taichi/common/meta.h>
#include "timer.h"

#ifndef DEBUG
//#define assert(x)
#endif

TC_NAMESPACE_BEGIN

using std::string;
using std::vector;
using std::map;

GLuint load_shader(GLint type, string filename);

GLuint load_program(string vertex_fn, string fragment_fn);

void register_mouse_button_callback(std::function<void(int, int, int)> func);

void gl_print_error_code(int code);

#define CGL {int code = glGetError(); if(code != GL_NO_ERROR) {printf("GL Error: "); gl_print_error_code(code); error("GL error detected");}};

class GLWindow;

class RenderingGuard {
public:
    RenderingGuard(GLWindow *target);

    ~RenderingGuard();
private:
    GLWindow *target;
};

class ContextGuard {
public:
    ContextGuard(GLWindow *target);

    ~ContextGuard();
private:
    GLWindow *target;
    GLFWwindow *old_context;
};

class GLWindow {
public:
    typedef std::function<void(int, int)> MouseMoveCallbackInt;
    typedef std::function<void(float, float)> MouseMoveCallbackFloat;
    typedef std::function<void(int, int, int, int)> KeyboardCallback;
    GLWindow(Config config);

    ~GLWindow();

    RenderingGuard create_rendering_guard() {
        return RenderingGuard(this);
    }

    ContextGuard create_context_guard() {
        return ContextGuard(this);
    }

    void print_actual_version();

    void begin_rendering();

    void end_rendering() {
        glfwSwapBuffers(window);
    }

    void add_keyboard_callback(KeyboardCallback callback) {
        keyboard_callbacks.push_back(callback);
    }

    void add_mouse_move_callback_int(MouseMoveCallbackInt callback) {
        mouse_move_callbacks_int.push_back(callback);
    }

    void add_mouse_move_callback_float(MouseMoveCallbackFloat callback) {
        mouse_move_callbacks_float.push_back(callback);
    }

    void wait_for_key() {
        waiting_for_key = true;
        while (waiting_for_key) {
            Time::usleep(1e3);
            glfwPollEvents();
        }
    }

    // NOTE: will reset key to false
    static bool key_pressed(int key) {
        bool p = pressed[key];
        pressed[key] = false;
        return p;
    }

    static void general_mouse_move_callback(GLFWwindow *window, double x, double y) {
        window_to_instance[window]->mouse_move_callback((float)x, (float)y);
    }

    static void general_keyboard_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
    {
        window_to_instance[window]->keyboard_callback(key, scancode, action, mods);
    }

    int get_width() const {
        return width;
    }

    int get_height() const {
        return height;
    }

    void make_current() {
        glfwMakeContextCurrent(window);
    }

    static GLWindow *get_gpgpu_window() {
        static GLWindow window(Config().set("hide", true));
        return &window;
    }

private:
    void mouse_move_callback(float x, float y) {
        y = height - y;
        for (auto &callback : mouse_move_callbacks_int) {
            callback((int)x, (int)y);
        }
        for (auto &callback : mouse_move_callbacks_float) {
            callback(x / width, y / height);
        }
    }

    void keyboard_callback(int key, int scancode, int action, int mods);

    vec3 background_color;
    int width, height;
    bool exit_on_esc;
    string title;
    GLFWwindow *window;
    static GLFWwindow *first_window;

    std::vector < KeyboardCallback> keyboard_callbacks;

    std::vector < MouseMoveCallbackInt> mouse_move_callbacks_int;
    std::vector < MouseMoveCallbackFloat> mouse_move_callbacks_float;

    static bool pressed[256];
    static bool first_window_created;
    static std::map<GLFWwindow *, GLWindow*> window_to_instance;
    bool waiting_for_key = false;
    static int x_start_position;
};

TC_NAMESPACE_END

