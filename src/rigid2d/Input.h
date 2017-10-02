/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#ifndef INPUT_H
#define INPUT_H

#include "Constants.h"
#include "Geometry.h"

class Mouse {
  friend class Input;

 private:
  int clickCount[256];
  bool pressed[256];

  void KeyDown(int key) {
    pressed[key] = true;
    clickCount[key]++;
  }

  void KeyUp(int key) { pressed[key] = false; }

 public:
  Matrix3x3 transform;
  Vector2D position;
  int x, y;

  void SetPos(int x, int y) {
    this->x = x;
    this->y = y;
    position = transform(Vector2D(x, y, 1));
  }

  Mouse() {
    memset(clickCount, 0, sizeof(clickCount));
    memset(pressed, 0, sizeof(pressed));
    transform[0][0] = transform[1][1] = transform[2][2] = 1.0;
  }

  void Set(int code, bool state) {
    if (pressed[code] != state) {
      if (!state)
        KeyUp(code);
      else
        KeyDown(code);
    }
  }

  bool IsPressed(int key) { return pressed[key]; }

  bool NeedProcess(int key) {
    if (clickCount[key]) {
      clickCount[key]--;
      return true;
    }
    return false;
  }
};

struct Keyboard {
  friend class Input;

 private:
  bool pressed[512];
  int clickCount[512];

  void KeyDown(int key) {
    pressed[key] = true;
    clickCount[key]++;
  }

  void KeyUp(int key) { pressed[key] = false; }

 public:
  void Set(int code, bool state) {
    if (pressed[code] != state) {
      if (!state)
        KeyUp(code);
      else
        KeyDown(code);
    }
  }

  Keyboard() {
    memset(pressed, 0, sizeof(pressed));
    memset(clickCount, 0, sizeof(clickCount));
  }

  bool NeedProcess(int key) {
    if (clickCount[key]) {
      clickCount[key]--;
      return true;
    }
    return false;
  }

  bool IsPressed(int key) { return pressed[key]; }
};

class Input {
 public:
  Mouse mouse;
  Keyboard keyboard;

  void Update();
};

#endif
