/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#ifndef GAME_H
#define GAME_H

#include "Physics.h"
#include "Input.h"
#include "ShapeFactory.h"
#include "Designer.h"
#include "BodyLinker.h"

class Game {
 private:
 public:
  Input input;
  Physics physics;
  Designer designer;
  BodyLinker bodyLinker;
  static int frameRate, frameCount;
  static void FrameRateThreadFun(void *arg);
  bool MouseButtonEvent(int buttion, int action);
  bool MouseWheelEvent(int delta);
  bool KeyEvent(int key, int action);
  void Test1();
  void Test2();
  void Test3();
  void Test4();
  void Test5();
  void Test6();
  void Test7();
  void Initialize();
  void Run();
};

void KeyEventCallback(int key, int action);
void MouseButtonCallback(int button, int action);
void MousePosCallback(int x, int y);
void MouseWheelCallback(int pos);

extern Game game;

#endif