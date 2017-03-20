/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#include "Game.h"

double gravity = 0;

int Game::frameCount, Game::frameRate;

void Game::FrameRateThreadFun(void *arg) {
    frameCount = 0;
    while (1) {
        //glfwSleep(1.0);
        settings.frameRate = frameCount;
        frameCount = 0;
    }
}


bool Game::MouseButtonEvent(int button, int action) {
    if (designer.MouseButtonEvent(button, action)) return true;
    if (bodyLinker.MouseButtonEvent(button, action)) return true;
    if (button == GLFW_MOUSE_BUTTON_1 && action == GLFW_PRESS) {
        Object *object = ShapeFactory::Generate();
        object->SetPosition(input.mouse.position);
        physics.AddObject(object);
        return true;
    }
    
    return false;
}

bool Game::KeyEvent(int key, int action) {
    if (designer.KeyEvent(key, action)) return true;
    if (bodyLinker.KeyEvent(key, action)) return true;
    if (ShapeFactory::KeyEvent(key, action)) return true;

    if (action == GLFW_RELEASE) return false;    
    if (key == GLFW_KEY_PAUSE) return settings.pause ^= 1, true;
    if (key == GLFW_KEY_F1) return Test1(), true;
    if (key == GLFW_KEY_F2) return Test2(), true;
    if (key == GLFW_KEY_F3) return Test3(), true;
    if (key == GLFW_KEY_F4) return Test4(), true;
    if (key == GLFW_KEY_F5) return Test5(), true;
    if (key == GLFW_KEY_F6) return Test6(), true;
    if (key == GLFW_KEY_F7) return Test7(), true;
    
    return false;
}

bool Game::MouseWheelEvent(int delta) {
    //return scoper.MouseWheelEvent(delta);
    return false;
}


void Game::Initialize() {
    srand((unsigned int)time(0));
    //graphics.Initialize();
    //glfwSetMousePosCallback(MousePosCallback);
    //glfwSetMouseButtonCallback(MouseButtonCallback);
    //glfwSetKeyCallback(KeyEventCallback);
    //glfwSetMouseWheelCallback(MouseWheelCallback);
    
    frameRate = 0;
    frameCount = 0;
    ShapeFactory::Init();
    settings.Init();
    designer.SetGame(this);
    bodyLinker.SetGame(this);
    //scoper.SetGame(this);

}

void Game::Run() {
    Initialize();
    Test1();

    Object *target = NULL; Vector2D shapeR;
    //glfwCreateThread(FrameRateThreadFun, NULL);
    for (int k = 0; ; k++) {
        input.Update();
        {
            if (input.keyboard.NeedProcess((int)'X')) {
                if (physics.objects.size() > 1)
                    while (physics.objects.size() > 1) physics.DeleteLastObject();
                else if (physics.objects.size() == 1)
                    physics.DeleteLastObject();
            }
//            if (input.keyboard.IsPressed(GLFW_LCTRL) && input.mouse.IsPressed(GLFW_MOUSE_1)) {
//                waterDrop
//                physics.AddObject();
//            }
        }
        if (input.mouse.NeedProcess(GLFW_MOUSE_BUTTON_2)) {
            physics.SelectObject(input.mouse.position, target, shapeR);
        }
        if (target != NULL) {
            if (input.mouse.IsPressed(GLFW_MOUSE_BUTTON_2)) {
                //physics.MouseAdjust(target, shapeR, input.mouse.position, &scoper);
                if (input.keyboard.NeedProcess('S')) {
                    target->SetFixed(!target->GetFixed());
                }
            } else target = NULL;
        }
        if (target != NULL) {
//            if (input.keyboard.NeedProcess('F')) target->SetFixed(!target->GetFixed());
            //if (input.keyboard.NeedProcess(GLFW_KEY_DEL) && target != physics.GetWorldBox()) {
                physics.DeleteObject(target);
                target = NULL;
            //}
        }
        
        if (!settings.pause) {
            physics.Proceed(timeInterval);
        } else exit(0);//glfwSleep(0.05);
        /*
        graphics.BeginRendering();
            physics.Redraw();
            if (target != NULL && input.mouse.IsPressed(GLFW_MOUSE_BUTTON_2)) {
                Vector2D p = target->GetTransformToWorld()(shapeR + Vector2D::Origin);
                graphics.DrawLine(p.x, p.y, input.mouse.position.x, input.mouse.position.y, HSB3f(0.0f, 0.8f, 0.8f), 5.0);
            }
            designer.Redraw();
            bodyLinker.Redraw();
        graphics.EndRendering();
        */
        frameCount++;
    }
}


Game game;

void KeyEventCallback(int key, int action) {
    //if (TwEventKeyGLFW(key, action)) return;
    if (game.KeyEvent(key, action)) return;
}

void MouseButtonCallback(int botton, int action) {
    //if (TwEventMouseButtonGLFW(botton, action)) return;
    if (game.MouseButtonEvent(botton, action)) return;
}

void MousePosCallback(int x, int y) {
    //TwEventMousePosGLFW(x, y);
}

void MouseWheelCallback(int pos) {
    static int lastPos;
    int delta = pos - lastPos;
    lastPos = pos;
    //TwEventMouseWheelGLFW(pos);
    game.MouseWheelEvent(delta);
}
