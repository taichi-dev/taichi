/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#include "Game.h"

void Game::Test1() {
  physics.SetAsDefaultWorld();

  Object *plant = ShapeFactory::GenerateBoxObject(700, 20);
  plant->SetPosition(Vector2D(400, 30, 1));
  plant->SetFixed(true);

  if (true) {
    double width = 20;
    for (int i = 0; i < 20; i++)
      for (int j = 0; j < 25; j++) {
        Object *box = ShapeFactory::GenerateBoxObject(width, width);
        box->SetPosition(
            Vector2D(100 + width * j * 1.05, 100 + width * i * 1.05, 1));
        physics.AddObject(box);
      }
  }

  physics.AddObject(plant);
}

void Game::Test2() {
  physics.SetAsDefaultWorld();
  int n = 16;
  for (int i = 0; i < n; i++) {
    double x = 100 + 36 * (i + 1);
    Circle *cir = Circle::GenerateCircle(18);
    cir->color = HSB3f(360.0f / n * i, 0.8f, 0.8f);
    cir->restitution = 1.0;
    cir->centroidPosition = Vector2D(x, 200, 1);
    Object *obj = new Object(cir);
    physics.AddObject(obj);
    DistanceConstraint *con = new DistanceConstraint(
        obj, Vector2D(0, 0, 0), physics.worldBox, Vector2D(x, 400, 0));
    con->color = cir->color;
    physics.AddConstantConstraint(con);
  }

  Object *box = ShapeFactory::GenerateBoxObject(700, 20);
  box->SetPosition(Vector2D(400, 50, 1));
  box->SetFixed(true);
  physics.AddObject(box);

  n = 32;
  for (int i = 0; i < n; i++) {
    box = ShapeFactory::GenerateBoxObject(7, 50);
    box->GetShape(0)->color = HSB3f(360.0f / n * i, 0.8f, 1.0f);
    box->GetShape(0)->friction = 0.4;
    box->SetPosition(Vector2D(100 + i * 20, 100, 1));
    physics.AddObject(box);
  }
}

void Game::Test3() {
  physics.SetAsDefaultWorld();
  const int n = 96;
  Object *boxes[n];
  for (int i = 0; i < n; i++) {
    boxes[i] = ShapeFactory::GenerateBoxObject(60, 3);
    boxes[i]->SetPosition(
        Vector2D(screenWidth / 2, screenHeight - 30 - i * 7, 1));
    boxes[i]->GetShape(0)->density = 0.1;
    boxes[i]->Update();
    boxes[i]->GetShape(0)->SetBoundaryWidth(0.0);
    boxes[i]->SetColor(HSB3f(180.0f / n * i, 0.7f, 0.8f));
    physics.AddObject(boxes[i]);
  }
  for (int i = 0; i < n - 1; i++) {
    Spring *spring0 = new Spring(boxes[i], Vector2D(-23, 0, 1), boxes[i + 1],
                                 Vector2D(-23, 0, 1), 8000, 5);
    Spring *spring1 = new Spring(boxes[i], Vector2D(23, 0, 1), boxes[i + 1],
                                 Vector2D(23, 0, 1), 8000, 5);
    spring0->color = spring1->color = HSB3f(180.0f / n * i, 0.7f, 0.8f);
    physics.AddForce(spring0);
    physics.AddForce(spring1);
  }
  boxes[0]->SetFixed(true);
}

void Game::Test4() {
  physics.SetAsDefaultWorld();
  RGB3f color0 = RGB3f::RandomBrightColor(),
        color1 = RGB3f::RandomBrightColor();
  vector<Vector2D> points(8);
  double R = 200, r = 10;
  double centerX = screenWidth / 2, centerY = screenHeight * 0.6;
  points[0] = Vector2D(centerX - R, centerY + r, 1);
  points[1] = Vector2D(centerX - R, centerY - r, 1);
  points[2] = Vector2D(centerX - r, centerY - r, 1);
  points[3] = Vector2D(centerX - r, centerY - R, 1);
  points[4] = Vector2D(centerX + r, centerY - R, 1);
  points[5] = Vector2D(centerX + r, centerY - r, 1);
  points[6] = Vector2D(centerX + R, centerY - r, 1);
  points[7] = Vector2D(centerX + R, centerY + r, 1);

  Object *mainObj = ShapeFactory::GeneratePolygonObject(points);
  mainObj->GetShape(0)->density = 10;
  mainObj->SetResistance(0.0, 0.0);
  physics.AddObject(mainObj);
  //    Vector2D p = mainObj->GetTransformToWorldInverse()(Vector2D(centerX,
  //    centerY, 1));
  DistanceConstraint *constraint = new DistanceConstraint(
      physics.worldBox, Vector2D(centerX, centerY, 0), mainObj,
      Vector2D(centerX, centerY, 1) - mainObj->GetPosition());
  physics.AddConstantConstraint(constraint);

  Object *subObj;
  subObj = ShapeFactory::GenerateBoxObject(r * 2, R * 1.5);
  subObj->SetPosition(Vector2D(centerX - R, centerY - R * 0.3, 1));
  subObj->SetLayerMask(0);
  subObj->SetColor(color1);
  subObj->SetResistance(0.0, 0.0);
  //    subObj->GetShape(0)->density = 0.3;
  physics.AddObject(subObj);
  physics.AddConstantConstraint(new DistanceConstraint(
      mainObj, Vector2D(centerX - R, centerY, 1) - mainObj->GetPosition(),
      subObj, Vector2D(centerX - R, centerY, 1) - subObj->GetPosition()));

  subObj = ShapeFactory::GenerateBoxObject(r * 2, R * 1.5);
  subObj->SetPosition(Vector2D(centerX + R, centerY - R * 0.3, 1));
  subObj->SetLayerMask(0);
  subObj->SetColor(color1);
  subObj->SetResistance(0.0, 0.0);
  subObj->GetShape(0)->density = 2.1;
  physics.AddObject(subObj);
  physics.AddConstantConstraint(new DistanceConstraint(
      mainObj, Vector2D(centerX + R, centerY, 1) - mainObj->GetPosition(),
      subObj, Vector2D(centerX + R, centerY, 1) - subObj->GetPosition()));

  subObj = ShapeFactory::GenerateBoxObject(r * 2, R * 1.5);
  subObj->SetPosition(Vector2D(centerX, centerY - R * 1.3, 1));
  subObj->SetLayerMask(0);
  subObj->SetColor(color1);
  subObj->SetResistance(0.0, 0.0);
  subObj->GetShape(0)->density = 0.7;
  physics.AddObject(subObj);
  physics.AddConstantConstraint(new DistanceConstraint(
      mainObj, Vector2D(centerX, centerY - R, 1) - mainObj->GetPosition(),
      subObj, Vector2D(centerX, centerY - R, 1) - subObj->GetPosition()));

  /*
  physics.DeleteAllShapes();

  Circle *cir = Circle::GenerateCircle(100);
  cir->angularVelocity = 2.4;
  cir->centroidPosition = Vector2D(600, 300, 1);
  physics.AddShape(cir);
  physics.AddConstantConstraint(new DistanceConstraint(cir, Vector2D(0, 0, 0),
  cir->centroidPosition));


  Polygon *box = Polygon::GenerateBox(300, 20);
  box->SetFixed(true);
  box->restitution = 0.0;
  box->friction = 0.0;
  box->centroidPosition = Vector2D(200, 282, 1);
  physics.AddShape(box);

  box = new Polygon(*box);
  box->centroidPosition = Vector2D(200, 322, 1);
  physics.AddShape(box);


  box = new Polygon(*box);
  box->SetFixed(false);
  box->centroidPosition = Vector2D(200, 300, 1);
  physics.AddShape(box);

  physics.AddConstantConstraint(new DistanceConstraint(cir, Vector2D(0, 70, 0),
  box, Vector2D(0, 0, 0)));
  */
}

void Game::Test5() {
  physics.SetAsDefaultWorld();
  int n = 16;
  for (int i = 0; i < n; i++) {
    double x = 100 + 36 * (i + 1);
    Circle *cir = Circle::GenerateCircle(18);
    cir->color = HSB3f(360.0f / n * i, 0.8f, 0.8f);
    cir->restitution = 1.0;
    cir->centroidPosition = Vector2D(x, 200, 1);
    Object *obj = new Object(cir);
    physics.AddObject(obj);
    DistanceConstraint *con = new DistanceConstraint(
        obj, Vector2D(0, 0, 0), physics.worldBox, Vector2D(x, 400, 0));
    con->color = cir->color;
    physics.AddConstantConstraint(con);
  }

  Object *box = ShapeFactory::GenerateBoxObject(700, 20);
  box->SetPosition(Vector2D(400, 50, 1));
  box->SetFixed(true);
  physics.AddObject(box);

  n = 32;
  for (int i = 0; i < n; i++) {
    box = ShapeFactory::GenerateBoxObject(7, 50);
    box->GetShape(0)->color = HSB3f(360.0f / n * i, 0.8f, 1.0f);
    box->GetShape(0)->friction = 0.4;
    box->SetPosition(Vector2D(100 + i * 20, 100, 1));
    physics.AddObject(box);
  }
}

void Game::Test6() {
  /*    physics.DeleteAllShapes();

      Polygon *box = Polygon::GenerateBox(700, 20);
      box->SetCenterPosition(Vector2D(400, 50, 1));
      box->SetFixed(true);
      physics.AddShape(box);

      int n = 3, m = 16;
      double R = 9.0;
      Circle *cir[3][20];
      RGB3f color = RGB3f::RandomBrightColor(), springColor =
     RGB3f::RandomBrightColor();
      for (int i = 0; i < n; i++) {
          for (int j = 0; j < m - i; j++) {
              cir[i][j] = Circle::GenerateCircle(R);
              cir[i][j]->color = color;
              cir[i][j]->SetCenterPosition(Vector2D(100 + R * 4 * j + R * i * 2,
     100 + R * 4 * i, 1));
              physics.AddShape(cir[i][j]);
          }
      }
      double strength = 18000.0;
      for (int i = 0; i < n; i++) {
          for (int j = 0; j < m - i; j++) {
              if (j + 1 < m - i) {
                  Spring *spring = new Spring(cir[i][j], Vector2D(0, 0, 1),
     cir[i][j + 1], Vector2D(0, 0, 1), strength);
                  spring->color = springColor;
                  physics.AddForce(spring);
              }
              if (i + 1 < n) {
                  if (j > 0) {
                      Spring *spring = new Spring(cir[i][j], Vector2D(0, 0, 1),
     cir[i + 1][j - 1], Vector2D(0, 0, 1), strength);
                      spring->color = springColor;
                      physics.AddForce(spring);
                  }
                  if (j < m - i - 1) {
                      Spring *spring = new Spring(cir[i][j], Vector2D(0, 0, 1),
     cir[i + 1][j], Vector2D(0, 0, 1), strength);
                      spring->color = springColor;
                      physics.AddForce(spring);
                  }
              }
          }
      }
      */
}

void Game::Test7() {}
