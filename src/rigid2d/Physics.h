/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#pragma once

#include "Constraints.h"
#include "QuickIntersectionTest.h"
#include "ShapeFactory.h"
#include "Timer.h"

class Physics {
private:
    IntersectionTest iTest;
public:
    vector<Object *> objects;
    vector<Constraint *> constraints;
    vector<Constraint *> constantConstraints;
    vector<Force *> forces;
    vector<Vector2D> contactPoints;
    Object *worldBox;
    Physics() {
    }
    void SetAsDefaultWorld() {
        DeleteAllObjects();
        settings.gravity = 9.8;
        worldBox = ShapeFactory::GenerateWorldBox();
        AddObject(worldBox);
    }
    void AddForce(Force *force) {
        forces.push_back(force);
    }
    void AddObject(Object *object) {
        object->Update();
        objects.push_back(object);
    }
    void AddConstantConstraint(Constraint *constraint) {
        constantConstraints.push_back(constraint);
    }
    Object *GetWorldBox() {
        return worldBox;
    }
    void DeleteLastObject() {
        DeleteObject(objects.back());
    }
    void DeleteAllObjects() {
        while ((int)objects.size()) DeleteLastObject();
    }
    void DeleteObject(Object *object) {
        delete object;
        sort(objects.begin(), objects.end());
        objects.erase(lower_bound(objects.begin(), objects.end(), object));
        vector<Constraint *> tmp = constantConstraints;
        constantConstraints.clear();
        for (int i = 0; i < (int)tmp.size(); i++)
            if (!tmp[i]->Linking(object))
                constantConstraints.push_back(tmp[i]);
        vector<Force *> tmp1 = forces;
        forces.clear();
        for (int i = 0; i < (int)tmp1.size(); i++)
            if (!tmp1[i]->Linking(object))
                forces.push_back(tmp1[i]);
    }
    void Proceed(double T) {
        CleanRubbish();
        for (int i = 0; i < settings.stepIteration; i++)
            ProceedSmallStep(T / settings.stepIteration);
        for (int i = 0; i < (int)objects.size(); i++)
            objects[i]->ApplyGravity(T);
        SolveSituation(T);
    }
    void ProceedSmallStep(double T) {
        for (int i = 0; i < (int)forces.size(); i++)
            forces[i]->Apply(T);
        for (int i = 0; i < (int)objects.size(); i++)
            objects[i]->Proceed(T);
    }
    void SolveSituation(double T) {

        contactPoints.clear();
        for (int i = 0; i < (int)constantConstraints.size(); i++)
            constraints.push_back(constantConstraints[i]->Copy());
        vector<pair<Shape*, Shape *> > potentialCols;

        iTest.Init(objects, 3.0);
        potentialCols = iTest.GetResult();


        for (int i = 0; i < (int)potentialCols.size(); i++)
            TestCollision(potentialCols[i].first, potentialCols[i].second);

        
        for (int K = 0; K < settings.velocityIteration; K++)
            for (int i = 0; i < (int)constraints.size(); i++)
                constraints[i]->ProcessVelocity();
        //Show Collision Points
        for (int i = 0; i < (int)constraints.size(); i++) {
            if (constraints[i]->type == TYPE_CONTACT)
                contactPoints.push_back(((Contact *)constraints[i])->p);
        }

        iTest.Init(objects, 5);
        potentialCols = iTest.GetResult();        
//        printf("Potential Collisions %d\n", (int)potentialCols.size());
        
        for (int K = 0; K < settings.positionIteration; K++) {
            for (int i = 0; i < (int)constraints.size(); i++) delete(constraints[i]);
            constraints.clear();
            for (int i = 0; i < (int)potentialCols.size(); i++)
                TestCollision(potentialCols[i].first, potentialCols[i].second);
//            random_shuffle(constraints.begin(), constraints.end());
            for (int i = 0; i < (int)constraints.size(); i++)
                constraints[i]->ProcessPosition();
            for (int i = 0; i < (int)constantConstraints.size(); i++)
                constantConstraints[i]->ProcessPosition();
            for (int i = 0; i < (int)objects.size(); i++)
                objects[i]->ApplyPositionCorrection(T);
        }
        for (; !constraints.empty(); ) {
            delete constraints.back();
            constraints.pop_back();
        }
    }
    void CleanRubbish() {
        bool flg = true;
        while (flg) {
            flg = false;
            for (int i = 0; i < (int)objects.size(); i++) {
                if ((objects[i]->position - Vector2D(screenWidth / 2, screenHeight / 2, 1)).GetLength() > 20000) {
                    flg = true;
                    DeleteObject(objects[i]);
                }
            }
        }
    }
    void SelectObject(Vector2D p, Object *&object, Vector2D &r) {
        object = NULL;
        for (int i = 0; i < (int)objects.size(); i++)
            if (objects[i]->IsPointInside(p) && (object == NULL || objects[i]->priority > object->priority)) {
                object = objects[i];
                r = object->transformToWorldInverse(p - object->position);
                Vector2D a = p - object->position;
            }
    }
    void Redraw() {
        /*
        for (int i = 0; i < (int)objects.size(); i++)
            objects[i]->Redraw();
        for (int i = 0; i < (int)constantConstraints.size(); i++)
            constantConstraints[i]->Redraw();
        for (int i = 0; i < (int)forces.size(); i++)
            forces[i]->Redraw();
        for (int i = 0; i < (int)contactPoints.size(); i++)
            graphics.DrawPoint(contactPoints[i].x, contactPoints[i].y, Colors::Black, 3);
        */
    }
    void TestCollision(Polygon *a, Polygon *b);
    void TestCollision(Circle *a, Circle *b);
    void TestCollision(Circle *a, Polygon *b);
    void TestCollision(Shape *a, Shape *b);
    void TestCollision(Object *a, Object *b);

    /*
    static void TW_CALL GetObjectNumberTW(void *value, void *data) {
        Physics *physics = (Physics *)data;
        (*(unsigned int *)value) = physics->objects.size();
    }
    static void TW_CALL GetShapeNumberTW(void *value, void *data) {
        Physics *physics = (Physics *)data;
        int ret = 0;
        for (int i = 0; i < (int)physics->objects.size(); i++)
            ret += (int)physics->objects[i]->shapes.size();
        (*(unsigned int *)value) = ret;
    }
    static void TW_CALL GetForceNumberTW(void *value, void *data) {
        Physics *physics = (Physics *)data;
        (*(unsigned int *)value) = physics->forces.size();
    }
    static void TW_CALL GetConstConstraintNumberTW(void *value, void *data) {
        Physics *physics = (Physics *)data;
        (*(unsigned int *)value) = physics->constantConstraints.size();
    }
    */
};

