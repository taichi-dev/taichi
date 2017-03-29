/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#ifndef OBJECT_H
#define OBJECT_H

#include "Polygon.h"
#include "Circle.h"
#include "Settings.h"

class Object {
public:
    vector<Shape *> shapes;
    double mass, base_mass;
    double inertia, base_inertia;
    double invMass, invInertia;
    bool fixed;
    Vector2D position;
    RGB3f color;
    Vector2D linearVelocity;
    double rotationAngle, angularVelocity;
    Angle angle;
    Matrix3x3 transformToWorld, transformToWorldInverse;
    double linearResistance, rotationResistance;
    Vector2D correctiveLinearVelocity;
    double correctiveAngularVelocity;
    bool fibRot, selected;
    int priority;
    int type;
    void Init() {
        color = RGB3f::RandomBrightColor();
        fixed = false;
        priority = 1;
        rotationAngle = 0.0;
        angularVelocity = 0;
        position = Vector2D(0, 0, 1);
        linearVelocity = Vector2D(0, 0, 0);
        correctiveLinearVelocity = Vector2D(0, 0, 0);
        correctiveAngularVelocity = 0.0;
        linearResistance = rotationResistance = 0.0001;
        selected = false;
        type = 0;
    }


    Vector2D GetPosition() {
        return position;
    }
    Matrix3x3 GetTransformToWorld() {
        return transformToWorld;
    }
    Vector2D GetTransformToWorld(Vector2D vec) {
        return transformToWorld(vec);
    }

    Matrix3x3 GetTransformToWorldInverse() {
        return transformToWorldInverse;
    }
    void SetPosition(Vector2D position) {
        this->position = position;
        ResetTransformToWorld();
    }
    void ResetTransformToWorld() {
        transformToWorld.SetAsAxisTransform(position, rotationAngle);
        transformToWorldInverse.SetAsAxisTransformInverse(position, rotationAngle);
        for (size_t i = 0; i < shapes.size(); i++)
            if (shapes[i]->GetType() == Polygon::ShapeType)
                ((Polygon *)shapes[i])->UpdateCurrentInformation();
        angle = Angle(rotationAngle);
    }
    Shape *GetShape(int id) {
        return shapes[id];
    }
    Object() {
        Init();
    }
    Object(Shape *shape) {
        Init();
        AddShape(shape);
    }
    Object(vector<Shape *> shapes) {
        Init();
        AddShapes(shapes);
    }
    void AddShape(Shape *shape) {
        shapes.push_back(shape);
        shape->object = this;
        Update();
    }
    void AddShapes(vector<Shape *> shapes) {
        for (int i = 0; i < (int)shapes.size(); i++)
            this->shapes.push_back(shapes[i]), shapes[i]->object = this;
        Update();
    }
    void SetColor(RGB3f color) {
        for (int i = 0; i < (int)shapes.size(); i++)
            shapes[i]->color = color;
    }
    void Update() {
        int n = shapes.size();
        base_mass = 0.0;
        for (int i = 0; i < n; i++) {
            shapes[i]->Update();
            base_mass += shapes[i]->mass;
        }
        Vector2D centroid = Vector2D(0, 0, 0);
        for (int i = 0; i < n; i++) {
            centroid += shapes[i]->centroidPosition * shapes[i]->mass;
        }
        centroid /= base_mass;
        centroid.z = 0;
        position += centroid;
        for (int i = 0; i < n; i++)
            shapes[i]->Move(-centroid);
        base_inertia = 0.0;
        for (int i = 0; i < n; i++)
            base_inertia += shapes[i]->inertia + shapes[i]->mass * shapes[i]->centroidPosition.GetLength2();
        if (fixed) {
            mass = DBL_INF;
            inertia = DBL_INF;
        } else mass = base_mass, inertia = base_inertia;
        invMass = 1.0 / mass;
        invInertia = 1.0 / inertia;
        ResetTransformToWorld();
    }
    bool IsPointInside(Vector2D point) {
        for (int i = 0; i < (int)shapes.size(); i++)
            if (shapes[i]->IsPointInside(point)) return true;
        return false;
    }
    void SetRotationAngle(double rotationAngle) {
        this->rotationAngle = rotationAngle;
        ResetTransformToWorld();
    }
    void ApplyGravity(double T) {
        if (fixed) return;
        ApplyImpulse(position, Vector2D(0, -settings.gravity, 0) * mass * T);
    }
    void SetFixed(bool fixed) {
        this->fixed = fixed;
        Update();
    }
    bool GetFixed() {
        return this->fixed;
    }
    void ApplyTorque(double torque) {
        angularVelocity += torque * invInertia;
    }
    void ApplyImpulse(Vector2D r, Vector2D p) {
        linearVelocity += p * invMass;
        ApplyTorque((r - position) % p);
    }
    void ApplyCorrectiveImpulse(Vector2D r, Vector2D p, bool paintNeed = true) {
        correctiveLinearVelocity += p * invMass;
        correctiveAngularVelocity += (r - position) % p * invInertia;
    }
    void Proceed(double T) {
        if (fixed) return; 
        position += T * (linearVelocity);
        rotationAngle += T * (angularVelocity);
        if (sgn(linearVelocity.GetLength2()))
            linearVelocity -= (1 - pow(1 - linearResistance, T / timeInterval)) * linearVelocity.GetLength2() * linearVelocity.GetDirection();
        angularVelocity -= (1 - pow(1 - rotationResistance, T / timeInterval)) * sqr(angularVelocity);
        correctiveLinearVelocity = Vector2D(0, 0, 0);
        correctiveAngularVelocity = 0.0;
        ResetTransformToWorld();
    }
    void ApplyPositionCorrection(double T) {
        if (fixed) return;
        position += T * correctiveLinearVelocity;
        rotationAngle += T * correctiveAngularVelocity;
        correctiveLinearVelocity = Vector2D(0, 0, 0);
        correctiveAngularVelocity = 0.0;
        ResetTransformToWorld();
    }
    Vector2D GetPointVelocity(Vector2D p) {
        if (fixed) return Vector2D(0, 0, 0);
        else return linearVelocity + Vector2D(-angularVelocity * (p.y - position.y), angularVelocity * (p.x - position.x), 0);
    }
    void Redraw() {
        for (int i = 0; i < (int)shapes.size(); i++)
            shapes[i]->Redraw();
    }
    void SetLayerMask(int layerMask) {
        for (int i = 0; i < (int)shapes.size(); i++)
            shapes[i]->layerMask = layerMask;
    }
    void SetResistance(double linear, double angular) {
        linearResistance = linear;
        rotationResistance = angular;
    }

public:
    static const int TYPE_WORLD_BOX = 1;
    static const int TYPE_GEAR = 2;


};

#endif