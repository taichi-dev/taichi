// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "gpuhelper.h"
#include "icosphere.h"
#include <GL/glu.h>
// PLEASE don't look at this old code... ;)

#include <fstream>
#include <algorithm>

GpuHelper gpu;

GpuHelper::GpuHelper()
{
    mVpWidth = mVpHeight = 0;
    mCurrentMatrixTarget = 0;
    mInitialized = false;
}

GpuHelper::~GpuHelper()
{
}

void GpuHelper::pushProjectionMode2D(ProjectionMode2D pm)
{
    // switch to 2D projection
    pushMatrix(Matrix4f::Identity(),GL_PROJECTION);

    if(pm==PM_Normalized)
    {
        //glOrtho(-1., 1., -1., 1., 0., 1.);
    }
    else if(pm==PM_Viewport)
    {
        GLint vp[4];
        glGetIntegerv(GL_VIEWPORT, vp);
        glOrtho(0., vp[2], 0., vp[3], -1., 1.);
    }

    pushMatrix(Matrix4f::Identity(),GL_MODELVIEW);
}

void GpuHelper::popProjectionMode2D(void)
{
    popMatrix(GL_PROJECTION);
    popMatrix(GL_MODELVIEW);
}

void GpuHelper::drawVector(const Vector3f& position, const Vector3f& vec, const Color& color, float aspect /* = 50.*/)
{
    static GLUquadricObj *cylindre = gluNewQuadric();
    glColor4fv(color.data());
    float length = vec.norm();
    pushMatrix(GL_MODELVIEW);
    glTranslatef(position.x(), position.y(), position.z());
    Vector3f ax = Matrix3f::Identity().col(2).cross(vec);
    ax.normalize();
    Vector3f tmp = vec;
    tmp.normalize();
    float angle = 180.f/M_PI * acos(tmp.z());
    if (angle>1e-3)
        glRotatef(angle, ax.x(), ax.y(), ax.z());
    gluCylinder(cylindre, length/aspect, length/aspect, 0.8*length, 10, 10);
    glTranslatef(0.0,0.0,0.8*length);
    gluCylinder(cylindre, 2.0*length/aspect, 0.0, 0.2*length, 10, 10);

    popMatrix(GL_MODELVIEW);
}

void GpuHelper::drawVectorBox(const Vector3f& position, const Vector3f& vec, const Color& color, float aspect)
{
    static GLUquadricObj *cylindre = gluNewQuadric();
    glColor4fv(color.data());
    float length = vec.norm();
    pushMatrix(GL_MODELVIEW);
    glTranslatef(position.x(), position.y(), position.z());
    Vector3f ax = Matrix3f::Identity().col(2).cross(vec);
    ax.normalize();
    Vector3f tmp = vec;
    tmp.normalize();
    float angle = 180.f/M_PI * acos(tmp.z());
    if (angle>1e-3)
        glRotatef(angle, ax.x(), ax.y(), ax.z());
    gluCylinder(cylindre, length/aspect, length/aspect, 0.8*length, 10, 10);
    glTranslatef(0.0,0.0,0.8*length);
    glScalef(4.0*length/aspect,4.0*length/aspect,4.0*length/aspect);
    drawUnitCube();
    popMatrix(GL_MODELVIEW);
}

void GpuHelper::drawUnitCube(void)
{
    static float vertices[][3] = {
        {-0.5,-0.5,-0.5},
        { 0.5,-0.5,-0.5},
        {-0.5, 0.5,-0.5},
        { 0.5, 0.5,-0.5},
        {-0.5,-0.5, 0.5},
        { 0.5,-0.5, 0.5},
        {-0.5, 0.5, 0.5},
        { 0.5, 0.5, 0.5}};

    glBegin(GL_QUADS);
    glNormal3f(0,0,-1); glVertex3fv(vertices[0]); glVertex3fv(vertices[2]); glVertex3fv(vertices[3]); glVertex3fv(vertices[1]);
    glNormal3f(0,0, 1); glVertex3fv(vertices[4]); glVertex3fv(vertices[5]); glVertex3fv(vertices[7]); glVertex3fv(vertices[6]);
    glNormal3f(0,-1,0); glVertex3fv(vertices[0]); glVertex3fv(vertices[1]); glVertex3fv(vertices[5]); glVertex3fv(vertices[4]);
    glNormal3f(0, 1,0); glVertex3fv(vertices[2]); glVertex3fv(vertices[6]); glVertex3fv(vertices[7]); glVertex3fv(vertices[3]);
    glNormal3f(-1,0,0); glVertex3fv(vertices[0]); glVertex3fv(vertices[4]); glVertex3fv(vertices[6]); glVertex3fv(vertices[2]);
    glNormal3f( 1,0,0); glVertex3fv(vertices[1]); glVertex3fv(vertices[3]); glVertex3fv(vertices[7]); glVertex3fv(vertices[5]);
    glEnd();
}

void GpuHelper::drawUnitSphere(int level)
{
  static IcoSphere sphere;
  sphere.draw(level);
}


