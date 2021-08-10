// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_QUATERNION_DEMO_H
#define EIGEN_QUATERNION_DEMO_H

#include "gpuhelper.h"
#include "camera.h"
#include "trackball.h"
#include <map>
#include <QTimer>
#include <QtGui/QApplication>
#include <QtOpenGL/QGLWidget>
#include <QtGui/QMainWindow>

class RenderingWidget : public QGLWidget
{
  Q_OBJECT

    typedef std::map<float,Frame> TimeLine;
    TimeLine m_timeline;
    Frame lerpFrame(float t);

    Frame mInitFrame;
    bool mAnimate;
    float m_alpha;

    enum TrackMode {
      TM_NO_TRACK=0, TM_ROTATE_AROUND, TM_ZOOM,
      TM_LOCAL_ROTATE, TM_FLY_Z, TM_FLY_PAN
    };

    enum NavMode {
      NavTurnAround,
      NavFly
    };

    enum LerpMode {
      LerpQuaternion,
      LerpEulerAngles
    };

    enum RotationMode {
      RotationStable,
      RotationStandard
    };

    Camera mCamera;
    TrackMode mCurrentTrackingMode;
    NavMode mNavMode;
    LerpMode mLerpMode;
    RotationMode mRotationMode;
    Vector2i mMouseCoords;
    Trackball mTrackball;

    QTimer m_timer;

    void setupCamera();

    std::vector<Vector3f> mVertices;
    std::vector<Vector3f> mNormals;
    std::vector<int> mIndices;

  protected slots:

    virtual void animate(void);
    virtual void drawScene(void);

    virtual void grabFrame(void);
    virtual void stopAnimation();

    virtual void setNavMode(int);
    virtual void setLerpMode(int);
    virtual void setRotationMode(int);
    virtual void resetCamera();

  protected:

    virtual void initializeGL();
    virtual void resizeGL(int width, int height);
    virtual void paintGL();
    
    //--------------------------------------------------------------------------------
    virtual void mousePressEvent(QMouseEvent * e);
    virtual void mouseReleaseEvent(QMouseEvent * e);
    virtual void mouseMoveEvent(QMouseEvent * e);
    virtual void keyPressEvent(QKeyEvent * e);
    //--------------------------------------------------------------------------------

  public: 
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    RenderingWidget();
    ~RenderingWidget() { }

    QWidget* createNavigationControlWidget();
};

class QuaternionDemo : public QMainWindow
{
  Q_OBJECT
  public:
    QuaternionDemo();
  protected:
    RenderingWidget* mRenderingWidget;
};

#endif // EIGEN_QUATERNION_DEMO_H
