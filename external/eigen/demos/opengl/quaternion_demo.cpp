// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "quaternion_demo.h"
#include "icosphere.h"

#include <Eigen/Geometry>
#include <Eigen/QR>
#include <Eigen/LU>

#include <iostream>
#include <QEvent>
#include <QMouseEvent>
#include <QInputDialog>
#include <QGridLayout>
#include <QButtonGroup>
#include <QRadioButton>
#include <QDockWidget>
#include <QPushButton>
#include <QGroupBox>

using namespace Eigen;

class FancySpheres
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    FancySpheres()
    {
      const int levels = 4;
      const float scale = 0.33;
      float radius = 100;
      std::vector<int> parents;

      // leval 0
      mCenters.push_back(Vector3f::Zero());
      parents.push_back(-1);
      mRadii.push_back(radius);

      // generate level 1 using icosphere vertices
      radius *= 0.45;
      {
        float dist = mRadii[0]*0.9;
        for (int i=0; i<12; ++i)
        {
          mCenters.push_back(mIcoSphere.vertices()[i] * dist);
          mRadii.push_back(radius);
          parents.push_back(0);
        }
      }

      static const float angles [10] = {
        0, 0,
        M_PI, 0.*M_PI,
        M_PI, 0.5*M_PI,
        M_PI, 1.*M_PI,
        M_PI, 1.5*M_PI
      };

      // generate other levels
      int start = 1;
      for (int l=1; l<levels; l++)
      {
        radius *= scale;
        int end = mCenters.size();
        for (int i=start; i<end; ++i)
        {
          Vector3f c = mCenters[i];
          Vector3f ax0 = (c - mCenters[parents[i]]).normalized();
          Vector3f ax1 = ax0.unitOrthogonal();
          Quaternionf q;
          q.setFromTwoVectors(Vector3f::UnitZ(), ax0);
          Affine3f t = Translation3f(c) * q * Scaling(mRadii[i]+radius);
          for (int j=0; j<5; ++j)
          {
            Vector3f newC = c + ( (AngleAxisf(angles[j*2+1], ax0)
                                * AngleAxisf(angles[j*2+0] * (l==1 ? 0.35 : 0.5), ax1)) * ax0)
                                * (mRadii[i] + radius*0.8);
            mCenters.push_back(newC);
            mRadii.push_back(radius);
            parents.push_back(i);
          }
        }
        start = end;
      }
    }

    void draw()
    {
      int end = mCenters.size();
      glEnable(GL_NORMALIZE);
      for (int i=0; i<end; ++i)
      {
        Affine3f t = Translation3f(mCenters[i]) * Scaling(mRadii[i]);
        gpu.pushMatrix(GL_MODELVIEW);
        gpu.multMatrix(t.matrix(),GL_MODELVIEW);
        mIcoSphere.draw(2);
        gpu.popMatrix(GL_MODELVIEW);
      }
      glDisable(GL_NORMALIZE);
    }
  protected:
    std::vector<Vector3f> mCenters;
    std::vector<float> mRadii;
    IcoSphere mIcoSphere;
};


// generic linear interpolation method
template<typename T> T lerp(float t, const T& a, const T& b)
{
  return a*(1-t) + b*t;
}

// quaternion slerp
template<> Quaternionf lerp(float t, const Quaternionf& a, const Quaternionf& b)
{ return a.slerp(t,b); }

// linear interpolation of a frame using the type OrientationType
// to perform the interpolation of the orientations
template<typename OrientationType>
inline static Frame lerpFrame(float alpha, const Frame& a, const Frame& b)
{
  return Frame(lerp(alpha,a.position,b.position),
               Quaternionf(lerp(alpha,OrientationType(a.orientation),OrientationType(b.orientation))));
}

template<typename _Scalar> class EulerAngles
{
public:
  enum { Dim = 3 };
  typedef _Scalar Scalar;
  typedef Matrix<Scalar,3,3> Matrix3;
  typedef Matrix<Scalar,3,1> Vector3;
  typedef Quaternion<Scalar> QuaternionType;

protected:

  Vector3 m_angles;

public:

  EulerAngles() {}
  inline EulerAngles(Scalar a0, Scalar a1, Scalar a2) : m_angles(a0, a1, a2) {}
  inline EulerAngles(const QuaternionType& q) { *this = q; }

  const Vector3& coeffs() const { return m_angles; }
  Vector3& coeffs() { return m_angles; }

  EulerAngles& operator=(const QuaternionType& q)
  {
    Matrix3 m = q.toRotationMatrix();
    return *this = m;
  }

  EulerAngles& operator=(const Matrix3& m)
  {
    // mat =  cy*cz          -cy*sz           sy
    //        cz*sx*sy+cx*sz  cx*cz-sx*sy*sz -cy*sx
    //       -cx*cz*sy+sx*sz  cz*sx+cx*sy*sz  cx*cy
    m_angles.coeffRef(1) = std::asin(m.coeff(0,2));
    m_angles.coeffRef(0) = std::atan2(-m.coeff(1,2),m.coeff(2,2));
    m_angles.coeffRef(2) = std::atan2(-m.coeff(0,1),m.coeff(0,0));
    return *this;
  }

  Matrix3 toRotationMatrix(void) const
  {
    Vector3 c = m_angles.array().cos();
    Vector3 s = m_angles.array().sin();
    Matrix3 res;
    res <<  c.y()*c.z(),                    -c.y()*s.z(),                   s.y(),
            c.z()*s.x()*s.y()+c.x()*s.z(),  c.x()*c.z()-s.x()*s.y()*s.z(),  -c.y()*s.x(),
            -c.x()*c.z()*s.y()+s.x()*s.z(), c.z()*s.x()+c.x()*s.y()*s.z(),  c.x()*c.y();
    return res;
  }

  operator QuaternionType() { return QuaternionType(toRotationMatrix()); }
};

// Euler angles slerp
template<> EulerAngles<float> lerp(float t, const EulerAngles<float>& a, const EulerAngles<float>& b)
{
  EulerAngles<float> res;
  res.coeffs() = lerp(t, a.coeffs(), b.coeffs());
  return res;
}


RenderingWidget::RenderingWidget()
{
  mAnimate = false;
  mCurrentTrackingMode = TM_NO_TRACK;
  mNavMode = NavTurnAround;
  mLerpMode = LerpQuaternion;
  mRotationMode = RotationStable;
  mTrackball.setCamera(&mCamera);

  // required to capture key press events
  setFocusPolicy(Qt::ClickFocus);
}

void RenderingWidget::grabFrame(void)
{
    // ask user for a time
    bool ok = false;
    double t = 0;
    if (!m_timeline.empty())
      t = (--m_timeline.end())->first + 1.;
    t = QInputDialog::getDouble(this, "Eigen's RenderingWidget", "time value: ",
      t, 0, 1e3, 1, &ok);
    if (ok)
    {
      Frame aux;
      aux.orientation = mCamera.viewMatrix().linear();
      aux.position = mCamera.viewMatrix().translation();
      m_timeline[t] = aux;
    }
}

void RenderingWidget::drawScene()
{
  static FancySpheres sFancySpheres;
  float length = 50;
  gpu.drawVector(Vector3f::Zero(), length*Vector3f::UnitX(), Color(1,0,0,1));
  gpu.drawVector(Vector3f::Zero(), length*Vector3f::UnitY(), Color(0,1,0,1));
  gpu.drawVector(Vector3f::Zero(), length*Vector3f::UnitZ(), Color(0,0,1,1));

  // draw the fractal object
  float sqrt3 = std::sqrt(3.);
  glLightfv(GL_LIGHT0, GL_AMBIENT, Vector4f(0.5,0.5,0.5,1).data());
  glLightfv(GL_LIGHT0, GL_DIFFUSE, Vector4f(0.5,1,0.5,1).data());
  glLightfv(GL_LIGHT0, GL_SPECULAR, Vector4f(1,1,1,1).data());
  glLightfv(GL_LIGHT0, GL_POSITION, Vector4f(-sqrt3,-sqrt3,sqrt3,0).data());

  glLightfv(GL_LIGHT1, GL_AMBIENT, Vector4f(0,0,0,1).data());
  glLightfv(GL_LIGHT1, GL_DIFFUSE, Vector4f(1,0.5,0.5,1).data());
  glLightfv(GL_LIGHT1, GL_SPECULAR, Vector4f(1,1,1,1).data());
  glLightfv(GL_LIGHT1, GL_POSITION, Vector4f(-sqrt3,sqrt3,-sqrt3,0).data());

  glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, Vector4f(0.7, 0.7, 0.7, 1).data());
  glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, Vector4f(0.8, 0.75, 0.6, 1).data());
  glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, Vector4f(1, 1, 1, 1).data());
  glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 64);

  glEnable(GL_LIGHTING);
  glEnable(GL_LIGHT0);
  glEnable(GL_LIGHT1);

  sFancySpheres.draw();
  glVertexPointer(3, GL_FLOAT, 0, mVertices[0].data());
  glNormalPointer(GL_FLOAT, 0, mNormals[0].data());
  glEnableClientState(GL_VERTEX_ARRAY);
  glEnableClientState(GL_NORMAL_ARRAY);
  glDrawArrays(GL_TRIANGLES, 0, mVertices.size());
  glDisableClientState(GL_VERTEX_ARRAY);
  glDisableClientState(GL_NORMAL_ARRAY);

  glDisable(GL_LIGHTING);
}

void RenderingWidget::animate()
{
  m_alpha += double(m_timer.interval()) * 1e-3;

  TimeLine::const_iterator hi = m_timeline.upper_bound(m_alpha);
  TimeLine::const_iterator lo = hi;
  --lo;

  Frame currentFrame;

  if(hi==m_timeline.end())
  {
    // end
    currentFrame = lo->second;
    stopAnimation();
  }
  else if(hi==m_timeline.begin())
  {
    // start
    currentFrame = hi->second;
  }
  else
  {
    float s = (m_alpha - lo->first)/(hi->first - lo->first);
    if (mLerpMode==LerpEulerAngles)
      currentFrame = ::lerpFrame<EulerAngles<float> >(s, lo->second, hi->second);
    else if (mLerpMode==LerpQuaternion)
      currentFrame = ::lerpFrame<Eigen::Quaternionf>(s, lo->second, hi->second);
    else
    {
      std::cerr << "Invalid rotation interpolation mode (abort)\n";
      exit(2);
    }
    currentFrame.orientation.coeffs().normalize();
  }

  currentFrame.orientation = currentFrame.orientation.inverse();
  currentFrame.position = - (currentFrame.orientation * currentFrame.position);
  mCamera.setFrame(currentFrame);

  updateGL();
}

void RenderingWidget::keyPressEvent(QKeyEvent * e)
{
    switch(e->key())
    {
      case Qt::Key_Up:
        mCamera.zoom(2);
        break;
      case Qt::Key_Down:
        mCamera.zoom(-2);
        break;
      // add a frame
      case Qt::Key_G:
        grabFrame();
        break;
      // clear the time line
      case Qt::Key_C:
        m_timeline.clear();
        break;
      // move the camera to initial pos
      case Qt::Key_R:
        resetCamera();
        break;
      // start/stop the animation
      case Qt::Key_A:
        if (mAnimate)
        {
          stopAnimation();
        }
        else
        {
          m_alpha = 0;
          connect(&m_timer, SIGNAL(timeout()), this, SLOT(animate()));
          m_timer.start(1000/30);
          mAnimate = true;
        }
        break;
      default:
        break;
    }

    updateGL();
}

void RenderingWidget::stopAnimation()
{
  disconnect(&m_timer, SIGNAL(timeout()), this, SLOT(animate()));
  m_timer.stop();
  mAnimate = false;
  m_alpha = 0;
}

void RenderingWidget::mousePressEvent(QMouseEvent* e)
{
  mMouseCoords = Vector2i(e->pos().x(), e->pos().y());
  bool fly = (mNavMode==NavFly) || (e->modifiers()&Qt::ControlModifier);
  switch(e->button())
  {
    case Qt::LeftButton:
      if(fly)
      {
        mCurrentTrackingMode = TM_LOCAL_ROTATE;
        mTrackball.start(Trackball::Local);
      }
      else
      {
        mCurrentTrackingMode = TM_ROTATE_AROUND;
        mTrackball.start(Trackball::Around);
      }
      mTrackball.track(mMouseCoords);
      break;
    case Qt::MidButton:
      if(fly)
        mCurrentTrackingMode = TM_FLY_Z;
      else
        mCurrentTrackingMode = TM_ZOOM;
      break;
    case Qt::RightButton:
        mCurrentTrackingMode = TM_FLY_PAN;
      break;
    default:
      break;
  }
}
void RenderingWidget::mouseReleaseEvent(QMouseEvent*)
{
    mCurrentTrackingMode = TM_NO_TRACK;
    updateGL();
}

void RenderingWidget::mouseMoveEvent(QMouseEvent* e)
{
    // tracking
    if(mCurrentTrackingMode != TM_NO_TRACK)
    {
        float dx =   float(e->x() - mMouseCoords.x()) / float(mCamera.vpWidth());
        float dy = - float(e->y() - mMouseCoords.y()) / float(mCamera.vpHeight());

        // speedup the transformations
        if(e->modifiers() & Qt::ShiftModifier)
        {
          dx *= 10.;
          dy *= 10.;
        }

        switch(mCurrentTrackingMode)
        {
          case TM_ROTATE_AROUND:
          case TM_LOCAL_ROTATE:
            if (mRotationMode==RotationStable)
            {
              // use the stable trackball implementation mapping
              // the 2D coordinates to 3D points on a sphere.
              mTrackball.track(Vector2i(e->pos().x(), e->pos().y()));
            }
            else
            {
              // standard approach mapping the x and y displacements as rotations
              // around the camera's X and Y axes.
              Quaternionf q = AngleAxisf( dx*M_PI, Vector3f::UnitY())
                            * AngleAxisf(-dy*M_PI, Vector3f::UnitX());
              if (mCurrentTrackingMode==TM_LOCAL_ROTATE)
                mCamera.localRotate(q);
              else
                mCamera.rotateAroundTarget(q);
            }
            break;
          case TM_ZOOM :
            mCamera.zoom(dy*100);
            break;
          case TM_FLY_Z :
            mCamera.localTranslate(Vector3f(0, 0, -dy*200));
            break;
          case TM_FLY_PAN :
            mCamera.localTranslate(Vector3f(dx*200, dy*200, 0));
            break;
          default:
            break;
        }

        updateGL();
    }

    mMouseCoords = Vector2i(e->pos().x(), e->pos().y());
}

void RenderingWidget::paintGL()
{
  glEnable(GL_DEPTH_TEST);
  glDisable(GL_CULL_FACE);
  glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
  glDisable(GL_COLOR_MATERIAL);
  glDisable(GL_BLEND);
  glDisable(GL_ALPHA_TEST);
  glDisable(GL_TEXTURE_1D);
  glDisable(GL_TEXTURE_2D);
  glDisable(GL_TEXTURE_3D);

  // Clear buffers
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  mCamera.activateGL();

  drawScene();
}

void RenderingWidget::initializeGL()
{
  glClearColor(1., 1., 1., 0.);
  glLightModeli(GL_LIGHT_MODEL_LOCAL_VIEWER, 1);
  glDepthMask(GL_TRUE);
  glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);

  mCamera.setPosition(Vector3f(-200, -200, -200));
  mCamera.setTarget(Vector3f(0, 0, 0));
  mInitFrame.orientation = mCamera.orientation().inverse();
  mInitFrame.position = mCamera.viewMatrix().translation();
}

void RenderingWidget::resizeGL(int width, int height)
{
    mCamera.setViewport(width,height);
}

void RenderingWidget::setNavMode(int m)
{
  mNavMode = NavMode(m);
}

void RenderingWidget::setLerpMode(int m)
{
  mLerpMode = LerpMode(m);
}

void RenderingWidget::setRotationMode(int m)
{
  mRotationMode = RotationMode(m);
}

void RenderingWidget::resetCamera()
{
  if (mAnimate)
    stopAnimation();
  m_timeline.clear();
  Frame aux0 = mCamera.frame();
  aux0.orientation = aux0.orientation.inverse();
  aux0.position = mCamera.viewMatrix().translation();
  m_timeline[0] = aux0;

  Vector3f currentTarget = mCamera.target();
  mCamera.setTarget(Vector3f::Zero());

  // compute the rotation duration to move the camera to the target
  Frame aux1 = mCamera.frame();
  aux1.orientation = aux1.orientation.inverse();
  aux1.position = mCamera.viewMatrix().translation();
  float duration = aux0.orientation.angularDistance(aux1.orientation) * 0.9;
  if (duration<0.1) duration = 0.1;

  // put the camera at that time step:
  aux1 = aux0.lerp(duration/2,mInitFrame);
  // and make it look at the target again
  aux1.orientation = aux1.orientation.inverse();
  aux1.position = - (aux1.orientation * aux1.position);
  mCamera.setFrame(aux1);
  mCamera.setTarget(Vector3f::Zero());

  // add this camera keyframe
  aux1.orientation = aux1.orientation.inverse();
  aux1.position = mCamera.viewMatrix().translation();
  m_timeline[duration] = aux1;

  m_timeline[2] = mInitFrame;
  m_alpha = 0;
  animate();
  connect(&m_timer, SIGNAL(timeout()), this, SLOT(animate()));
  m_timer.start(1000/30);
  mAnimate = true;
}

QWidget* RenderingWidget::createNavigationControlWidget()
{
  QWidget* panel = new QWidget();
  QVBoxLayout* layout = new QVBoxLayout();

  {
    QPushButton* but = new QPushButton("reset");
    but->setToolTip("move the camera to initial position (with animation)");
    layout->addWidget(but);
    connect(but, SIGNAL(clicked()), this, SLOT(resetCamera()));
  }
  {
    // navigation mode
    QGroupBox* box = new QGroupBox("navigation mode");
    QVBoxLayout* boxLayout = new QVBoxLayout;
    QButtonGroup* group = new QButtonGroup(panel);
    QRadioButton* but;
    but = new QRadioButton("turn around");
    but->setToolTip("look around an object");
    group->addButton(but, NavTurnAround);
    boxLayout->addWidget(but);
    but = new QRadioButton("fly");
    but->setToolTip("free navigation like a spaceship\n(this mode can also be enabled pressing the \"shift\" key)");
    group->addButton(but, NavFly);
    boxLayout->addWidget(but);
    group->button(mNavMode)->setChecked(true);
    connect(group, SIGNAL(buttonClicked(int)), this, SLOT(setNavMode(int)));
    box->setLayout(boxLayout);
    layout->addWidget(box);
  }
  {
    // track ball, rotation mode
    QGroupBox* box = new QGroupBox("rotation mode");
    QVBoxLayout* boxLayout = new QVBoxLayout;
    QButtonGroup* group = new QButtonGroup(panel);
    QRadioButton* but;
    but = new QRadioButton("stable trackball");
    group->addButton(but, RotationStable);
    boxLayout->addWidget(but);
    but->setToolTip("use the stable trackball implementation mapping\nthe 2D coordinates to 3D points on a sphere");
    but = new QRadioButton("standard rotation");
    group->addButton(but, RotationStandard);
    boxLayout->addWidget(but);
    but->setToolTip("standard approach mapping the x and y displacements\nas rotations around the camera's X and Y axes");
    group->button(mRotationMode)->setChecked(true);
    connect(group, SIGNAL(buttonClicked(int)), this, SLOT(setRotationMode(int)));
    box->setLayout(boxLayout);
    layout->addWidget(box);
  }
  {
    // interpolation mode
    QGroupBox* box = new QGroupBox("spherical interpolation");
    QVBoxLayout* boxLayout = new QVBoxLayout;
    QButtonGroup* group = new QButtonGroup(panel);
    QRadioButton* but;
    but = new QRadioButton("quaternion slerp");
    group->addButton(but, LerpQuaternion);
    boxLayout->addWidget(but);
    but->setToolTip("use quaternion spherical interpolation\nto interpolate orientations");
    but = new QRadioButton("euler angles");
    group->addButton(but, LerpEulerAngles);
    boxLayout->addWidget(but);
    but->setToolTip("use Euler angles to interpolate orientations");
    group->button(mNavMode)->setChecked(true);
    connect(group, SIGNAL(buttonClicked(int)), this, SLOT(setLerpMode(int)));
    box->setLayout(boxLayout);
    layout->addWidget(box);
  }
  layout->addItem(new QSpacerItem(0,0,QSizePolicy::Minimum,QSizePolicy::Expanding));
  panel->setLayout(layout);
  return panel;
}

QuaternionDemo::QuaternionDemo()
{
  mRenderingWidget = new RenderingWidget();
  setCentralWidget(mRenderingWidget);

  QDockWidget* panel = new QDockWidget("navigation", this);
  panel->setAllowedAreas((QFlags<Qt::DockWidgetArea>)(Qt::RightDockWidgetArea | Qt::LeftDockWidgetArea));
  addDockWidget(Qt::RightDockWidgetArea, panel);
  panel->setWidget(mRenderingWidget->createNavigationControlWidget());
}

int main(int argc, char *argv[])
{
  std::cout << "Navigation:\n";
  std::cout << "  left button:           rotate around the target\n";
  std::cout << "  middle button:         zoom\n";
  std::cout << "  left button + ctrl     quake rotate (rotate around camera position)\n";
  std::cout << "  middle button + ctrl   walk (progress along camera's z direction)\n";
  std::cout << "  left button:           pan (translate in the XY camera's plane)\n\n";
  std::cout << "R : move the camera to initial position\n";
  std::cout << "A : start/stop animation\n";
  std::cout << "C : clear the animation\n";
  std::cout << "G : add a key frame\n";

  QApplication app(argc, argv);
  QuaternionDemo demo;
  demo.resize(600,500);
  demo.show();
  return app.exec();
}

#include "quaternion_demo.moc"

