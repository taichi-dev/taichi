/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#ifndef POLYGON_H
#define POLYGON_H

#include "Shape.h"

class Object;

class Polygon : public Shape {
  friend class Physics;

 private:
  int nPoints;
  vector<Vector2D> points, normals, curPoints, curNormals;

  double GetCutLengthX(double x) {
    double minI = -DBL_INF, maxI = DBL_INF;
    Line cutLine(Vector2D(x, 0, 1), Vector2D(x, 1, 1));
    for (int i = 0; i < nPoints; i++) {
      Line edge(points[i], points[(i + 1) % nPoints]);
      if (sgn(edge.v.x) == 0)
        continue;
      else {
        double c = (edge * cutLine).y;
        if (sgn(edge.v.x) < 0) {
          maxI = min(maxI, c);
        } else {
          minI = max(minI, c);
        }
      }
    }
    return maxI - minI;
  }
  double GetCutLengthY(double y) {
    double minI = -DBL_INF, maxI = DBL_INF;
    Line cutLine(Vector2D(0, y, 1), Vector2D(1, y, 1));
    for (int i = 0; i < nPoints; i++) {
      Line edge(points[i], points[(i + 1) % nPoints]);
      if (sgn(edge.v.y) == 0)
        continue;
      else {
        double c = (edge * cutLine).x;
        if (sgn(edge.v.y) > 0) {
          maxI = min(maxI, c);
        } else {
          minI = max(minI, c);
        }
      }
    }
    return maxI - minI;
  }

 public:
  static const int ShapeType = 0;
  static const double GearD;
  Polygon() : Shape() { nPoints = 0; }
  Polygon(Vector2D *begin, Vector2D *end) {
    nPoints = end - begin;
    points = vector<Vector2D>(begin, end);
    Update();
    curPoints.resize(nPoints);
    curNormals.resize(nPoints);
    for (int i = 0; i < nPoints; i++)
      normals.push_back((points[loopNext(i, nPoints)] - points[i])
                            .GetInverseRotate()
                            .GetDirection());
  }
  Polygon(vector<Vector2D> points) : Shape() {
    nPoints = (int)points.size();
    this->points = points;
    Update();
    curPoints.resize(nPoints);
    curNormals.resize(nPoints);
    for (int i = 0; i < nPoints; i++)
      normals.push_back((points[loopNext(i, nPoints)] - points[i])
                            .GetInverseRotate()
                            .GetDirection());
  }
  void Move(Vector2D vec) {
    for (int i = 0; i < (int)points.size(); i++)
      points[i] += vec;
    Update();
  }
  void Redraw() {
    // graphics.DrawPolygon(curPoints, color, boundaryWidth);
  }
  const Vector2D &GetNormal(int i) const { return curNormals[i]; }
  Vector2D GetLowestPoint(Vector2D normal) {
    double lowest = DBL_INF;
    Vector2D ret;
    for (int i = 0; i < nPoints; i++) {
      double prod = normal * curPoints[i];
      if (prod < lowest) {
        ret = curPoints[i];
        lowest = prod;
      }
    }
    return ret;
  }
  const Vector2D &GetPoint(int i) const { return curPoints[i]; }
  const Line GetEdge(int i) const {
    return Line(GetPoint(i), GetPoint(loopNext(i, nPoints)));
  }
  bool IsPointInside(Vector2D p) const {
    p = GetTransformToWorldInverse()(p);
    for (int i = 0; i < nPoints; i++)
      if (sgn((p - points[i]) % (points[loopNext(i, nPoints)] - points[i])) > 0)
        return false;
    return true;
  }
  double CalcMass() {
    double ret = 0;
    for (int i = 0; i < nPoints; i++)
      ret += points[i] % points[loopNext(i, nPoints)];
    ret *= density / 2;
    return ret;
  }
  void ResetCentroid() {
    mass = CalcMass();
    Vector2D newCenter = Vector2D(0, 0, 0);
    for (int i = 0; i < nPoints; i++)
      newCenter += (points[i] + points[(i + 1) % nPoints]) *
                   (points[i] % points[(i + 1) % nPoints]) / 2;
    newCenter /= (mass / density) * 3;
    newCenter.z = 0;
    centroidPosition = newCenter;
  }
  double CalcInertia() {
    double ret = 0;
    vector<double> xList, yList;
    for (int i = 0; i < nPoints; i++)
      xList.push_back(points[i].x), yList.push_back(points[i].y);
    sort(xList.begin(), xList.end());
    sort(yList.begin(), yList.end());
    for (int i = 0; i < nPoints - 1; i++) {
      double x0 = xList[i], x3 = xList[i + 1];
      double x1 = x0 + (x3 - x0) / 3, x2 = x0 + (x3 - x0) / 3 * 2;
      ret += (GetCutLengthX(x0) * x0 * x0 + GetCutLengthX(x1) * x1 * x1 * 3 +
              GetCutLengthX(x2) * x2 * x2 * 3 + GetCutLengthX(x3) * x3 * x3) /
             8.0 * (x3 - x0);
    }
    for (int i = 0; i < nPoints - 1; i++) {
      double y0 = yList[i], y3 = yList[i + 1];
      double y1 = y0 + (y3 - y0) / 3, y2 = y0 + (y3 - y0) / 3 * 2;
      ret += (GetCutLengthY(y0) * y0 * y0 + GetCutLengthY(y1) * y1 * y1 * 3 +
              GetCutLengthY(y2) * y2 * y2 * 3 + GetCutLengthY(y3) * y3 * y3) /
             8.0 * (y3 - y0);
    }
    return ret;
  }
  void ResetDirection() {
    bool isCW = false, isCCW = false;
    for (int i = 0; i < nPoints; i++) {
      Vector2D a = points[(i + 1) % nPoints] - points[i],
               b = points[(i + 2) % nPoints] - points[(i + 1) % nPoints];
      int t = sgn(a % b);
      if (t > 0)
        isCCW = true;
      if (t < 0)
        isCW = true;
    }
    if (isCCW && isCW) {
      printf("Warning : Polygon non-convex!!!\n");
    }
    if (isCW)
      reverse(points.begin(), points.end());
  }
  void Update() {
    ResetDirection();
    ResetCentroid();
    inertia = CalcInertia();
  }
  void GetProjection(const Vector2D &normal, double &minI, double &maxI) {
    assert(nPoints);
    minI = maxI = GetPoint(0) * normal;
    for (int i = 1; i < nPoints; i++) {
      double prod = curPoints[i] * normal;
      if (prod < minI)
        minI = prod;
      if (prod > maxI)
        maxI = prod;
    }
  }
  static Polygon *GeneratePolygon(int nPoints, double r) {
    vector<Vector2D> points;
    points.resize(nPoints);
    for (int i = 0; i < nPoints; i++)
      points[i] = Vector2D::Origin +
                  r * Vector2D::RotatedUnitVector(-pi / 2 + pi / nPoints +
                                                  2 * pi / nPoints * i);
    Polygon *poly = new Polygon(points);
    return poly;
  }
  int GetType() { return ShapeType; }
  static Polygon *GenerateBox(double w, double h) {
    Vector2D points[4];
    w /= 2;
    h /= 2;
    points[0] = Vector2D(-w, -h, 1);
    points[1] = Vector2D(w, -h, 1);
    points[2] = Vector2D(w, h, 1);
    points[3] = Vector2D(-w, h, 1);
    Polygon *poly = new Polygon(points, points + 4);
    return poly;
  }
  static vector<Vector2D> GenerateGearPoints(Vector2D center,
                                             double r,
                                             double rad) {
    double d = 14, R = r;
    int n = (int)ceil(2 * pi / rad);
    rad = 2 * pi / n;
    vector<Vector2D> points;
    for (int i = 0; i < n; i++) {
      points.push_back(center +
                       Vector2D::RotatedUnitVector(i * rad + 0.000 * rad) *
                           (R - d));
      points.push_back(center +
                       Vector2D::RotatedUnitVector(i * rad + 0.250 * rad) *
                           (R - d));
      points.push_back(center +
                       Vector2D::RotatedUnitVector(i * rad + 0.525 * rad) *
                           (R + d));
      points.push_back(center +
                       Vector2D::RotatedUnitVector(i * rad + 0.725 * rad) *
                           (R + d));
    }
    return points;
  }
  AABB GetAABB() {
    AABB ret;
    GetProjection(Vector2D(1, 0, 0), ret.x0, ret.x1);
    GetProjection(Vector2D(0, 1, 0), ret.y0, ret.y1);
    return ret;
  }
  void UpdateCurrentInformation();
};

#endif
