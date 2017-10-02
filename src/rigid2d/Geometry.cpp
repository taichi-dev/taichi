/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#include "Geometry.h"
#include <climits>

Vector2D Vector2D::Origin = Vector2D(0, 0, 1);

inline int TestConvex(vector<Vector2D> points) {
  int n = (int)points.size();
  bool isCW = false, isCCW = false;
  for (int i = 0; i < n; i++) {
    Vector2D a = points[(i + 1) % n] - points[i],
             b = points[(i + 2) % n] - points[(i + 1) % n];
    int t = sgn(a % b);
    if (t > 0)
      isCCW = true;
    if (t < 0)
      isCW = true;
  }
  if (isCCW && isCW)
    return 0;
  return isCCW ? CONVEX_CCW : CONVEX_CW;
}

bool IsInside(Vector2D p, vector<Vector2D> points) {
  int n = (int)points.size();
  int cnt = 0;
  for (int i = 0; i < n; i++) {
    Vector2D a = points[i], b = points[(i + 1) % n];
    if (sgn(a.y - b.y) == 0)
      continue;
    if (a.y < b.y)
      swap(a, b);
    Line line(a, b);
    Vector2D q = line * Line(p, p + Vector2D(1, 0, 0));
    if (sgn(q.x - p.x) >= 0 && sgn(b.y - q.y) <= 0 && sgn(q.y - a.y) < 0)
      cnt++;
  }
  return cnt % 2 == 1;
}

bool IsInside(Vector2D p, Line l) {
  return sgn((p - l.a) * l.v) * sgn((p - l.b) * l.v) <= 0;
}

bool IsInsideStrict(Vector2D p, Line l) {
  return sgn((p - l.a) * l.v) * sgn((p - l.b) * l.v) < 0;
}

bool LegalCut(Line line, vector<Vector2D> points) {
  if (!IsInside(line.a + line.v * 0.5, points))
    return false;
  int n = (int)points.size();
  for (int i = 0; i < n; i++) {
    Line m(points[i], points[(i + 1) % n]);
    if (sgn(line.v % m.v) == 0) {
      if (sgn((m.a - line.a) % line.v) == 0)
        if (IsInsideStrict(line.a, m) || IsInsideStrict(line.b, m) ||
            IsInsideStrict(m.a, line) || IsInsideStrict(m.b, line))
          return false;
    } else {
      Vector2D p = line * m;
      if (p == line.a || p == line.b)
        continue;
      if (IsInside(p, line) && IsInside(p, m))
        return false;
    }
  }
  return true;
}

double GetArea(vector<Vector2D> points) {
  double ret = 0;
  int n = (int)points.size();
  for (int i = 0; i < n; i++)
    ret += points[i] % points[(i + 1) % n];
  return ret / 2;
}

vector<Vector2D> CleanCollinear(vector<Vector2D> points) {
  int n = (int)points.size();
  vector<Vector2D> ret;
  for (int i = 0; i < n; i++) {
    if (sgn((points[(i + 2) % n] - points[i]) %
            (points[(i + 1) % n] - points[i])))
      ret.push_back(points[(i + 1) % n]);
  }
  return ret;
}

vector<vector<Vector2D> > CutIntoConvex(vector<Vector2D> points) {
  points = CleanCollinear(points);
  int t = TestConvex(points);
  if (t == CONVEX_CCW) {
    return vector<vector<Vector2D> >(sgn(GetArea(points)) != 0, points);
  } else if (t == CONVEX_CW) {
    reverse(points.begin(), points.end());
    return vector<vector<Vector2D> >(sgn(GetArea(points)) != 0, points);
  }
  vector<Vector2D> points0, points1;
  int n = (int)points.size();
  bool found = false;
  double optLen = 1e300;
  for (int i = 0; i < n; i++) {
    for (int j = i + 3; j <= n; j++)
      if (j - i >= 3 && n - (j - i) >= 1) {
        Line cutLine(points[i], points[j - 1]);
        if (cutLine.v.GetLength() < optLen && LegalCut(cutLine, points)) {
          optLen = cutLine.v.GetLength();
          found = true;
          points0 = vector<Vector2D>(points.begin() + i, points.begin() + j);
          points1 = vector<Vector2D>(points.begin(), points.begin() + i + 1);
          points1.insert(points1.end(), points.begin() + j - 1, points.end());
        }
      }
  }
  assert(found);
  vector<vector<Vector2D> > ret0, ret1;
  ret0 = CutIntoConvex(points0);
  ret1 = CutIntoConvex(points1);
  ret0.insert(ret0.end(), ret1.begin(), ret1.end());
  return ret0;
}
