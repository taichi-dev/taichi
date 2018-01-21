/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#pragma once

#include <taichi/common/interface.h>

TC_NAMESPACE_BEGIN

#ifdef TC_USE_ANN

class ANNkd_tree;

class NearestNeighbour2D {
 public:
  NearestNeighbour2D();

  NearestNeighbour2D(const std::vector<Vector2> &data_points);

  void clear();

  void initialize(const std::vector<Vector2> &data_points);

  Vector2 query_point(Vector2 p) const;

  int query_index(Vector2 p) const;

  void query(Vector2 p, int &index, float &dist) const;

  void query_n(Vector2 p,
               int n,
               std::vector<int> &index,
               std::vector<float> &dist) const;

  void query_n_index(Vector2 p, int n, std::vector<int> &index) const;

 private:
  std::vector<Vector2> data_points;
  std::shared_ptr<ANNkd_tree> ann_kdtree;
};

#else

class NearestNeighbour2D {
 public:
  NearestNeighbour2D() {
    TC_ERROR("no impl");
  }

  NearestNeighbour2D(const std::vector<Vector2> &data_points) {
    TC_ERROR("no impl");
  }

  void clear() {
  }

  void initialize(const std::vector<Vector2> &data_points) {
  }

  Vector2 query_point(Vector2 p) const {
    return Vector2(0.0_f);
  }

  int query_index(Vector2 p) const {
    return -1;
  }

  void query(Vector2 p, int &index, float &dist) const {
  }

  void query_n(Vector2 p,
               int n,
               std::vector<int> &index,
               std::vector<real> &dist) const {
  }

  void query_n_index(Vector2 p, int n, std::vector<int> &index) const {
  }
};

#endif

TC_NAMESPACE_END
