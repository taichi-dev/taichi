#pragma once
#include "common/config.h"

class ANNkd_tree;

TC_NAMESPACE_BEGIN

void point_cloud_demo(Config config);

class NearestNeighbour2D {
public:
	NearestNeighbour2D();

	NearestNeighbour2D(const std::vector<Vector2> &data_points);

	void clear();

	void initialize(const std::vector<Vector2> &data_points);

	Vector2 query_point(Vector2 p) const;

	int query_index(Vector2 p) const;

	void query(Vector2 p, int &index, float &dist) const;

	void query_n(Vector2 p, int n, std::vector<int> &index, std::vector<float> &dist) const;

	void query_n_index(Vector2 p, int n, std::vector<int> &index) const;

private:
	std::vector<Vector2> data_points;
	std::shared_ptr<ANNkd_tree> ann_kdtree;
};


TC_NAMESPACE_END

