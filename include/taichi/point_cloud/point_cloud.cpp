#include <ANN/ANN.h>
#include <memory>
#include <vector>

#include "point_cloud.h"

TC_NAMESPACE_BEGIN

NearestNeighbour2D::NearestNeighbour2D() {
}

NearestNeighbour2D::NearestNeighbour2D(const std::vector<Vector2>& data_points) {
	initialize(data_points);
}

void NearestNeighbour2D::clear() {
	data_points.clear();
	ann_kdtree = nullptr;
}

void NearestNeighbour2D::initialize(const std::vector<Vector2>& data_points) {
	assert_info(data_points.size() != 0, "data points empty.");
	clear();

	auto ann_data_points = annAllocPts((int)data_points.size(), 2);
	for (unsigned int i = 0; i < data_points.size(); i++)
		for (unsigned int k = 0; k < 2; k++)
			ann_data_points[i][k] = data_points[i][k];

	this->data_points = data_points;

	ann_kdtree = std::shared_ptr<ANNkd_tree>(new ANNkd_tree(ann_data_points, (int)data_points.size(), 2));
}

Vector2 NearestNeighbour2D::query_point(Vector2 p) const {
	assert_info(!data_points.empty(), "No points for NN!");
	return data_points[query_index(p)];
}

int NearestNeighbour2D::query_index(Vector2 p) const {
	int index;
	float _;
	query(p, index, _);
	return index;
}

void NearestNeighbour2D::query(Vector2 p, int & index, float & dist_out) const
{
	ANNidx idx;
	ANNdist dist;
	ANNcoord point[2];
	for (unsigned int k = 0; k < 2; k++)
		point[k] = p[k];
	ann_kdtree->annkSearch(point, 1, &idx, &dist, 0.0);
	index = idx;
	dist_out = (float)dist;

}

void NearestNeighbour2D::query_n(Vector2 p, int n, std::vector<int>& index, std::vector<float>& dist) const
{
	std::vector<ANNidx> index_in;
	std::vector<ANNdist> dist_in;
	index_in.resize(n);
	dist_in.resize(n);
	ANNcoord point[2];
	for (unsigned int k = 0; k < 2; k++)
		point[k] = p[k];
	int actual_n = std::min(n, (int)data_points.size());
	ann_kdtree->annkSearch(point, actual_n, &index_in[0], &dist_in[0], 0.0);

	index.resize(n);
	dist.resize(n);

	for (int i = 0; i < actual_n; i++) {
		index[i] = index_in[i];
		dist[i] = (float)dist_in[i];
	}
	for (int i = actual_n; i < n; i++) {
		index[i] = -1;
		dist[i] = 1e30f;
	}
}

void NearestNeighbour2D::query_n_index(Vector2 p, int n, std::vector<int>& index) const
{
	std::vector<float> _;
	query_n(p, n, index, _);
}

TC_NAMESPACE_END
