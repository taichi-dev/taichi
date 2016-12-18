#pragma once

#include <functional>
#include <taichi/math/linalg.h>

TC_NAMESPACE_BEGIN

class HashGrid {
private:
	real hash_cell_size;
	std::vector<std::pair<int, int>> cache;
	std::vector<int *> heads;
	std::vector<int> cell_count;
	std::vector<int> built_data;
	int num_grids;
public:
	// spatial hash function from smallppm, http://www.ci.i.u-tokyo.ac.jp/~hachisuka/smallppm_exp.cpp
	unsigned int spatial_hash(const int ix, const int iy, const int iz) const {
		return (unsigned int)((ix * 73856093) ^ (iy * 19349663) ^ (iz * 83492791)) % num_grids;
	}

	void initialize(const real hash_cell_size, int num_grids) {
		this->hash_cell_size = hash_cell_size;
		this->num_grids = num_grids;
		cell_count.resize(num_grids + 1);
		heads.resize(num_grids + 1);
		clear_cache();
	}

	void clear_cache() {
		cache.clear();
	}

	void build_grid() {
		memset(&cell_count[0], 0, sizeof(cell_count[0]) * cell_count.size());
		built_data.resize(cache.size());
		for (auto p : cache) {
			cell_count[p.first]++;
		}
		for (int i = 0; i + 2 < cell_count.size(); i++) {
			cell_count[i + 1] += cell_count[i];
		}
		heads[0] = &built_data[0];
		for (int i = (int)cell_count.size() - 1; i > 0; i--) {
			cell_count[i] = cell_count[i - 1];
			heads[i] = heads[0] + cell_count[i];
		}
		cell_count[0] = 0;
		for (auto p : cache) {
			built_data[cell_count[p.first]++] = p.second;
		}
	}

	int *begin(Vector3 p) const {
		Vector3 ip = p / hash_cell_size;
		return begin(spatial_hash((int)ip.x, (int)ip.y, (int)ip.z));
	}

	int *end(Vector3 p) const {
		Vector3 ip = p / hash_cell_size;
		return end(spatial_hash((int)ip.x, (int)ip.y, (int)ip.z));
	}

	int *begin(int cell) const {
		return heads[cell];
	}

	int *end(int cell) const {
		return heads[cell + 1];
	}

	void push_back_to_all_cells_in_range(const Vector3 &pos, real range, int val) {
		int bounds[3][2];
		for (int k = 0; k < 3; k++) {
			bounds[k][0] = (int)floor((pos[k] - range) / hash_cell_size);
			bounds[k][1] = (int)ceil((pos[k] + range) / hash_cell_size);
		}
		for (int x = bounds[0][0]; x <= bounds[0][1]; x++)
			for (int y = bounds[1][0]; y <= bounds[1][1]; y++)
				for (int z = bounds[2][0]; z <= bounds[2][1]; z++) {
					int cell_id = spatial_hash(x, y, z);
					cache.push_back(std::make_pair(cell_id, val));
				}
	}

};

TC_NAMESPACE_END
