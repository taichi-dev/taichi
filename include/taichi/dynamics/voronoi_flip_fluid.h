#pragma once

#include "flip_fluid.h"

TC_NAMESPACE_BEGIN

class VoronoiFLIP : public FLIPFluid {
public:
	void rasterize();
	bool check_u_activity_loose(int i, int j) {
		return (i > 0 && cell_types[i - 1][j] == CellType::WATER) ||
			(i < width && cell_types[i][j] == CellType::WATER);
	}

	bool check_v_activity_loose(int i, int j) {
		return (j > 0 && cell_types[i][j - 1] == CellType::WATER) ||
			(j < height && cell_types[i][j] == CellType::WATER);
	}
};

TC_NAMESPACE_END

