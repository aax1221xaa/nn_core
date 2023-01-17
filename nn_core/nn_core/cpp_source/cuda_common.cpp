#include "cuda_common.h"


dim3 get_grid_size(const dim3 block, uint x = 1, uint y = 1, uint z = 1) {
	dim3 grid = {
		(x + block.x - 1) / block.x,
		(y + block.y - 1) / block.y,
		(z + block.z - 1) / block.z
	};

	return grid;
}