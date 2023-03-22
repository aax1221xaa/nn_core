#include "cuda_common.h"


dim3 get_grid_size(const dim3 block, uint x, uint y, uint z) {
	dim3 grid = {
		(x + block.x - 1) / block.x,
		(y + block.y - 1) / block.y,
		(z + block.z - 1) / block.z
	};

	return grid;
}

Object_Linker NN_Shared_Ptr::linker;

NN_Shared_Ptr::NN_Shared_Ptr() :
	id(NULL)
{
}

const size_t get_elem_size(const NN_Tensor4D& tensor) {
	return size_t(tensor.h * tensor.c * tensor.h * tensor.w);
}

