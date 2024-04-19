#ifndef _CAST_CUH
#define _CAST_CUH

#include "../cpp_source/cuda_common.h"

#ifndef __CUDACC__
#define __CUDACC__
#endif

#include <device_functions.h>
#include <device_launch_parameters.h>


template <typename sT, typename dT>
__global__ void __cast(
	sT* src,
	dT* dst,
	size_t elem_size
) {
	size_t index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < elem_size) dst[index] = dT(src[index]);
}


template <typename _ST, typename _DT>
void type_cast(_ST* src, _DT* dst, size_t len) {
	dim3 threads(BLOCK_1024);
	dim3 blocks((BLOCK_1024 + len - 1) / BLOCK_1024);

	__cast<<<blocks, threads>>>(src, dst, len);
}


#endif // !_CAST_CUH
