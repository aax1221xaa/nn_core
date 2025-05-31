#include "nn_concat.cuh"
#include "cuda_misc.cuh"


#ifndef __CUDACC__
#define __CUDACC__
#endif

#include <device_functions.h>
#include <device_launch_parameters.h>


__global__ void __memcpy(
	const nn_type* input,
	nn_type* output,
	cuint* in_shape,
	cuint* out_shape,
	cint rank,
	cint axis,
	cuint axis_offset,
	cuint t_len
) {
	cuint xidx = blockIdx.x * blockDim.x + threadIdx.x;
	uint step = 1;
	uint index = 0;
	uint quo = xidx;
	int m_rank = rank;

	while (m_rank-- > 0) {
		cuint m = in_shape[m_rank];
		cuint k = out_shape[m_rank];
		cuint n = quo % m;

		index += step * (axis == m_rank ? n + axis_offset : n);
		step *= k;
		quo /= m;
	}

	if (xidx < t_len) output[index] = input[xidx];
}

void concat_test(
	NN_Stream& stream,
	const NN_List<GpuTensor<nn_type>>& src,
	GpuTensor<nn_type>& dst,
	cuint axis
) {
	nn_type* dst_data = dst.get_ptr();
	const NN_Shape dst_shape = dst.get_shape();

	int dim_size = dst_shape.ranks();
	uint* p_dims = new uint[dim_size];

	for (int i = 0; i < dim_size; ++i) p_dims[i] = (uint)dst_shape[i];

	cuint* g_out_dims = set_const_mem(p_dims, dim_size, 0);
	uint axis_offset = 0;
	int n = 0;

	delete p_dims;

	for (const NN_List<GpuTensor<nn_type>>& tensor : src) {
		const GpuTensor<nn_type>& src_tensor = tensor.val();
		const nn_type* src_data = src_tensor.get_ptr();
		const NN_Shape src_shape = src_tensor.get_shape();
		dim3 threads(1024);
		dim3 blocks = get_grid_size(threads, src_shape.total_size());

		p_dims = new uint[src_shape.ranks()];

		for (int i = 0; i < src_shape.ranks(); ++i) p_dims[i] = (uint)src_shape[i];
		cuint* g_in_dims = set_const_mem(p_dims, src_shape.ranks(), dst_shape.ranks());

		delete[] p_dims;

		__memcpy<<<blocks, threads, 0, stream[n++ % STREAMS]>>>(
			src_data,
			dst_data,
			g_in_dims,
			g_out_dims,
			src_shape.ranks(),
			axis,
			axis_offset,
			src_shape.total_size()
		);
		axis_offset += src_shape[axis];

		//break;
	}
	check_cuda(cudaDeviceSynchronize());
	check_cuda(cudaGetLastError());
}