#include "nn_concat.cuh"
#include "cuda_misc.cuh"

/*
#ifndef __CUDACC__
#define __CUDACC__
#endif

#include <device_functions.h>
#include <device_launch_parameters.h>
*/

#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

__global__ void __memcpy(
	const nn_type* input,
	nn_type* output,
	cuint* in_shape,
	cuint* out_shape,
	uint rank,
	cuint axis,
	cuint axis_offset,
	cuint t_len
) {
	cuint xidx = blockIdx.x * blockDim.x + threadIdx.x;
	uint step = 1;
	uint index = 0;
	uint quo = xidx;

	while (rank-- > 0) {
		cuint m = in_shape[rank];
		cuint k = out_shape[rank];
		cuint n = quo % m;

		index += step * (axis == rank ? n + axis_offset : n);
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

	delete p_dims;

	int n = 0;

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

/**********************************************/
/*                                            */
/*                  NN_Concat                 */
/*                                            */
/**********************************************/

NN_Concat::NN_Concat(int axis, const std::string& name) :
	NN_Layer(name, "concat"),
	_axis(axis)
{

}

void NN_Concat::get_output_shape(const NN_List<NN_Shape>& input_shape, NN_List<NN_Shape>& output_shape) {
	NN_List<NN_Shape> shapes = input_shape;
	int n = 0;

	for (NN_List<NN_Shape>& shape : shapes) {
		NN_Shape& m_shape = shape.val();

		n += m_shape.pop(_axis);
	}

	const NN_Shape& first = shapes[0].val();

	for (const NN_List<NN_Shape>& shape : shapes) {
		const NN_Shape& m_shape = shape.val();

		if (m_shape != first) {
			ErrorExcept(
				"[NN_Concat::get_output_shape] Different shapes %s != %s",
				shape_to_str(first),
				shape_to_str(m_shape)
			);
		}
	}

	NN_Shape out_shape = first;

	out_shape.insert(_axis, n);
	output_shape.append(out_shape);
}

void NN_Concat::run(NN_Stream& st, const NN_List<GpuTensor<nn_type>>& input, NN_List<GpuTensor<nn_type>>& output) {
	GpuTensor<nn_type>& m_output = output.val();
	const NN_Shape out_shape = m_output.get_shape();
	nn_type* out_data = m_output.get_ptr();
	
	int ranks = out_shape.ranks();
	uint* h_dims = new uint[ranks];

	for (int i = 0; i < ranks; ++i) h_dims[i] = (uint)out_shape[i];

	cuint* g_out_dims = set_const_mem(h_dims, ranks, 0);

	uint n = 0;
	uint offset = 0;

	for (const NN_List<GpuTensor<nn_type>>& m_input : input) {
		const GpuTensor<nn_type>& in_tensor = m_input.val();
		const NN_Shape in_shape = in_tensor.get_shape();
		const nn_type* in_data = in_tensor.get_ptr();

		for (int i = 0; i < ranks; ++i) h_dims[i] = (uint)in_shape[i];

		cuint* g_in_dims = set_const_mem(h_dims, ranks, ranks);
		dim3 threads(BLOCK_1024);
		dim3 blocks = get_grid_size(threads, in_shape.total_size());

		__memcpy<<<blocks, threads, 0, st[n++ % STREAMS]>>>(
			in_data,
			out_data,
			g_in_dims,
			g_out_dims,
			ranks,
			_axis,
			offset,
			in_shape.total_size()
		);

		offset += in_shape[_axis];
	}

	delete[] h_dims;
}