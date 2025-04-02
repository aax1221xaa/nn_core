#include "nn_batch_normalize.cuh"

#ifndef __CUDACC__
#define __CUDACC__
#endif

#include <device_functions.h>
#include <device_launch_parameters.h>


__global__ void __batch_normalization(
	const nn_type* input,
	const nn_type* mean,
	const nn_type* var,
	const nn_type* alpha,
	const nn_type* gamma,
	nn_type* output,
	cuint n,
	cuint c
) {
	cuint xidx = blockIdx.x * blockDim.x + threadIdx.x;
	cuint yidx = blockIdx.y;

	cuint idx = (c * xidx) + yidx;

	__shared__ nn_type sm[4];

	if (threadIdx.x == 0) {
		sm[0] = mean[yidx];
		sm[1] = var[yidx];
		sm[2] = alpha[yidx];
		sm[3] = gamma[yidx];
	}

	__syncthreads();

	if (xidx < n) {
		output[idx] = sm[2] * ((input[idx] - sm[0]) / __frsqrt_rn(sm[1] + EPSILON)) + sm[3];
	}
}


/**********************************************/
/*                                            */
/*				 NN_BatchNormalize            */
/*                                            */
/**********************************************/

NN_BatchNormalize::NN_BatchNormalize(const std::string& name) :
	NN_Layer(name)
{

}

void NN_BatchNormalize::get_output_shape(const NN_List<NN_Shape>& input_shape, NN_List<NN_Shape>& output_shape) {
	output_shape.append(input_shape[0].val());
}

void NN_BatchNormalize::build(const NN_List<NN_Shape>& input_shape, NN_List<GpuTensor<nn_type>>& weights) {
	const NN_Shape shape = input_shape[0].val();

	_means = GpuTensor<nn_type>::zeros({ shape[-1], });
	_var = GpuTensor<nn_type>::zeros({ shape[-1], });
	_alpha = GpuTensor<nn_type>::zeros({ shape[-1], });
	_gamma = GpuTensor<nn_type>::zeros({ shape[-1], });

	weights.append({ _means, _var, _alpha, _gamma });
}

void NN_BatchNormalize::run(NN_Stream& st, const NN_List<GpuTensor<nn_type>>& input, NN_List<GpuTensor<nn_type>>& output) {
	const GpuTensor<nn_type>& m_input = input[0].val();
	GpuTensor<nn_type>& m_output = output[0].val();

	const NN_Shape in_shape = m_input.get_shape();
	const nn_type* in_data = m_input.get_ptr();
	nn_type* out_data = m_output.get_ptr();
	const nn_type* mean_data = _means.get_ptr();
	const nn_type* var_data = _var.get_ptr();
	const nn_type* alpha_data = _alpha.get_ptr();
	const nn_type* gamma_data = _gamma.get_ptr();

	uint len = 1;
	for (int i = 1; i < in_shape.ranks() - 1; ++i) len *= in_shape[i];

	dim3 threads(BLOCK_1024);
	dim3 blocks = get_grid_size(threads, len, in_shape[-1]);
	cuint n = in_shape[0];
	cuint c = in_shape[-1];

	for (uint i = 0; i < n; ++i) {
		const nn_type* p_input = in_data + (len * c * i);
		nn_type* p_output = out_data + (len * c * i);

		__batch_normalization<<<blocks, threads, 0, st[i % STREAMS]>>>(
			p_input,
			mean_data,
			var_data,
			alpha_data,
			gamma_data,
			p_output,
			len,
			c
		);
	}

#if _DEBUG
	check_cuda(cudaDeviceSynchronize());
	check_cuda(cudaGetLastError());
#endif
}

NN_List<GpuTensor<nn_type>> NN_BatchNormalize::get_weight() {
	return { _means, _var, _alpha, _gamma };
}

void NN_BatchNormalize::set_output(const NN_List<NN_Shape>& output_shape, NN_List<GpuTensor<nn_type>>& input, NN_List<GpuTensor<nn_type>>& output) {
	output.append(GpuTensor<nn_type>(output_shape[0].val()));
}