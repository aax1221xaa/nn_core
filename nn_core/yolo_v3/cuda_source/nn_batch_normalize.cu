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
	const nn_type* beta,
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
		sm[2] = beta[yidx];
		sm[3] = gamma[yidx];
	}

	__syncthreads();

	if (xidx < n) {
		output[idx] = sm[3] * ((input[idx] - sm[0]) / __powf(sm[1] + EPSILON, 0.5f)) + sm[2];
	}
}


/**********************************************/
/*                                            */
/*				 NN_BatchNormalize            */
/*                                            */
/**********************************************/

NN_BatchNormalize::NN_BatchNormalize(const std::string& name) :
	NN_Layer(name, "batch_normalization")
{

}

void NN_BatchNormalize::get_output_shape(const NN_List<NN_Shape>& input_shape, NN_List<NN_Shape>& output_shape) {
	output_shape.append(input_shape[0].val());
}

void NN_BatchNormalize::build(const NN_List<NN_Shape>& input_shape, NN_List<GpuTensor<nn_type>>& weights) {
	const NN_Shape shape = input_shape[0].val();

	_means = GpuTensor<nn_type>::zeros({ shape[-1], });
	_var = GpuTensor<nn_type>::zeros({ shape[-1], });
	_beta = GpuTensor<nn_type>::zeros({ shape[-1], });
	_gamma = GpuTensor<nn_type>::zeros({ shape[-1], });

	weights.append({ _gamma, _beta, _means, _var });
}

void NN_BatchNormalize::run(NN_Stream& st, const NN_List<GpuTensor<nn_type>>& input, NN_List<GpuTensor<nn_type>>& output) {
	const GpuTensor<nn_type>& m_input = input[0].val();
	GpuTensor<nn_type>& m_output = output[0].val();
#if 0
	const Tensor<nn_type> h_input = m_input;
	const Tensor<nn_type> h_gamma = _gamma;
	const Tensor<nn_type> h_beta = _beta;
	const Tensor<nn_type> h_means = _means;
	const Tensor<nn_type> h_var = _var;
	Tensor<nn_type> h_output(m_output.get_shape());
	
	const NN_Tensor4dShape in_shape = h_input.get_shape().get_4d_shape();
	const nn_type* in_data = h_input.get_ptr();
	const nn_type* gamma_data = h_gamma.get_ptr();
	const nn_type* beta_data = h_beta.get_ptr();
	const nn_type* mean_data = h_means.get_ptr();
	const nn_type* var_data = h_var.get_ptr();
	nn_type* out_data = h_output.get_ptr();

	for (int i = 0; i < in_shape._n; ++i) {
		const nn_type* in_ptr = in_data + (in_shape._c * in_shape._h * in_shape._w * i);
		nn_type* out_ptr = out_data + (in_shape._c * in_shape._h * in_shape._w * i);

		for (int j = 0; j < in_shape._c; ++j) {
			const nn_type gamma = gamma_data[j];
			const nn_type beta = beta_data[j];
			const nn_type mean = mean_data[j];
			const nn_type var = var_data[j];
			const nn_type* in_c = in_ptr + j;
			nn_type* out_c = out_ptr + j;

			for (int h = 0; h < in_shape._h; ++h) {
				const nn_type* in_h = in_c + (in_shape._w * in_shape._c * h);
				nn_type* out_h = out_c + (in_shape._w * in_shape._c * h);

				for (int w = 0; w < in_shape._w; ++w) {
					const nn_type& in = in_h[in_shape._c * w];
					nn_type& out = out_h[in_shape._c * w];

					out = gamma * ((in - mean) / sqrtf(var + EPSILON)) + beta;
				}
			}
		}
	}

	m_output = h_output;

#else
	const NN_Shape in_shape = m_input.get_shape();
	const nn_type* in_data = m_input.get_ptr();
	nn_type* out_data = m_output.get_ptr();
	const nn_type* mean_data = _means.get_ptr();
	const nn_type* var_data = _var.get_ptr();
	const nn_type* beta_data = _beta.get_ptr();
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
			beta_data,
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
#endif
}

NN_List<GpuTensor<nn_type>> NN_BatchNormalize::get_weight() {
	return { _gamma, _beta, _means, _var };
}

void NN_BatchNormalize::set_output(const NN_List<NN_Shape>& output_shape, NN_List<GpuTensor<nn_type>>& input, NN_List<GpuTensor<nn_type>>& output) {
	output.append(GpuTensor<nn_type>(output_shape[0].val()));
}