#include "matmul.cuh"
#include "cuda_misc.cuh"


#ifndef __CUDACC__
#define __CUDACC__
#endif

#include <device_functions.h>
#include <device_launch_parameters.h>


/**********************************************/
/*											  */
/*				 kernel function			  */
/*										      */
/**********************************************/

__global__ void __matmul(
	const nn_type* a,
	const nn_type* b,
	nn_type* c,
	cuint m,
	cuint k,
	cuint n
) {
	cuint cx = blockIdx.x * blockDim.x + threadIdx.x;
	cuint cy = blockIdx.y * blockDim.y + threadIdx.y;

	__shared__ nn_type sm_a[BLOCK_32 * BLOCK_32];
	__shared__ nn_type sm_b[BLOCK_32 * BLOCK_32];

	cuint tidx = threadIdx.y * BLOCK_32 + threadIdx.x;
	nn_type val = 0.f;

	for (uint i = 0; i < k; i += BLOCK_32) {
		__syncthreads();

		sm_a[tidx] = (threadIdx.x + i) < k && cy < m ? a[cy * k + (threadIdx.x + i)] : 0.f;
		sm_b[tidx] = cx < n && (threadIdx.y + i) < k ? b[(threadIdx.y + i) * n + cx] : 0.f;

		__syncthreads();

#pragma unroll
		for (uint e = 0; e < BLOCK_32; ++e) {
			val += sm_a[threadIdx.y * BLOCK_32 + e] * sm_b[e * BLOCK_32 + threadIdx.x];
		}
	}

	if (cx < n && cy < m) {
		c[cy * n + cx] = val;
	}
}


/**********************************************/
/*                                            */
/*                   NN_Dense                 */
/*                                            */
/**********************************************/

NN_Dense::NN_Dense(const int amounts, const std::string& name) :
	NN_Layer(name, "dense"),
	_amounts(amounts)
{
}

void NN_Dense::get_output_shape(const NN_List<NN_Shape>& input_shape, NN_List<NN_Shape>& output_shape) {
	const NN_Shape& shape = input_shape[0].val();

	output_shape.append(NN_Shape({ shape[0], _amounts }));
}

void NN_Dense::build(const NN_List<NN_Shape>& input_shape, NN_List<GpuTensor<nn_type>>& weights) {
	const NN_Shape& shape = input_shape[0].val();


	_weight = GpuTensor<nn_type>(NN_Shape({ shape[1], _amounts }));
	_bias = GpuTensor<nn_type>::zeros(NN_Shape({ _amounts }));
	set_random_uniform(_weight, -0.1f, 0.1f);

	weights.append(_weight);
	weights.append(_bias);
}

void NN_Dense::run(NN_Stream& st, const NN_List<GpuTensor<nn_type>>& input, NN_List<GpuTensor<nn_type>>& output) {
	const GpuTensor<nn_type>& m_input = input[0].val();
	GpuTensor<nn_type>& m_output = output[0].val();

	const NN_Shape& in = m_input.get_shape();
	const NN_Shape& out = m_output.get_shape();

	dim3 threads(BLOCK_32, BLOCK_32);
	dim3 blocks = get_grid_size(threads, out[1], out[0]);

	__matmul<<<blocks, threads>>>(
		m_input.get_ptr(),
		_weight.get_ptr(),
		m_output.get_ptr(),
		(uint)in[0],
		(uint)in[1],
		(uint)out[1]
	);
#if _DEBUG
	check_cuda(cudaDeviceSynchronize());
	check_cuda(cudaGetLastError());
#endif
	add_bias_1d(m_output, _bias, m_output);
}

NN_Backward* NN_Dense::create_backward(std::vector<bool>& mask) {
	return new NN_dDense(*this);
}

NN_List<GpuTensor<nn_type>> NN_Dense::get_weight() {
	return { _weight, _bias };
}


/**********************************************/
/*                                            */
/*                   NN_dDense                */
/*                                            */
/**********************************************/

NN_dDense::NN_dDense(NN_Dense& dense) :
	NN_Backward_t(dense)
{
}

void NN_dDense::run(
	NN_Stream& st,
	const NN_List<GpuTensor<nn_type>>& input,
	const NN_List<GpuTensor<nn_type>>& doutput,
	NN_List<GpuTensor<nn_type>>& dinput
) {

}

NN_Optimizer* NN_dDense::create_optimizer(const NN_Optimizer& optimizer) {
	return optimizer.create({ _layer._weight, _layer._bias });
}