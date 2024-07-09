#define CUDA_API_PER_THREAD_DEFAULT_STEAM 
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

void test_matmul(
	const GpuTensor<nn_type>& input,
	const GpuTensor<nn_type>& weight,
	const GpuTensor<nn_type>& bias,
	GpuTensor<nn_type>& output
) {
	const NC in = input.get_shape().get_nc();
	const NC out = output.get_shape().get_nc();

	dim3 threads(BLOCK_32, BLOCK_32);
	dim3 blocks = get_grid_size(threads, out.c, out.n);

	__matmul<<<blocks, threads>>>(
		input.get_ptr(),
		weight.get_ptr(),
		output.get_ptr(),
		in.n,
		in.c,
		out.c
	);

	check_cuda(cudaDeviceSynchronize());
	check_cuda(cudaGetLastError());

	add_bias_1d(output, bias, output);
}

NN_Dense::NN_Dense(const int amounts, const char* name) :
	NN_Layer(name),
	_amounts(amounts)
{
}

void NN_Dense::get_output_shape(const NN_List<NN_Shape>& input_shape, NN_List<NN_Shape>& output_shape) {
	const NN_Shape& shape = input_shape[0].val();

	output_shape.append(NN_Shape({ shape[0], _amounts }));
}

void NN_Dense::build(const NN_List<NN_Shape>& input_shape, NN_Link* p_node) {
	const NN_Shape& shape = input_shape[0].val();


	_weight = GpuTensor<nn_type>({ shape[1], _amounts });
	_bias = GpuTensor<nn_type>(NN_Shape({ _amounts }));
	set_random_uniform(_weight, -0.1f, 0.1f);

	Tensor<nn_type> tmp(NN_Shape({ _amounts }));
	tmp = 0.f;
	_bias = tmp;

	p_node->set_weights(_weight);
	p_node->set_weights(_bias);
}

void NN_Dense::run(NN_Stream& st, const NN_List<GpuTensor<nn_type>>& input, NN_List<GpuTensor<nn_type>>& output) {
	const GpuTensor<nn_type>& m_input = input[0].val();
	GpuTensor<nn_type>& m_output = output[0].val();

	const NC in = m_input.get_shape().get_nc();
	const NC out = m_output.get_shape().get_nc();

	dim3 threads(BLOCK_32, BLOCK_32);
	dim3 blocks = get_grid_size(threads, out.c, out.n);

	check_cuda(cudaDeviceSynchronize());
	check_cuda(cudaGetLastError());

	__matmul<<<blocks, threads>>>(
		m_input.get_ptr(),
		_weight.get_ptr(),
		m_output.get_ptr(),
		in.n,
		in.c,
		out.c
	);

	check_cuda(cudaDeviceSynchronize());
	check_cuda(cudaGetLastError());

	add_bias_1d(m_output, _bias, m_output);
}

NN_Backward* NN_Dense::create_backward(NN_Optimizer* optimizer) {
	return new NN_dDense(this, optimizer);
}

NN_List<GpuTensor<nn_type>> NN_Dense::get_weight() {
	return { _weight, _bias };
}


/**********************************************/
/*                                            */
/*                   NN_dDense                */
/*                                            */
/**********************************************/

NN_dDense::NN_dDense(NN_Dense* dense, NN_Optimizer* optimizer) :
	NN_Backward(optimizer),
	_dense(dense)
{

}

void NN_dDense::get_dinput_shape(const NN_List<NN_Shape>& dout_shape, NN_List<NN_Shape>& din_shape) {

}

void NN_dDense::run(
	NN_Stream& st,
	const NN_List<GpuTensor<nn_type>>& input,
	const NN_List<GpuTensor<nn_type>>& doutput,
	NN_List<GpuTensor<nn_type>>& dinput
) {

}