#include "dens.cuh"
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
	const float* a,
	const float* b,
	float* c,
	const uint m,
	const uint n,
	const uint k
) {
	uint cx = blockIdx.x * blockDim.x + threadIdx.x;
	uint cy = blockIdx.y * blockDim.y + threadIdx.y;

	__shared__ float sm_a[BLOCK_SIZE * BLOCK_SIZE];
	__shared__ float sm_b[BLOCK_SIZE * BLOCK_SIZE];

	uint tidx = threadIdx.y * BLOCK_SIZE + threadIdx.x;
	float val = 0.f;

	for (int i = 0; i < n; i += BLOCK_SIZE) {
		__syncthreads();

		sm_a[tidx] = (threadIdx.x + i) < n && cy < m ? a[cy * n + (threadIdx.x + i)] : 0.f;
		sm_b[tidx] = cx < k && (threadIdx.y + i) < n ? b[(threadIdx.y + i) * k + cx] : 0.f;

		__syncthreads();

#pragma unroll
		for (int e = 0; e < BLOCK_SIZE; ++e) {
			val += sm_a[threadIdx.y * BLOCK_SIZE + e] * sm_b[e * BLOCK_SIZE + threadIdx.x];
		}
	}

	if (cx < k && cy < m) {
		c[cy * k + cx] = val;
	}
}

/**********************************************

				  denseSolution

**********************************************/

denseSolution::denseSolution(const tensor4d& input, const tensor4d& weight) :
	_input(input),
	_weight(weight)
{
}

const tensor4d denseSolution::calculate_size() {
	if (!_input.is_valid() || !_weight.is_valid() || _input._c != _weight._n) {
		ErrorExcept(
			"[denseSolution::calculate_size()] invalid arguments. input: %s, weight: %s",
			tensor4d::shape_to_str(_input),
			tensor4d::shape_to_str(_weight)
		);
	}

	int out_n = _input._n;
	int out_c = _weight._c;
	_output.set(out_n, out_c, 1, 1);

	if (!_output.is_valid()) {
		ErrorExcept(
			"[densSolution::calculate_size()] missmatched output size. input: %s, weight: %s, output: %s",
			tensor4d::shape_to_str(_input),
			tensor4d::shape_to_str(_weight),
			tensor4d::shape_to_str(_output)
		);
	}

	return _output;
}

void denseSolution::operator()(const nn_type* input, const nn_type* weight, nn_type* output) {
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocks = get_grid_size(threads, _output._c, _output._n);

	__matmul<<<blocks, threads>>>(
		input,
		weight,
		output,
		_input._n,
		_weight._n,
		_output._c
	);
}

/**********************************************

				  dDenseSolution

**********************************************/

dDenseSolution::dDenseSolution(const tensor4d& d_output, const denseSolution& dense) :
	_d_output(d_output),
	_dense(dense)
{
}

const tensor4d dDenseSolution::calcuiate_size() {
	if (!_d_output.is_valid() || _d_output._c != _dense._weight._c) {
		ErrorExcept(
			"[dDenseSolution::calculate_size()] invalid arguments. d_output: %s, weight: %s",
			tensor4d::shape_to_str(_d_output),
			tensor4d::shape_to_str(_dense._weight)
		);
	}
	_is_calculated = true;

	return _dense._input;
}

const size_t dDenseSolution::get_workspace_size() {
	if (!_is_calculated) {
		ErrorExcept(
			"[dDenseSolution::get_workspace_size()] not calculated sizes."
		);
	}

	return _dense._weight.get_size();
}

void dDenseSolution::operator()(const nn_type* d_output, const nn_type* weight, nn_type* d_input, void* workspace) {
	if (!_is_calculated) {
		ErrorExcept(
			"[dDenseSolution::get_workspace_size()] not calculated sizes."
		);
	}
	/*
	dout [64, 10]
	kernel [128 10]
	din [64, 128]
	*/

	nn_type* t_weight = (nn_type*)workspace;

	transpose(_dense._weight, weight, t_weight);

	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocks = get_grid_size(threads, _dense._input._c, _dense._input._n);

	__matmul<<<blocks, threads>>>(
		d_output,
		t_weight,
		d_input,
		_d_output._n,
		_dense._weight._c,
		_dense._input._c
	);
}

/**********************************************

				  wDenseSolution

**********************************************/

wDenseSolution::wDenseSolution(const dDenseSolution& d_dense) :
	_d_dense(d_dense)
{
}

const size_t wDenseSolution::get_workspace_size() {
	return _d_dense._dense._input.get_size();
}

void wDenseSolution::operator()(const nn_type* d_output, const nn_type* input, nn_type* gradient, void* workspace) {
	/*
	input = [64, 128]
	kernel = [128, 10]
	output = [64, 10]

	d_output = [64, 10]
	t_input = [128, 64]
	gradient = [128, 10]
	
	gradient = t_input * d_output
	*/

	nn_type* t_input = (nn_type*)workspace;

	transpose(_d_dense._dense._input, input, t_input);

	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocks = get_grid_size(threads, _d_dense._dense._weight._c, _d_dense._dense._weight._n);

	__matmul<<<blocks, threads>>>(
		t_input, 
		d_output, 
		gradient, 
		_d_dense._dense._input._c, 
		_d_dense._dense._weight._n, 
		_d_dense._dense._weight._c
	);
}