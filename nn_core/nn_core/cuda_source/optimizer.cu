#define CUDA_API_PER_THREAD_DEFAULT_STEAM 
#include "optimizer.cuh"

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

__global__ void __sgd(
	float* gradient,
	float* weight,
	float* w_momentum,
	cuint w_len,
	float learn_rate,
	float momentum_rate
) {
	cuint idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < w_len) {
		float m = momentum_rate * w_momentum[idx] + learn_rate * gradient[idx];
		weight[idx] -= m;
		w_momentum[idx] = m;
	}
}

__global__ void __rms_prop(
	float* gradient,
	float* weight,
	float* g,
	cuint w_len,
	float learn_rate,
	float decay_rate
) {
	cuint idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < w_len) {
		float grad = gradient[idx];
		float _g = decay_rate * g[idx] + (1 - decay_rate) * __powf(grad, 2.f);
		
		weight[idx] -= learn_rate / __powf(_g + EPSILON, 0.5f) * grad;
		g[idx] = _g;
	}
}

__global__ void __adam(
	float* gradient,
	float* weight,
	float* square_g,
	float* decay_g,
	cuint w_len,
	float learn_rate,
	float beta_1,
	float beta_2
) {
	cuint idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < w_len) {
		float grad = gradient[idx];
		float m = beta_1 * decay_g[idx] + (1 - beta_1) * grad;
		float v = beta_2 * square_g[idx] + (1 - beta_2) * __powf(grad, 2.f);
		
		float _m = m / (1 - beta_1);
		float _v = v / (1 - beta_2);

		weight[idx] -= learn_rate / (_v + EPSILON) * _m;
		decay_g[idx] = m;
		square_g[idx] = v;
	}
}


/**********************************************/
/*											  */
/*				  host function 			  */
/*										      */
/**********************************************/

void sgd(
	nn_type* gradient,
	nn_type* momentum,
	nn_type* weight,
	const uint len,
	float learn_rate,
	float momentum_rate
) {
	dim3 threads(BLOCK_1024);
	dim3 blocks = get_grid_size(threads, len);

	__sgd<<<blocks, threads>>>(
		gradient,
		weight,
		momentum,
		len,
		learn_rate,
		momentum_rate
	);
}

void rms_prop(
	nn_type* gradient,
	nn_type* square_g,
	nn_type* weight,
	const uint len,
	float decay_rate,
	float learn_rate
) {
	dim3 threads(BLOCK_1024);
	dim3 blocks = get_grid_size(threads, len);

	__rms_prop<<<blocks, threads>>>(
		gradient,
		weight,
		square_g,
		len,
		learn_rate,
		decay_rate
	);
}

void adam(
	nn_type* gradient,
	nn_type* square_g,
	nn_type* decay_g,
	nn_type* weight,
	const uint len,
	float learn_rate,
	float beta_1,
	float beta_2
) {
	dim3 threads(BLOCK_1024);
	dim3 blocks = get_grid_size(threads, len);

	__adam<<<blocks, threads>>>(
		gradient,
		weight,
		square_g,
		decay_g,
		len,
		learn_rate,
		beta_1,
		beta_2
	);
}