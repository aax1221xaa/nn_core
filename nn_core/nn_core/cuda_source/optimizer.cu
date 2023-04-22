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

void check_sgd(
	const CudaTensor gradient,
	const CudaTensor momentum,
	const CudaTensor weight
) {
	uint g_size = get_elem_size(gradient);
	uint m_size = get_elem_size(momentum);
	uint w_size = get_elem_size(weight);

	if (w_size != g_size || w_size != m_size) {
		ErrorExcept(
			"[check_sgd] invalid size. \
			gradient = %zd, momentum = %zd, weight = %zd",
			g_size, m_size, w_size
		);
	}
}

void check_rms_prop(
	const CudaTensor gradient,
	const CudaTensor g,
	const CudaTensor weight
) {
	uint g_size = get_elem_size(gradient);
	uint m_size = get_elem_size(g);
	uint w_size = get_elem_size(weight);

	if (w_size != g_size || w_size != m_size) {
		ErrorExcept(
			"[check_rms_prop] invalid size. \
			gradient = %zd, momentum = %zd, weight = %zd",
			g_size, m_size, w_size
		);
	}
}

void check_adam(
	const CudaTensor gradient,
	const CudaTensor square_g,
	const CudaTensor decay_avg,
	const CudaTensor weight
) {
	uint g_size = get_elem_size(gradient);
	uint sqg_size = get_elem_size(square_g);
	uint davg_size = get_elem_size(decay_avg);
	uint w_size = get_elem_size(weight);

	if (w_size != g_size || w_size != sqg_size || w_size != davg_size) {
		ErrorExcept(
			"[check_adam] invalid size. \
			gradient = %zd, square_gradient = %zd, decay_avg = %zd, weight = %zd",
			g_size,
			sqg_size,
			davg_size,
			w_size
		);
	}
}

void sgd(
	cudaStream_t stream,
	const CudaTensor gradient,
	CudaTensor momentum,
	CudaTensor weight,
	float learn_rate,
	float momentum_rate
) {
	check_sgd(gradient, momentum, weight);

	uint length = get_elem_size(weight);
	dim3 threads(SQR_BLOCK_SIZE);
	dim3 blocks = get_grid_size(threads, length);

	__sgd << <blocks, threads, 0, stream >> > (
		gradient.data,
		weight.data,
		momentum.data,
		length,
		learn_rate,
		momentum_rate
	);
	check_cuda(cudaStreamSynchronize(stream));
}

void rms_prop(
	cudaStream_t stream,
	const CudaTensor gradient,
	CudaTensor square_g,
	CudaTensor weight,
	float decay_rate,
	float learn_rate
) {
	check_rms_prop(gradient, square_g, weight);

	uint length = get_elem_size(weight);
	dim3 threads(SQR_BLOCK_SIZE);
	dim3 blocks = get_grid_size(threads, length);

	__rms_prop<<<blocks, threads, 0, stream>>>(
		gradient.data,
		weight.data,
		square_g.data,
		length,
		learn_rate,
		decay_rate
	);
	check_cuda(cudaStreamSynchronize(stream));
}

void adam(
	cudaStream_t stream,
	const CudaTensor gradient,
	CudaTensor square_g,
	CudaTensor decay_g,
	CudaTensor weight,
	float learn_rate,
	float beta_1,
	float beta_2
) {
	check_adam(
		gradient,
		square_g,
		decay_g,
		weight
	);

	uint length = get_elem_size(weight);
	dim3 threads(SQR_BLOCK_SIZE);
	dim3 blocks = get_grid_size(threads, length);

	__adam<<<blocks, threads, 0, stream>>>(
		gradient.data,
		weight.data,
		square_g.data,
		decay_g.data,
		length,
		learn_rate,
		beta_1,
		beta_2
	);
	check_cuda(cudaStreamSynchronize(stream));
}