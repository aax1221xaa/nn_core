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
	const Tensor* gradient,
	const Tensor* momentum,
	const Tensor* weight
) {
	size_t g_size = GetTotalSize(gradient);
	size_t m_size = GetTotalSize(momentum);
	size_t w_size = GetTotalSize(weight);

	if (w_size != g_size || w_size != m_size) {
		ErrorExcept(
			"[check_sgd] invalid size. \
			gradient = %zd, momentum = %zd, weight = %zd",
			g_size, m_size, w_size
		);
	}
}

void check_rms_prop(
	const Tensor* gradient,
	const Tensor* g,
	const Tensor* weight
) {
	size_t g_size = GetTotalSize(gradient);
	size_t m_size = GetTotalSize(g);
	size_t w_size = GetTotalSize(weight);

	if (w_size != g_size || w_size != m_size) {
		ErrorExcept(
			"[check_rms_prop] invalid size. \
			gradient = %zd, momentum = %zd, weight = %zd",
			g_size, m_size, w_size
		);
	}
}

void check_adam(
	const Tensor* gradient,
	const Tensor* square_g,
	const Tensor* decay_avg,
	const Tensor* weight
) {
	size_t g_size = GetTotalSize(gradient);
	size_t sqg_size = GetTotalSize(square_g);
	size_t davg_size = GetTotalSize(decay_avg);
	size_t w_size = GetTotalSize(weight);

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
	Stream* stream,
	const Tensor* gradient,
	Tensor* momentum,
	Tensor* weight,
	float learn_rate,
	float momentum_rate
) {
	check_sgd(gradient, momentum, weight);

	uint length = weight->n * weight->h * weight->w * weight->c;
	dim3 threads(BLOCK_SIZE * BLOCK_SIZE);
	dim3 blocks((length + threads.x - 1) / threads.x);

	__sgd<<<blocks, threads, 0, stream->st[0]>>>(
		gradient->data,
		weight->data,
		momentum->data,
		length,
		learn_rate,
		momentum_rate
	);
	checkCuda(cudaStreamSynchronize(stream->st[0]));
}

void rms_prop(
	Stream* stream,
	const Tensor* gradient,
	Tensor* g,
	Tensor* weight,
	float decay_rate,
	float learn_rate
) {
	check_rms_prop(gradient, g, weight);

	uint length = weight->n * weight->h * weight->w * weight->c;
	dim3 threads(BLOCK_SIZE * BLOCK_SIZE);
	dim3 blocks((length + threads.x - 1) / threads.x);

	__rms_prop<<<blocks, threads, 0, stream->st[0]>>>(
		gradient->data,
		weight->data,
		g->data,
		length,
		learn_rate,
		decay_rate
	);
	checkCuda(cudaStreamSynchronize(stream->st[0]));
}

void adam(
	Stream* stream,
	const Tensor* gradient,
	Tensor* square_g,
	Tensor* decay_g,
	Tensor* weight,
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

	uint length = weight->n * weight->h * weight->w * weight->c;
	dim3 threads(BLOCK_SIZE * BLOCK_SIZE);
	dim3 blocks((length + threads.x - 1) / threads.x);

	__adam << <blocks, threads, 0, stream->st[0] >> > (
		gradient->data,
		weight->data,
		square_g->data,
		decay_g->data,
		length,
		learn_rate,
		beta_1,
		beta_2
	);
	checkCuda(cudaStreamSynchronize(stream->st[0]));
}