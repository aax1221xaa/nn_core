#include "convolution.cuh"

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

__global__ void __conv_2d(
	float* input,
	float* kernel,
	float* output,
	cuint in_w,
	cuint k_n,
	cuint k_w,
	cuint k_h,
	cuint k_c,
	cuint out_w,
	cuint out_h,
	cuint st_w,
	cuint st_h
) {
	cuint cx = blockIdx.x * blockDim.x + threadIdx.x;
	cuint cy = blockIdx.y * blockDim.y + threadIdx.y;
	cuint sidx = threadIdx.y * BLOCK_SIZE + threadIdx.x;

	cuint x0 = (cx % out_w) * st_w;
	cuint y0 = (cx / out_w) * st_h;

	cuint n = k_w * k_h * k_c;
	cuint k = out_w * out_h;

	__shared__ float share_in[BLOCK_SIZE * BLOCK_SIZE];
	__shared__ float share_k[BLOCK_SIZE * BLOCK_SIZE];

	float* p_input = input + (y0 * in_w + x0);
	float* p_kernel = kernel + (cy * k_w * k_h * k_c);

	float sum = 0.f;

	for (uint i = 0; i < n; i += BLOCK_SIZE) {
		__syncthreads();

		share_k[sidx] = (i + threadIdx.x) < n && cy < k_n ? p_kernel[threadIdx.x + i] : 0.f;
		share_in[sidx] = cx < k && (threadIdx.y + i) < n ? p_input[__indices[threadIdx.y + i]] : 0.f;

		__syncthreads();

		#pragma unroll
		for (uint e = 0; e < BLOCK_SIZE; ++e) {
			sum += share_in[e * BLOCK_SIZE + threadIdx.x] * share_k[threadIdx.y * BLOCK_SIZE + e];
		}
	}

	if (cx < k && cy < k_n) {
		output[cy * k + cx] = sum;
	}
}



/**********************************************/
/*											  */
/*				  host function 			  */
/*										      */
/**********************************************/

uint get_output_size(
	int input_size,
	int kernel_size,
	int pad_size,
	int stride
) {
	return (input_size + (2 * pad_size) - kernel_size) / stride + 1;
}

void check_conv_2d(
	const Tensor* d_input,
	const Tensor* d_kernel,
	const Tensor* d_output,
	int st_w,
	int st_h
) {
	int out_h = get_output_size(d_input->h, d_kernel->h, 0, st_h);
	int out_w = get_output_size(d_input->w, d_kernel->w, 0, st_w);

	if (d_output->h != out_h || d_output->w != out_w) {
		ErrorExcept(
			"[conv_2d] invalid output size [%d, %d, %d, %d]",
			d_output->n,
			d_output->h,
			d_output->w,
			d_output->c
		);
	}
	else if (d_kernel->c != d_input->c || d_kernel->n != d_output->c) {
		ErrorExcept(
			"[conv_2d] invalid channels input:[%d, %d, %d, %d], kernel:[%d, %d, %d, %d], output:[%d, %d, %d, %d]",
			d_input->n,
			d_input->h,
			d_input->w,
			d_input->c,
			d_kernel->n,
			d_kernel->h,
			d_kernel->w,
			d_kernel->c,
			d_output->n,
			d_output->h,
			d_output->w,
			d_output->c
		);
	}
}

void conv_2d(
	const Stream* stream,
	const Tensor* d_input,
	const Tensor* d_kernel,
	Tensor* d_output,
	int st_w,
	int st_h
) {
	uint m_indices[CONST_SIZE] = { 0, };

	check_conv_2d(
		d_input,
		d_kernel,
		d_output,
		st_w,
		st_h
	);

	for (int c = 0; c < d_kernel->c; ++c) {
		for (int i = 0; i < d_kernel->h; ++i) {
			for (int j = 0; j < d_kernel->w; ++j) {
				m_indices[c * (d_kernel->h * d_kernel->w) + i * d_kernel->w + j] = c * (d_input->w * d_input->h) + i * d_input->w + j;
			}
		}
	}

	checkCuda(cudaMemcpyToSymbol(__indices, m_indices, sizeof(m_indices)));

	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocks(
		GetBlockSize(d_output->w * d_output->h),
		GetBlockSize(d_output->c)
	);

	for (int i = 0; i < stream->st_size; ++i) {
		float* d_in = d_input->data + (i * d_input->w * d_input->h * d_input->c);
		float* d_out = d_output->data + (i * d_output->w * d_output->h * d_output->c);

		__conv_2d<<<blocks, threads, 0, stream->st[i]>>>(
			d_in,
			d_kernel->data,
			d_out,
			d_input->w,
			d_kernel->n,
			d_kernel->w,
			d_kernel->h,
			d_kernel->c,
			d_output->w,
			d_output->h,
			st_w,
			st_h
			);
	}
	SyncStreams(stream);
}
