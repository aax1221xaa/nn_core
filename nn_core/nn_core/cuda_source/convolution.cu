#include "convolution.cuh"
#include "cuda_indice.cuh"


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

__global__ void __conv2d(
	const uint* indice,
	const float* input,
	const float* kernel,
	float* output,
	cuint in_h,
	cuint in_w,
	cuint k_n,
	cuint k_c,
	cuint k_h,
	cuint k_w,
	cuint out_h,
	cuint out_w,
	cuint st_h,
	cuint st_w
) {
	cuint cx = blockIdx.x * blockDim.x + threadIdx.x;
	cuint cy = blockIdx.y * blockDim.y + threadIdx.y;
	cuint sidx = threadIdx.y * BLOCK_32 + threadIdx.x;

	cuint x0 = (cx % out_w) * st_w;
	cuint y0 = (cx / out_w) * st_h;

	cuint n = k_w * k_h * k_c;
	cuint k = out_w * out_h;

	__shared__ float share_in[BLOCK_32 * BLOCK_32];
	__shared__ float share_k[BLOCK_32 * BLOCK_32];

	const float* p_input = input + (y0 * in_w + x0);
	const float* p_kernel = kernel + (cy * k_w * k_h * k_c);

	float sum = 0.f;

	for (uint i = 0; i < n; i += BLOCK_32) {
		uint th_x = i + threadIdx.x;
		uint th_y = i + threadIdx.y;

		__syncthreads();

		share_k[sidx] = th_x < n && cy < k_n ? p_kernel[th_x] : 0.f;
		share_in[sidx] = cx < k && th_y < n ? p_input[indice[th_y]] : 0.f;

		__syncthreads();

#pragma unroll
		for (uint e = 0; e < BLOCK_32; ++e) {
			sum += share_in[e * BLOCK_32 + threadIdx.x] * share_k[threadIdx.y * BLOCK_32 + e];
		}
	}

	if (cx < k && cy < k_n) {
		output[cy * k + cx] = sum;
	}
}

__global__ void __kernel_conv2d(
	const uint* indice,
	const float* d_output,
	const float* input,
	float* gradient,
	cuint d_output_c,
	cuint d_output_h,
	cuint d_output_w,
	cuint input_c,
	cuint input_h,
	cuint input_w,
	cuint gradient_h,
	cuint gradient_w
) {
	cuint n = d_output_h * d_output_w;
	cuint k = gradient_h * gradient_w * input_c;

	cuint cx = blockIdx.x * blockDim.x + threadIdx.x;
	cuint cy = blockIdx.y * blockDim.y + threadIdx.y;

	cuint x0 = cx % gradient_w;
	cuint y0 = (cx / gradient_w) % gradient_h;
	cuint c0 = cx / (gradient_h * gradient_w);

	cuint sidx = threadIdx.y * BLOCK_32 + threadIdx.x;

	__shared__ float sm_in[BLOCK_32 * BLOCK_32];
	__shared__ float sm_dout[BLOCK_32 * BLOCK_32];

	const float* p_dout = d_output + (cy * d_output_h * d_output_w);
	const float* p_in = input + (c0 * (input_h * input_w) + y0 * input_w + x0);

	float sum = 0.f;

	for (int i = 0; i < n; i += BLOCK_32) {
		cuint th_x = threadIdx.x + i;
		cuint th_y = threadIdx.y + i;

		__syncthreads();

		sm_dout[sidx] = th_x < n && cy < d_output_c ? p_dout[th_x] : 0.f;
		sm_in[sidx] = cx < k && th_y < n ? p_in[indice[th_y]] : 0.f;

		__syncthreads();

#pragma unroll
		for (int e = 0; e < BLOCK_32; ++e) {
			sum += sm_dout[threadIdx.y * BLOCK_32 + e] * sm_in[e * BLOCK_32 + threadIdx.x];
		}
	}

	if (cx < k && cy < d_output_c) {
		gradient[cy * k + cx] += sum;
	}
}

/**********************************************

				    Conv2d

**********************************************/

void conv2d(
	cudaStream_t s,
	cuint* indice,
	const nn_type* input,
	const nn_type* kernel,
	nn_type* output,
	const nn_shape& in_shape,
	const nn_shape& k_shape,
	const nn_shape& out_shape,
	cuint h_stride,
	cuint w_stride
) {
	dim3 threads(BLOCK_32, BLOCK_32);
	dim3 blocks = get_grid_size(threads, out_shape[2] * out_shape[3], out_shape[1]);

	__conv2d<<<blocks, threads, 0, s>>>(
		indice,
		input,
		kernel,
		output,
		in_shape[2],
		in_shape[3],
		k_shape[0],
		k_shape[1],
		k_shape[2],
		k_shape[3],
		out_shape[2],
		out_shape[3],
		h_stride,
		w_stride
	);
}

/**********************************************

			     KernelConv2d

**********************************************/

void kernel_conv2d(
	const nn_type* d_output,
	const nn_type* input,
	nn_type* grad,
	const nn_shape& out_shape,
	const nn_shape& in_shape,
	const nn_shape& grad_shape
) {
	cint hw = out_shape[2] * out_shape[3];
	uint* indice = NULL;
	uint* _indice = new uint[hw];

	for (int h = 0; h < out_shape[2]; ++h) {
		for (int w = 0; w < out_shape[3]; ++w) {
			_indice[h * out_shape[3] + w] = h * in_shape[3] + w;
		}
	}

	if (hw > CONST_ELEM_SIZE) {
		check_cuda(cudaMalloc(&indice, sizeof(uint) * hw));
		check_cuda(cudaMemcpy(indice, _indice, sizeof(uint) * hw, cudaMemcpyHostToDevice));
	}
	else {
		indice = get_indice_ptr();
		set_indice(_indice, sizeof(uint) * hw, 0);
	}
	delete[] _indice;

	dim3 threads(BLOCK_32, BLOCK_32);
	dim3 blocks = get_grid_size(threads, grad_shape[1] * grad_shape[2] * grad_shape[3], grad_shape[0]);

	for (int i = 0; i < out_shape[0]; ++i) {
		const nn_type* d_dout = d_output + (i * out_shape[1] * out_shape[2] * out_shape[3]);
		const nn_type* d_input = input + (i * in_shape[1] * in_shape[2] * in_shape[3]);

		__kernel_conv2d<<<blocks, threads>>>(
			indice,
			d_dout,
			d_input,
			grad,
			out_shape[1],
			out_shape[2],
			out_shape[3],
			in_shape[1],
			in_shape[2],
			in_shape[3],
			grad_shape[2],
			grad_shape[3]
		);
	}

	if (hw > CONST_ELEM_SIZE) check_cuda(cudaFree(indice));
}