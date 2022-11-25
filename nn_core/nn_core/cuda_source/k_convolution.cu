#include "k_convolution.cuh"
#include "convolution.cuh"


#ifndef __CUDACC__
#define __CUDACC__
#endif

#include <device_functions.h>
#include <device_launch_parameters.h>

#if false

/**********************************************/
/*											  */
/*				 kernel function			  */
/*										      */
/**********************************************/

__constant__ uint __indices[CONST_ELEM_SIZE];

__global__ void __kernel_conv_2d_32x32_c_ind(
	float* input,
	float* d_output,
	float* gradient,
	cuint input_h,
	cuint input_w,
	cuint input_c,
	cuint d_output_h,
	cuint d_output_w,
	cuint d_output_c,
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

	cuint sidx = threadIdx.y * BLOCK_SIZE_32 + threadIdx.x;

	__shared__ float sm_in[BLOCK_SIZE_32 * BLOCK_SIZE_32];
	__shared__ float sm_dout[BLOCK_SIZE_32 * BLOCK_SIZE_32];

	float* p_dout = d_output + (cy * d_output_h * d_output_w);
	float* p_in = input + (c0 * (input_h * input_w) + y0 * input_w + x0);
	
	float sum = 0.f;

	for (int i = 0; i < n; i += BLOCK_SIZE_32) {
		__syncthreads();

		sm_dout[sidx] = (threadIdx.x + i) < n && cy < d_output_c ? p_dout[threadIdx.x + i] : 0.f;
		sm_in[sidx] = cx < k && (threadIdx.y + i) < n ? p_in[__indices[threadIdx.y + i]] : 0.f;

		__syncthreads();

		#pragma unroll
		for (int e = 0; e < BLOCK_SIZE_32; ++e) {
			sum += sm_dout[threadIdx.y * BLOCK_SIZE_32 + e] * sm_in[e * BLOCK_SIZE_32 + threadIdx.x];
		}
	}

	if (cx < k && cy < d_output_c) {
		gradient[cy * k + cx] += sum;
	}
}

__global__ void __kernel_conv_2d_32x32_g_ind(
	float* input,
	float* d_output,
	float* gradient,
	uint* input_indices,
	cuint input_h,
	cuint input_w,
	cuint input_c,
	cuint d_output_h,
	cuint d_output_w,
	cuint d_output_c,
	cuint gradient_h,
	cuint gradient_w
) {
	cuint n = d_output_h * d_output_w;
	cuint k = gradient_h * gradient_w * input_c;

	cuint cx = blockIdx.x + blockDim.x + threadIdx.x;
	cuint cy = blockIdx.y + blockDim.y + threadIdx.y;

	cuint x0 = cx % gradient_w;
	cuint y0 = (cx / gradient_w) % gradient_h;
	cuint c0 = cx / (gradient_h * gradient_w);

	cuint sidx = threadIdx.y * BLOCK_SIZE_32 + threadIdx.x;

	__shared__ float sm_in[BLOCK_SIZE_32 * BLOCK_SIZE_32];
	__shared__ float sm_dout[BLOCK_SIZE_32 * BLOCK_SIZE_32];

	float* p_dout = d_output + (cy * d_output_h * d_output_w);
	float* p_in = input + (c0 * (input_h * input_w) + y0 * input_w + x0);

	float sum = 0.f;

	for (int i = 0; i < n; i += BLOCK_SIZE_32) {
		__syncthreads();

		sm_dout[sidx] = (threadIdx.x + i) < n && cy < d_output_c ? p_dout[threadIdx.x + i] : 0.f;
		sm_in[sidx] = cx < k && (threadIdx.y + i) < n ? p_in[input_indices[threadIdx.y + i]] : 0.f;

		__syncthreads();

		#pragma unroll
		for (int e = 0; e < BLOCK_SIZE_32; ++e) {
			sum += sm_dout[threadIdx.y * BLOCK_SIZE_32 + e] * sm_in[e * BLOCK_SIZE_32 + threadIdx.x];
		}
	}

	if (cx < k && cy < n) {
		gradient[cy * k + cx] += sum;
	}
}



/**********************************************/
/*											  */
/*				  host function 			  */
/*										      */
/**********************************************/

void check_kernel_conv_2d(
	const Tensor& d_input,
	const Tensor& d_doutput,
	Tensor& d_gradient
) {
	int in_h = d_input.h - d_doutput.h + 1;
	int in_w = d_input.w - d_doutput.w + 1;

	if (d_gradient.h != in_h || d_gradient.w != in_w || d_gradient.n != d_doutput.c || d_gradient.c != d_input.c || d_input.n != d_doutput.n) {
		ErrorExcept(
			"[check_kernel_conv_2d] invalid tensor arguments size. d_input: %s, d_doutput: %s, gradient: %s",
			dim_to_str(d_input),
			dim_to_str(d_doutput),
			dim_to_str(d_gradient)
		);
	}
}

void kernel_conv_2d(
	const Stream& stream,
	const Tensor& d_input,
	const Tensor& d_doutput,
	Tensor& gradient
) {
	check_kernel_conv_2d(
		d_input,
		d_doutput,
		gradient
	);

	MemBlock<uint> indices;

	create_host_memblock(indices, d_doutput.h * d_doutput.w);

	for (int h = 0; h < d_doutput.h; ++h) {
		for (int w = 0; w < d_doutput.w; ++w) {
			indices.data[h * d_doutput.w + w] = h * d_input.w + w;
		}
	}

	check_cuda(cudaMemset(gradient.data, 0, get_mem_size(gradient)));

	dim3 threads(BLOCK_SIZE_32, BLOCK_SIZE_32);
	dim3 blocks = get_grid_size(
		threads,
		gradient.h * gradient.w * gradient.c,
		gradient.n
	);

	if (indices.len < CONST_ELEM_SIZE) {
		
		check_cuda(cudaMemcpyToSymbol(__indices, indices.data, sizeof(uint) * indices.len));

		for (uint i = 0; i < stream.str_size; ++i) {
			float* d_in = d_input.data + (i * d_input.h * d_input.w * d_input.c);
			float* d_dout = d_doutput.data + (i * d_doutput.h * d_doutput.w * d_doutput.c);

			__kernel_conv_2d_32x32_c_ind << <blocks, threads, 0, stream.str[i] >>>(
				d_in,
				d_dout,
				gradient.data,
				d_input.h,
				d_input.w,
				d_input.c,
				d_doutput.h,
				d_doutput.w,
				d_doutput.c,
				gradient.h,
				gradient.w
			);
			check_cuda(cudaStreamSynchronize(stream.str[i]));
		}
	}
	else {
		MemBlock<uint> d_indices;

		create_dev_memblock(d_indices, indices.len);
		check_cuda(cudaMemcpy(d_indices.data, indices.data, sizeof(uint) * indices.len, cudaMemcpyHostToDevice));

		for (uint i = 0; i < stream.str_size; ++i) {
			float* d_in = d_input.data + (i * d_input.h * d_input.w * d_input.c);
			float* d_dout = d_doutput.data + (i * d_doutput.h * d_doutput.w * d_doutput.c);

			__kernel_conv_2d_32x32_g_ind << <blocks, threads, 0, stream.str[i] >> > (
				d_in,
				d_dout,
				gradient.data,
				d_indices.data,
				d_input.h,
				d_input.w,
				d_input.c,
				d_doutput.h,
				d_doutput.w,
				d_doutput.c,
				gradient.h,
				gradient.w
			);
			check_cuda(cudaStreamSynchronize(stream.str[i]));
		}

		free_memblock(d_indices);
	}

	free_memblock(indices);
}

#endif