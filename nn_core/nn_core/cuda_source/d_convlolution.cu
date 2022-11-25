#include "d_convolution.cuh"

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

__global__ void __conv_2d2(
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
	cuint sidx = threadIdx.y * BLOCK_SIZE_32 + threadIdx.x;

	cuint x0 = (cx % out_w) * st_w;
	cuint y0 = (cx / out_w) * st_h;

	cuint n = k_w * k_h * k_c;
	cuint k = out_w * out_h;

	__shared__ float share_in[BLOCK_SIZE_32 * BLOCK_SIZE_32];
	__shared__ float share_k[BLOCK_SIZE_32 * BLOCK_SIZE_32];

	float* p_input = input + (y0 * in_w + x0);
	float* p_kernel = kernel + (cy * k_w * k_h * k_c);

	float sum = 0.f;

	for (uint i = 0; i < n; i += BLOCK_SIZE_32) {
		__syncthreads();

		share_k[sidx] = (i + threadIdx.x) < n && cy < k_n ? p_kernel[threadIdx.x + i] : 0.f;
		share_in[sidx] = cx < k && (threadIdx.y + i) < n ? p_input[__indices[threadIdx.y + i]] : 0.f;

		__syncthreads();

#pragma unroll
		for (uint e = 0; e < BLOCK_SIZE_32; ++e) {
			sum += share_in[e * BLOCK_SIZE_32 + threadIdx.x] * share_k[threadIdx.y * BLOCK_SIZE_32 + e];
		}
	}

	if (cx < k && cy < k_n) {
		output[cy * k + cx] = sum;
	}
}

__global__ void __transpose(
	float* input,
	float* output,
	cuint n,
	cuint c,
	cuint h,
	cuint w
) {
	cuint tidx = blockIdx.x * blockDim.x + threadIdx.x;
	cuint k_idx = tidx % (w * h);
	cuint k_count = tidx / (w * h);

	cuint row = k_count % n;
	cuint col = k_count / n;

	float* p_input = input + (row * (w * h * n) + col * (w * h));

	if (tidx < (n * h * w * c)) {
		output[tidx] = p_input[k_idx];
	}
}

__global__ void __dilation_2d(
	float* input,
	float* output,
	cuint iw,
	cuint ih,
	cuint ic,
	cuint ow,
	cuint oh,
	cuint scale,
	cint offset_x,
	cint offset_y
) {
	cuint in_cx = blockIdx.x * blockDim.x + threadIdx.x;
	cuint in_cy = blockIdx.y * blockDim.y + threadIdx.y;

	cuint out_cx = in_cx * scale + offset_x;
	cuint out_cy = in_cy * scale + offset_y;

	for (int c = 0; c < ic; ++c) {
		if (in_cx < iw && in_cy < ih) {
			output[c * (ow * oh) + out_cy * ow + out_cx] = input[c * (iw * ih) + in_cy * iw + in_cx];
		}
	}
}



/**********************************************/
/*											  */
/*				  host function 			  */
/*										      */
/**********************************************/

void check_correl_2d(
	const Tensor& d_doutput,
	const Tensor& d_tkernel,
	const Tensor& d_dinput
) {
	int d_in_w = d_doutput.w - d_tkernel.w + 1;
	int d_in_h = d_doutput.h - d_tkernel.h + 1;

	if (
		d_doutput.c != d_tkernel.n ||
		d_dinput.c != d_tkernel.c ||
		d_dinput.w != d_in_w ||
		d_dinput.h != d_in_h
		) {
		ErrorExcept(
			"[check_correl_2d] invalid (d_output, kernel, d_input) size. d_doutput: %s, d_tkernel: %s, d_dinput: %s",
			dim_to_str(d_doutput), dim_to_str(d_tkernel), dim_to_str(d_dinput)
		);
	}
}

void _conv_2d2(
	const Stream& stream,
	const Tensor& d_input,
	const Tensor& d_kernel,
	Tensor& d_output,
	int st_w,
	int st_h
) {
	dim3 threads(BLOCK_SIZE_32, BLOCK_SIZE_32);
	dim3 blocks = get_grid_size(threads, d_output.w * d_output.h, d_output.c);

	for (int i = 0; i < stream.str_size; ++i) {
		float* d_in = d_input.data + (i * d_input.w * d_input.h * d_input.c);
		float* d_out = d_output.data + (i * d_output.w * d_output.h * d_output.c);

		__conv_2d2 << <blocks, threads, 0, stream.str[i] >> > (
			d_in,
			d_kernel.data,
			d_out,
			d_input.w,
			d_kernel.n,
			d_kernel.w,
			d_kernel.h,
			d_kernel.c,
			d_output.w,
			d_output.h,
			st_w,
			st_h
			);
	}
	sync_streams(stream);
}

void correl_2d(
	const Stream& stream,
	const Tensor& d_doutput,
	const Tensor& d_kernel,
	Tensor& d_dinput
) {
	MemBlock<uint> indices;

	create_host_memblock(indices, get_elem_size(d_kernel));

	for (int c = 0; c < d_kernel.n; ++c) {
		uint* p_indices = indices.data + (c * d_kernel.h * d_kernel.w);
		for (int h = 0; h < d_kernel.h; ++h) {
			for (int w = 0; w < d_kernel.w; ++w) {
				p_indices[h * d_kernel.w + w] = (c * d_doutput.h * d_doutput.w) + h * d_doutput.w + w;
			}
		}
	}
	check_cuda(cudaMemcpyToSymbol(__indices, indices.data, sizeof(uint) * indices.len));

	_conv_2d2(
		stream,
		d_doutput,
		d_kernel,
		d_dinput,
		1, 1
	);

	free_memblock(indices);
}

void dilation_2d(
	const Stream& stream,
	const Tensor& input,
	Tensor& output,
	uint scale,
	int offset_x,
	int offset_y
) {
	check_cuda(cudaMemset(output.data, 0, get_mem_size(output)));

	dim3 threads(BLOCK_SIZE_32, BLOCK_SIZE_32);
	dim3 blocks = get_grid_size(threads, input.w, input.h);

	for (int i = 0; i < stream.str_size; ++i) {
		float* d_in = input.data + (i * input.h * input.w * input.c);
		float* d_out = output.data + (i * output.h * output.w * output.c);

		__dilation_2d << <blocks, threads, 0, stream.str[i] >> > (
			d_in,
			d_out,
			input.w,
			input.h,
			input.c,
			output.w,
			output.h,
			scale,
			offset_x,
			offset_y
		);
	}

	sync_streams(stream);
}

#endif