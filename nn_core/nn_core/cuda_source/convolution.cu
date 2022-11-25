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

__constant__ uint __indices[CONST_ELEM_SIZE];


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

uint get_output_size(
	int input_size,
	int kernel_size,
	int pad_size,
	int stride
) {
	return (input_size + (2 * pad_size) - kernel_size) / stride + 1;
}

void check_conv_2d(
	const Tensor& d_input,
	const Tensor& d_kernel,
	const Tensor& d_output,
	int st_w,
	int st_h
) {
	int out_h = get_output_size(d_input.h, d_kernel.h, 0, st_h);
	int out_w = get_output_size(d_input.w, d_kernel.w, 0, st_w);

	if (d_output.h != out_h || d_output.w != out_w) {
		ErrorExcept(
			"[check_conv_2d] invalid output dimension %s",
			dim_to_str(d_output)
		);
	}
	else if (d_kernel.c != d_input.c || d_kernel.n != d_output.c) {
		ErrorExcept(
			"[check_conv_2d] invalid channels input: %s, kernel: %s, output: %s",
			dim_to_str(d_input),
			dim_to_str(d_kernel),
			dim_to_str(d_output)
		);
	}
}

void conv_2d(
	const Stream& stream,
	const Tensor& d_input,
	const Tensor& d_kernel,
	Tensor& d_output,
	int st_w,
	int st_h
) {
	check_conv_2d(
		d_input,
		d_kernel,
		d_output,
		st_w,
		st_h
	);

	MemBlock<uint> indices;

	create_host_memblock(indices, get_elem_size(d_kernel));

	for (int c = 0; c < d_kernel.c; ++c) {
		for (int h = 0; h < d_kernel.h; ++h) {
			uint* p_indices = indices.data + (c * d_kernel.h * d_kernel.w);
			for (int w = 0; w < d_kernel.w; ++w) {
				p_indices[h * d_kernel.w + w] = (c * d_input.h * d_input.w) + (h * d_input.w) + w;
			}
		}
	}
	check_cuda(cudaMemcpyToSymbol(__indices, indices.data, sizeof(uint) * indices.len));

	dim3 threads(BLOCK_SIZE_32, BLOCK_SIZE_32);
	dim3 blocks = get_grid_size(threads, d_output.w * d_output.h, d_output.c);

	for (int i = 0; i < stream.str_size; ++i) {
		float* d_in = d_input.data + (i * d_input.w * d_input.h * d_input.c);
		float* d_out = d_output.data + (i * d_output.w * d_output.h * d_output.c);

		__conv_2d << <blocks, threads, 0, stream.str[i] >> > (
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

	free_memblock(indices);
}



void check_correl_2d(
	const Tensor& d_doutput,
	const Tensor& d_kernel,
	const Tensor& d_dinput
) {
	int d_in_w = d_doutput.w - d_kernel.w + 1;
	int d_in_h = d_doutput.h - d_kernel.h + 1;

	if (
		d_doutput.c != d_kernel.n ||
		d_dinput.c != d_kernel.c ||
		d_dinput.w != d_in_w ||
		d_dinput.h != d_in_h
		) {
		ErrorExcept(
			"[check_correl_2d] invalid (d_output, kernel, d_input) size. d_doutput: %s, d_tkernel: %s, d_dinput: %s",
			dim_to_str(d_doutput), dim_to_str(d_kernel), dim_to_str(d_dinput)
		);
	}
}

void correl_2d(
	const Stream& stream,
	const Tensor& d_doutput,
	const Tensor& d_kernel,
	Tensor& d_dinput
) {
	check_correl_2d(
		d_doutput,
		d_kernel,
		d_dinput
	);

	MemBlock<uint> indices;

	create_host_memblock(indices, get_elem_size(d_kernel));

	for (int c = 0; c < d_kernel.n; ++c) {
		uint* p_indices = indices.data + (c * d_kernel.h * d_kernel.w);
		for (int h = 0; h < d_kernel.h; ++h) {
			for (int w = 0; w < d_kernel.w; ++w) {
				p_indices[h * d_kernel.w + w] = (c * d_doutput.h * d_doutput.w) + (d_kernel.h - h - 1) * d_doutput.w + (d_kernel.w - w - 1);
			}
		}
	}
	check_cuda(cudaMemcpyToSymbol(__indices, indices.data, sizeof(uint) * indices.len));

	Tensor t_kernel;
	create_dev_tensor(t_kernel, d_kernel.c, d_kernel.n, d_kernel.h, d_kernel.w);

	dim3 threads(BLOCK_SIZE_1024);
	dim3 blocks = get_grid_size(threads, get_elem_size(d_kernel));

	__transpose<<<blocks, threads, 0, stream.str[0]>>>(
		d_kernel.data,
		t_kernel.data,
		d_kernel.n,
		d_kernel.c,
		d_kernel.h,
		d_kernel.w
	);
	check_cuda(cudaStreamSynchronize(stream.str[0]));

	threads = dim3(BLOCK_SIZE_32, BLOCK_SIZE_32);
	blocks = get_grid_size(threads, d_dinput.w * d_dinput.h, d_dinput.c);

	for (int i = 0; i < stream.str_size; ++i) {
		float* d_dout = d_doutput.data + (i * d_doutput.c * d_doutput.h * d_doutput.w);
		float* d_din = d_dinput.data + (i * d_dinput.c * d_dinput.h * d_dinput.w);

		__conv_2d<<<blocks, threads, 0, stream.str[i]>>>(
			d_dout,
			t_kernel.data,
			d_din,
			d_doutput.w,
			t_kernel.n,
			t_kernel.w,
			t_kernel.h,
			t_kernel.c,
			d_dinput.w,
			d_dinput.h,
			1, 1
		);
	}
	sync_streams(stream);

	free_memblock(indices);
	free_tensor(t_kernel);
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

			__kernel_conv_2d_32x32_c_ind << <blocks, threads, 0, stream.str[i] >> > (
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