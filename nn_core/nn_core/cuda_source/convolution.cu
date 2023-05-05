#include "convolution.cuh"

#ifndef __CUDACC__
#define __CUDACC__
#endif

#include <device_functions.h>
#include <device_launch_parameters.h>

#ifdef FIX_MODE
/**********************************************/
/*											  */
/*				 kernel function			  */
/*										      */
/**********************************************/

__constant__ uint __indice[CONST_ELEM_SIZE];


__global__ void __conv_2d(
	const float* input,
	const float* kernel,
	float* output,
	cuint in_w,
	cuint in_h,
	cuint k_n,
	cuint k_w,
	cuint k_h,
	cuint k_c,
	cuint out_w,
	cuint out_h,
	cuint st_w,
	cuint st_h,
	cuint indice_offset
) {
	cuint cx = blockIdx.x * blockDim.x + threadIdx.x;
	cuint cy = blockIdx.y * blockDim.y + threadIdx.y;
	cuint sidx = threadIdx.y * BLOCK_SIZE + threadIdx.x;

	cuint x0 = (cx % out_w) * st_w;
	cuint y0 = (cx / out_w) * st_h;

	cuint n = k_w * k_h * k_c;
	cuint k = out_w * out_h;
	cuint khw = k_w * k_h;
	cuint in_wh = in_w * in_h;

	__shared__ float share_in[BLOCK_SIZE * BLOCK_SIZE];
	__shared__ float share_k[BLOCK_SIZE * BLOCK_SIZE];

	const float* p_input = input + (y0 * in_w + x0);
	const float* p_kernel = kernel + (cy * k_w * k_h * k_c);

	float sum = 0.f;

	uint th_x = 0;
	uint th_y = 0;
	uint in_ch_start = 0;
	uint in_k_index = 0;

	for (uint i = 0; i < n; i += BLOCK_SIZE) {
		th_x = i + threadIdx.x;
		th_y = i + threadIdx.y;
		in_ch_start = (th_y / khw) * in_wh;
		in_k_index = th_y % khw + indice_offset;

		__syncthreads();

		share_k[sidx] = th_x < n && cy < k_n ? p_kernel[th_x] : 0.f;
		share_in[sidx] = cx < k && th_y < n ? p_input[__indice[in_k_index] + in_ch_start] : 0.f;

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

__global__ void __correl_2d(
	float* d_output,
	float* d_kernel,
	float* d_input,
	cuint dout_w,
	cuint dk_n,
	cuint dk_w,
	cuint dk_h,
	cuint dk_c,
	cuint din_w,
	cuint din_h
) {
	cuint cx = blockIdx.x * blockDim.x + threadIdx.x;
	cuint cy = blockIdx.y * blockDim.y + threadIdx.y;
	cuint sidx = threadIdx.y * BLOCK_SIZE + threadIdx.x;

	cuint x0 = cx % din_w;
	cuint y0 = cx / din_w;

	//cuint n = dk_w * dk_h * dk_c;
	cuint tn = dk_w * dk_h * dk_n;
	cuint k = din_w * din_h;

	__shared__ float share_in[BLOCK_SIZE * BLOCK_SIZE];
	__shared__ float share_k[BLOCK_SIZE * BLOCK_SIZE];

	float* p_dout = d_output + (y0 * dout_w + x0);
	float* p_kernel = d_kernel + (cy * dk_w * dk_h);

	float sum = 0.f;

	for (uint i = 0; i < tn; i += BLOCK_SIZE) {
		__syncthreads();

		cuint wh = (threadIdx.x + i) % (dk_w * dk_h);
		cuint t_c = (threadIdx.x + i) / (dk_w * dk_h);
		float* pk = p_kernel + (t_c * dk_w * dk_h * dk_c);

		share_k[sidx] = (i + threadIdx.x) < tn && cy < dk_c ? pk[wh] : 0.f;
		share_in[sidx] = cx < k && (threadIdx.y + i) < tn ? p_dout[__indice[threadIdx.y + i]] : 0.f;

		__syncthreads();

#pragma unroll
		for (uint e = 0; e < BLOCK_SIZE; ++e) {
			sum += share_in[e * BLOCK_SIZE + threadIdx.x] * share_k[threadIdx.y * BLOCK_SIZE + e];
		}
	}

	if (cx < k && cy < dk_c) {
		d_input[cy * k + cx] = sum;
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

	cuint sidx = threadIdx.y * BLOCK_SIZE + threadIdx.x;

	__shared__ float sm_in[BLOCK_SIZE * BLOCK_SIZE];
	__shared__ float sm_dout[BLOCK_SIZE * BLOCK_SIZE];

	float* p_dout = d_output + (cy * d_output_h * d_output_w);
	float* p_in = input + (c0 * (input_h * input_w) + y0 * input_w + x0);

	float sum = 0.f;

	for (int i = 0; i < n; i += BLOCK_SIZE) {
		__syncthreads();

		sm_dout[sidx] = (threadIdx.x + i) < n && cy < d_output_c ? p_dout[threadIdx.x + i] : 0.f;
		sm_in[sidx] = cx < k && (threadIdx.y + i) < n ? p_in[__indice[threadIdx.y + i]] : 0.f;

		__syncthreads();

#pragma unroll
		for (int e = 0; e < BLOCK_SIZE; ++e) {
			sum += sm_dout[threadIdx.y * BLOCK_SIZE + e] * sm_in[e * BLOCK_SIZE + threadIdx.x];
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

	cuint sidx = threadIdx.y * BLOCK_SIZE + threadIdx.x;

	__shared__ float sm_in[BLOCK_SIZE * BLOCK_SIZE];
	__shared__ float sm_dout[BLOCK_SIZE * BLOCK_SIZE];

	float* p_dout = d_output + (cy * d_output_h * d_output_w);
	float* p_in = input + (c0 * (input_h * input_w) + y0 * input_w + x0);

	float sum = 0.f;

	for (int i = 0; i < n; i += BLOCK_SIZE) {
		__syncthreads();

		sm_dout[sidx] = (threadIdx.x + i) < n && cy < d_output_c ? p_dout[threadIdx.x + i] : 0.f;
		sm_in[sidx] = cx < k && (threadIdx.y + i) < n ? p_in[input_indices[threadIdx.y + i]] : 0.f;

		__syncthreads();

#pragma unroll
		for (int e = 0; e < BLOCK_SIZE; ++e) {
			sum += sm_dout[threadIdx.y * BLOCK_SIZE + e] * sm_in[e * BLOCK_SIZE + threadIdx.x];
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

	cuint row = k_count % c;
	cuint col = k_count / c;

	float* p_out = output + (row * (w * h * n) + col * (w * h));

	if (tidx < (n * h * w * c)) {
		p_out[k_idx] = input[tidx];
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

__global__ void __padding_2d(
	const float* input,
	float* output,
	cuint off_x,
	cuint off_y,
	cuint width,
	cuint height,
	cuint channel
) {
	cuint cx = blockIdx.x * blockDim.x + threadIdx.x;
	cuint cy = blockIdx.y * blockDim.y + threadIdx.y;
	cuint cz = blockIdx.z * blockDim.z + threadIdx.z;

	if (cx < width && cy < height && cz < channel) {
		output[cz * (width * height) + (cy + off_y) * width + (cx + off_x)] = input[cz * (width * height) + cy * width + cx];
	}
}

/**********************************************/
/*											  */
/*				  host function 			  */
/*										      */
/**********************************************/

void copy_to_indice(
	const uint* indice,
	const size_t size,
	const size_t offset
) {
	check_cuda(cudaMemcpyToSymbol(__indice, indice, size, offset));
}

void get_indice(
	uint* indice,
	size_t size,
	size_t offset
) {
	check_cuda(cudaMemcpyFromSymbol(indice, __indice, size, offset));
}

/*                convolution_2d              */

void conv_2d(
	cudaStream_t* streams,
	const CudaTensor d_input,
	const CudaTensor d_kernel,
	CudaTensor d_output,
	int st_w,
	int st_h,
	int indice_offset
) {
	//check_conv_2d(
	//	d_input,
	//	d_kernel,
	//	d_output,
	//	st_w,
	//	st_h
	//);

	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocks = get_grid_size(threads, d_output.w * d_output.h, d_output.c);
	
	for (int i = 0; i < d_input.n; ++i) {
		float* d_in = d_input.data + (i * d_input.w * d_input.h * d_input.c);
		float* d_out = d_output.data + (i * d_output.w * d_output.h * d_output.c);

		__conv_2d<<<blocks, threads, 0, streams[i % STREAMS]>>>(
			d_in,
			d_kernel.data,
			d_out,
			d_input.w,
			d_input.h,
			d_kernel.n,
			d_kernel.w,
			d_kernel.h,
			d_kernel.c,
			d_output.w,
			d_output.h,
			st_w,
			st_h,
			indice_offset
			);
		
	}
}


/*             correlation_2d            */

void correl_2d(
	cudaStream_t stream,
	const CudaTensor d_doutput,
	const CudaTensor d_kernel,
	CudaTensor d_dinput
) {
	//check_correl_2d(
	//	d_doutput,
	//	d_kernel,
	//	d_dinput
	//);

	uint* indices = new uint[d_kernel.n * d_kernel.h * d_kernel.w];

	for (int c = 0; c < d_kernel.n; ++c) {
		uint* p_indices = indices + (c * d_kernel.h * d_kernel.w);
		for (int h = 0; h < d_kernel.h; ++h) {
			for (int w = 0; w < d_kernel.w; ++w) {
				p_indices[h * d_kernel.w + w] = (c * d_doutput.h * d_doutput.w) + (d_kernel.h - h - 1) * d_doutput.w + (d_kernel.w - w - 1);
			}
		}
	}
	check_cuda(cudaMemcpyToSymbol(__indice, indices, sizeof(uint) * d_kernel.n * d_kernel.h * d_kernel.w));

	dim3 threads = dim3(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocks = get_grid_size(threads, d_dinput.w * d_dinput.h, d_dinput.c);

	for (int i = 0; i < d_doutput.n; ++i) {
		float* d_dout = d_doutput.data + (i * d_doutput.c * d_doutput.h * d_doutput.w);
		float* d_din = d_dinput.data + (i * d_dinput.c * d_dinput.h * d_dinput.w);

		__correl_2d << <blocks, threads, 0, stream >> > (
			d_dout,
			d_kernel.data,
			d_din,
			d_doutput.w,
			d_kernel.n,
			d_kernel.w,
			d_kernel.h,
			d_kernel.c,
			d_dinput.w,
			d_dinput.h
			);
	}
	check_cuda(cudaStreamSynchronize(stream));
	delete[] indices;
}

/*            transpose             */

void transpose(
	cudaStream_t stream,
	const CudaTensor d_input,
	CudaTensor d_output
) {
	//check_transpose(d_input, d_output);

	dim3 threads(SQR_BLOCK_SIZE);
	dim3 blocks = get_grid_size(threads, get_elem_size(d_input));

	__transpose << <blocks, threads, 0, stream >> > (
		d_input.data,
		d_output.data,
		d_input.n,
		d_input.c,
		d_input.h,
		d_input.w
		);
	check_cuda(cudaStreamSynchronize(stream));
}

/*            dilation_2d           */

void dilation_2d(
	cudaStream_t stream,
	const CudaTensor d_input,
	CudaTensor d_output,
	uint scale,
	int offset_x,
	int offset_y
) {
	//check_dilation_2d(
	//	d_input,
	//	d_output,
	//	scale,
	//	offset_x,
	//	offset_y
	//);

	check_cuda(cudaMemset(d_output.data, 0, sizeof(float) * get_elem_size(d_output)));

	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocks = get_grid_size(threads, d_input.w, d_input.h);

	for (int i = 0; i < d_input.n; ++i) {
		float* d_in = d_input.data + (i * d_input.h * d_input.w * d_input.c);
		float* d_out = d_output.data + (i * d_output.h * d_output.w * d_output.c);

		__dilation_2d << <blocks, threads, 0, stream >> > (
			d_in,
			d_out,
			d_input.w,
			d_input.h,
			d_input.c,
			d_output.w,
			d_output.h,
			scale,
			offset_x,
			offset_y
			);
	}

	check_cuda(cudaStreamSynchronize(stream));
}

/*               padding_conv_2d                */

void padding_conv_2d(
	cudaStream_t* s,
	const CudaTensor d_input,
	CudaTensor d_pad,
	const CudaTensor d_kernel,
	CudaTensor d_output,
	int st_w,
	int st_h,
	int indice_offset
) {
	cuint off_x = (d_pad.w - d_input.w) / 2;
	cuint off_y = (d_pad.h - d_input.h) / 2;

	uint* indice = new uint[d_kernel.h * d_kernel.w];

	for (uint h = 0; h < d_kernel.h; ++h) {
		for (uint w = 0; w < d_kernel.w; ++w) {
			indice[h * d_kernel.w + w] = (h * d_input.w) + w;
		}
	}

	check_cuda(cudaMemcpyToSymbol(
		__indice,
		indice,
		sizeof(uint) * d_kernel.h * d_kernel.w,
		0
	));

	delete[] indice;

	for (int n = 0; n < d_input.n; ++n) {
		int n_stream = n % STREAMS;
		const float* _d_input = d_input.data + (n * d_input.c * d_input.h * d_input.w);
		float* _d_output = d_output.data + (n * d_output.c * d_output.h * d_output.w);

		dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
		dim3 blocks = get_grid_size(threads, d_input.w, d_input.h, d_input.c);

		float* _d_pad = d_pad.data + (n_stream * d_pad.c * d_pad.h * d_pad.w);

		__padding_2d<<<blocks, threads, 0, s[n_stream]>>>(
			_d_input,
			_d_pad,
			off_x,
			off_y,
			d_input.w,
			d_input.h,
			d_input.c
		);

		blocks = get_grid_size(threads, d_output.w * d_output.h, d_output.c);

		__conv_2d<<<blocks, threads, 0, s[n_stream]>>>(
			_d_pad,
			d_kernel.data,
			_d_output,
			d_pad.w,
			d_pad.h,
			d_kernel.n,
			d_kernel.w,
			d_kernel.h,
			d_kernel.c,
			d_output.w,
			d_output.w,
			st_w,
			st_h,
			indice_offset
		);
	}
}

/*          kernel_convolution_2d          */

void kernel_conv_2d(
	cudaStream_t stream,
	const CudaTensor d_doutput,
	const CudaTensor d_input,
	CudaTensor d_gradient
) {
	//check_kernel_conv_2d(
	//	d_doutput,
	//	d_input,
	//	d_gradient
	//);

	uint* indices = new uint[d_doutput.h * d_doutput.w];

	for (int h = 0; h < d_doutput.h; ++h) {
		for (int w = 0; w < d_doutput.w; ++w) {
			indices[h * d_doutput.w + w] = h * d_input.w + w;
		}
	}

	check_cuda(cudaMemset(d_gradient.data, 0, sizeof(float) * get_elem_size(d_gradient)));

	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocks = get_grid_size(
		threads,
		d_gradient.h * d_gradient.w * d_gradient.c,
		d_gradient.n
	);

	if ((d_doutput.h * d_doutput.w) < CONST_ELEM_SIZE) {

		check_cuda(cudaMemcpyToSymbol(__indice, indices, sizeof(uint) * d_doutput.h * d_doutput.w));

		for (uint i = 0; i < d_doutput.n; ++i) {
			float* d_in = d_input.data + (i * d_input.h * d_input.w * d_input.c);
			float* d_dout = d_doutput.data + (i * d_doutput.h * d_doutput.w * d_doutput.c);

			__kernel_conv_2d_32x32_c_ind << <blocks, threads, 0, stream >> > (
				d_in,
				d_dout,
				d_gradient.data,
				d_input.h,
				d_input.w,
				d_input.c,
				d_doutput.h,
				d_doutput.w,
				d_doutput.c,
				d_gradient.h,
				d_gradient.w
				);
			check_cuda(cudaStreamSynchronize(stream));
		}
	}
	else {
		uint* d_indices = NULL;

		check_cuda(cudaMalloc(&d_indices, sizeof(uint) * d_doutput.h * d_doutput.w));
		check_cuda(cudaMemcpy(d_indices, indices, sizeof(uint) * d_doutput.h * d_doutput.w, cudaMemcpyHostToDevice));

		for (uint i = 0; i < d_doutput.n; ++i) {
			float* d_in = d_input.data + (i * d_input.h * d_input.w * d_input.c);
			float* d_dout = d_doutput.data + (i * d_doutput.h * d_doutput.w * d_doutput.c);

			__kernel_conv_2d_32x32_g_ind << <blocks, threads, 0, stream >> > (
				d_in,
				d_dout,
				d_gradient.data,
				d_indices,
				d_input.h,
				d_input.w,
				d_input.c,
				d_doutput.h,
				d_doutput.w,
				d_doutput.c,
				d_gradient.h,
				d_gradient.w
				);
			check_cuda(cudaStreamSynchronize(stream));
		}
		check_cuda(cudaFree(d_indices));
	}
	delete[] indices;
}
#endif