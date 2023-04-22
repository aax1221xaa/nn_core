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

	cuint n = dk_w * dk_h * dk_c;
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
		share_in[sidx] = cx < k && (threadIdx.y + i) < tn ? p_dout[__indices[threadIdx.y + i]] : 0.f;

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
		sm_in[sidx] = cx < k && (threadIdx.y + i) < n ? p_in[__indices[threadIdx.y + i]] : 0.f;

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



/**********************************************/
/*											  */
/*				  host function 			  */
/*										      */
/**********************************************/

/*                convolution_2d              */
/*
int get_output_size(
	int input_size,
	int kernel_size,
	int pad_size,
	int stride
) {
	return (input_size + (2 * pad_size) - kernel_size) / stride + 1;
}

void check_conv_2d(
	const NN_Tensor4D d_input,
	const NN_Tensor4D d_kernel,
	const NN_Tensor4D d_output,
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
*/

void conv_2d(
	cudaStream_t stream,
	const CudaTensor d_input,
	const CudaTensor d_kernel,
	CudaTensor d_output,
	int st_w,
	int st_h
) {
	//check_conv_2d(
	//	d_input,
	//	d_kernel,
	//	d_output,
	//	st_w,
	//	st_h
	//);

	uint *indices = new uint[d_kernel.c * d_kernel.h * d_kernel.w];

	for (int c = 0; c < d_kernel.c; ++c) {
		for (int h = 0; h < d_kernel.h; ++h) {
			uint* p_indices = indices + (c * d_kernel.h * d_kernel.w);
			for (int w = 0; w < d_kernel.w; ++w) {
				p_indices[h * d_kernel.w + w] = (c * d_input.h * d_input.w) + (h * d_input.w) + w;
			}
		}
	}
	check_cuda(cudaMemcpyToSymbol(__indices, indices, sizeof(uint) * d_kernel.c * d_kernel.h * d_kernel.w));

	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocks = get_grid_size(threads, d_output.w * d_output.h, d_output.c);

	for (int i = 0; i < d_input.n; ++i) {
		float* d_in = d_input.data + (i * d_input.w * d_input.h * d_input.c);
		float* d_out = d_output.data + (i * d_output.w * d_output.h * d_output.c);

		__conv_2d << <blocks, threads, 0, stream >> > (
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
	check_cuda(cudaStreamSynchronize(stream));
	delete[] indices;
}


/*             correlation_2d            */

//void check_correl_2d(
//	const NN_Tensor4D d_doutput,
//	const NN_Tensor4D d_kernel,
//	const NN_Tensor4D d_dinput
//) {
//	int d_in_w = d_doutput.w - d_kernel.w + 1;
//	int d_in_h = d_doutput.h - d_kernel.h + 1;
//
//	if (
//		d_doutput.c != d_kernel.n ||
//		d_dinput.c != d_kernel.c ||
//		d_dinput.w != d_in_w ||
//		d_dinput.h != d_in_h
//		) {
//		ErrorExcept(
//			"[check_correl_2d] invalid (d_output, kernel, d_input) size. d_doutput: %s, d_tkernel: %s, d_dinput: %s",
//			dim_to_str(d_doutput), dim_to_str(d_kernel), dim_to_str(d_dinput)
//		);
//	}
//}

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
	check_cuda(cudaMemcpyToSymbol(__indices, indices, sizeof(uint) * d_kernel.n * d_kernel.h * d_kernel.w));

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

//void check_transpose(
//	const NN_Tensor4D d_input,
//	const NN_Tensor4D d_output
//) {
//	if (get_elem_size(d_input) != get_elem_size(d_output)) {
//		ErrorExcept(
//			"[transpose] invalid input, output size. input: %d, output: %d",
//			get_elem_size(d_input),
//			get_elem_size(d_output)
//		);
//	}
//
//	if (d_input.n != d_output.c || d_input.c != d_output.n) {
//		ErrorExcept(
//			"[check_transpose] input, output tensor 0, 1 channels are invalid. input: %s, output: %s",
//			dim_to_str(d_input),
//			dim_to_str(d_output)
//		);
//	}
//}

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

//void check_dilation_2d(
//	const NN_Tensor4D input,
//	const NN_Tensor4D output,
//	uint scale,
//	int offset_x,
//	int offset_y
//) {
//	int out_w = input.w * scale + offset_x;
//	int out_h = input.h * scale + offset_y;
//
//	if (out_w > output.w || out_h > output.h) {
//		ErrorExcept(
//			"[check_dilation_2d] output is too small. output: %s, expect output: [%d, %d, %d, %d]",
//			dim_to_str(output),
//			output.n,
//			output.c,
//			out_h,
//			out_w
//		);
//	}
//}

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


/*          kernel_convolution_2d          */

//void check_kernel_conv_2d(
//	const NN_Tensor4D d_doutput,
//	const NN_Tensor4D d_input,
//	NN_Tensor4D d_gradient
//) {
//	int in_h = d_input.h - d_doutput.h + 1;
//	int in_w = d_input.w - d_doutput.w + 1;
//
//	if (d_gradient.h != in_h ||
//		d_gradient.w != in_w ||
//		d_gradient.n != d_doutput.c ||
//		d_gradient.c != d_input.c ||
//		d_input.n != d_doutput.n) {
//
//		ErrorExcept(
//			"[check_kernel_conv_2d] invalid tensor arguments size. d_input: %s, d_doutput: %s, gradient: %s",
//			dim_to_str(d_input),
//			dim_to_str(d_doutput),
//			dim_to_str(d_gradient)
//		);
//	}
//}

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

		check_cuda(cudaMemcpyToSymbol(__indices, indices, sizeof(uint) * d_doutput.h * d_doutput.w));

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