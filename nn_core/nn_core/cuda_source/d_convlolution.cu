#include "d_convolution.cuh"
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

__global__ void __transpose(
	float* input,
	float* output,
	cuint n,
	cuint h,
	cuint w,
	cuint c
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
	cint scale,
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
	const Tensor* d_output,
	const Tensor* kernel,
	const Tensor* d_input,
	const Tensor* work_space
) {
	int d_in_w = d_output->w - kernel->w + 1;
	int d_in_h = d_output->h - kernel->h + 1;
	size_t k_size = GetTotalSize(kernel);
	size_t w_size = GetTotalSize(work_space);

	if (
		d_output->c != kernel->n ||
		d_input->c != kernel->c ||
		d_input->w != d_in_w ||
		d_input->h != d_in_h
		) {
		ErrorExcept(
			"[check_correl_2d] invalid (d_output, kernel, d_input) size.\
			 d_output: [%d, %d, %d, %d]\
			 kernel: [%d, %d, %d, %d]\
			 d_input: [%d, %d, %d, %d]",
			d_output->n,
			d_output->h,
			d_output->w,
			d_output->c,
			kernel->n,
			kernel->h,
			kernel->w,
			kernel->c,
			d_input->n,
			d_input->h,
			d_input->w,
			d_input->c
		);
	}
	else if (w_size < k_size) {
		ErrorExcept(
			"[check_correl_2d] work space is smaller then kernel. w_space: %d < kernel: %d",
			w_size, k_size
		);
	}
}

void correl_2d(
	const Stream* stream,
	const Tensor* d_output,
	const Tensor* kernel,
	Tensor* d_input,
	Tensor* work_space
) {
	uint k_index[CONST_SIZE] = { 0, };
	int kn = kernel->n;
	int kh = kernel->h;
	int kw = kernel->w;
	int kc = kernel->c;

	check_correl_2d(
		d_output,
		kernel,
		d_input,
		work_space
	);

	for (int n = 0; n < kn; ++n) {
		for (int h = 0; h < kh; ++h) {
			for (int w = 0; w < kw; ++w) {
				k_index[n * (kh * kw) + h * kw + w] = n * (d_output->w * d_output->h) + (kh - h - 1) * d_output->w + (kw - w - 1);
				printf("%d ", k_index[n * (kh * kw) + h * kw + w]);
			}
			printf("\n");
		}
		printf("\n");
	}

	checkCuda(cudaMemcpyToSymbol(__indices, k_index, sizeof(k_index)));
	
	dim3 threads(BLOCK_SIZE * BLOCK_SIZE);
	dim3 blocks((kn * kh * kw * kc + threads.x - 1) / threads.x);

	__transpose<<<blocks, threads, 0, stream->st[0]>>>(
		kernel->data,
		work_space->data,
		kn,
		kh,
		kw,
		kc
	);

	checkCuda(cudaStreamSynchronize(stream->st[0]));

	threads = dim3(BLOCK_SIZE, BLOCK_SIZE);
	blocks = dim3(
		GetBlockSize(d_input->w * d_input->h),
		GetBlockSize(d_input->c)
	);

	for (int i = 0; i < stream->st_size; ++i) {
		float* d_in = d_output->data + (i * d_output->w * d_output->h * d_output->c);
		float* d_out = d_input->data + (i * d_input->w * d_input->h * d_input->c);

		__conv_2d<<<blocks, threads, 0, stream->st[i]>>>(
			d_in,
			work_space->data,
			d_out,
			d_output->w,
			kc,
			kw,
			kh,
			kn,
			d_input->w,
			d_input->h,
			1, 1
		);
	}

	SyncStreams(stream);
}

void dilation_2d(
	const Stream* stream,
	const Tensor* input,
	Tensor* output,
	int scale,
	int offset_x,
	int offset_y
) {
	size_t out_size = GetTotalSize(output);
	checkCuda(cudaMemset(output->data, 0, out_size));

	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocks(
		GetBlockSize(input->w),
		GetBlockSize(input->h)
	);

	for (int i = 0; i < stream->st_size; ++i) {
		float* d_in = input->data + (i * input->h * input->w * input->c);
		float* d_out = output->data + (i * output->h * output->w * output->c);

		__dilation_2d<<<blocks, threads, 0, stream->st[i]>>>(
			d_in,
			d_out,
			input->w,
			input->h,
			input->c,
			output->w,
			output->h,
			scale,
			offset_x,
			offset_y
		);
	}

	SyncStreams(stream);
}