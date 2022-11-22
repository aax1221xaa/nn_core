#include "k_convolution.cuh"
#include "convolution.cuh"


#ifndef __CUDACC__
#define __CUDACC__
#endif

#include <device_functions.h>
#include <device_launch_parameters.h>


// __constant__ uint __indices[];

/**********************************************/
/*											  */
/*				 kernel function			  */
/*										      */
/**********************************************/

__declspec(deprecated("This function is too slow.")) __global__ void __kernel_conv_2d_1x1024_g_ind(
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

	cuint cx = blockIdx.x;
	cuint cy = blockIdx.y;

	cuint x0 = cx % gradient_w;
	cuint y0 = (cx / gradient_w) % gradient_h;
	cuint c0 = cx / (gradient_h * gradient_w);

	__shared__ float sm_a[BLOCK_SIZE * BLOCK_SIZE];
	__shared__ float sm_b[BLOCK_SIZE * BLOCK_SIZE];
	
	float sum = 0.f;
	float* p_dout = d_output + (cy * d_output_h * d_output_w);
	float* p_input = input + (c0 * (input_h * input_w) + y0 * input_w + x0);
	
	for (int i = 0; i < n; i += (BLOCK_SIZE * BLOCK_SIZE)) {
		if (threadIdx.x + i < n) {
			sm_a[threadIdx.x] = p_dout[threadIdx.x + i];
			sm_b[threadIdx.x] = p_input[input_indices[threadIdx.x + i]];
		}
		else {
			sm_a[threadIdx.x] = 0.f;
			sm_b[threadIdx.x] = 0.f;
		}
		__syncthreads();

		sum += sm_a[threadIdx.x] * sm_b[threadIdx.x];

		__syncthreads();
	}

	sm_a[threadIdx.x] = sum;

	__syncthreads();

	#pragma unroll
	for (int i = (BLOCK_SIZE * BLOCK_SIZE) / 2; i > 0; i /= 2) {
		if (threadIdx.x < i) sm_a[threadIdx.x] += sm_a[threadIdx.x + i];

		__syncthreads();
	}

	if (threadIdx.x == 0) {
		gradient[cy * k + cx] += sm_a[0];
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



/**********************************************/
/*											  */
/*				  host function 			  */
/*										      */
/**********************************************/

void check_kernel_conv_2d(
	const Tensor& in,
	const Tensor& dout,
	Tensor& g
) {
	int in_h = in.h - dout.h + 1;
	int in_w = in.w - dout.w + 1;

	if (g.h != in_h || g.w != in_w || g.n != dout.c || g.c != in.c || in.n != dout.n) {
		ErrorExcept(
			"[check_kernel_conv_2d] invalid gradient size \
			 input: [%d, %d, %d, %d], \
			 d_output: [%d, %d, %d, %d], \
			 gradient: [%d, %d, %d, %d]",
			in.n, in.h, in.w, in.c,
			dout.n, dout.h, dout.w, dout.c,
			g.n, g.h, g.w, g.c
		);
	}
}


#if _DEBUG

void kernel_conv_2d(
	const Stream* stream,
	const Tensor* input,
	const Tensor* d_output,
	Tensor* gradient
) {
	check_kernel_conv_2d(
		*input,
		*d_output,
		*gradient
	);

	uint* h_indices = new uint[d_output->h * d_output->w];

	for (int h = 0; h < d_output->h; ++h) {
		for (int w = 0; w < d_output->w; ++w) {
			h_indices[h * d_output->w + w] = h * input->w + w;
		}
	}

	checkCuda(cudaMemset(gradient->data, 0, GetTotalSize(gradient)));

	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocks(
		GetBlockSize(gradient->h * gradient->w * gradient->c),
		GetBlockSize(gradient->n)
	);

	if (d_output->h * d_output->w < CONST_SIZE) {
		
		checkCuda(cudaMemcpyToSymbol(__indices, h_indices, sizeof(uint) * d_output->h * d_output->w));

		for (uint i = 0; i < stream->st_size; ++i) {
			float* d_in = input->data + (i * input->h * input->w * input->c);
			float* d_dout = d_output->data + (i * d_output->h * d_output->w * d_output->c);

			__kernel_conv_2d_32x32_c_ind<<<blocks, threads, 0, stream->st[i]>>>(
				d_in,
				d_dout,
				gradient->data,
				input->h,
				input->w,
				input->c,
				d_output->h,
				d_output->w,
				d_output->c,
				gradient->h,
				gradient->w
			);
			checkCuda(cudaStreamSynchronize(stream->st[i]));
		}
	}
	else {
		uint* d_indices = NULL;

		checkCuda(cudaMalloc(&d_indices, sizeof(uint) * d_output->h * d_output->w));
		checkCuda(cudaMemcpy(d_indices, h_indices, sizeof(uint) * d_output->h * d_output->w, cudaMemcpyHostToDevice));

		for (uint i = 0; i < stream->st_size; ++i) {
			float* d_in = input->data + (i * input->h * input->w * input->c);
			float* d_dout = d_output->data + (i * d_output->h * d_output->w * d_output->c);

			__kernel_conv_2d_32x32_g_ind<<<blocks, threads, 0, stream->st[i]>>>(
				d_in,
				d_dout,
				gradient->data,
				d_indices,
				input->h,
				input->w,
				input->c,
				d_output->h,
				d_output->w,
				d_output->c,
				gradient->h,
				gradient->w
			);
			checkCuda(cudaStreamSynchronize(stream->st[i]));
		}

		checkCuda(cudaFree(d_indices));
	}

	delete[] h_indices;
}

__declspec(deprecated) void kernel_conv_2d_1x1024_g_ind(
	const Stream* stream,
	const Tensor* input,
	const Tensor* d_output,
	Tensor* gradient
) {
	check_kernel_conv_2d(
		*input,
		*d_output,
		*gradient
	);

	uint* d_indices = NULL;
	uint* h_indices = new uint[d_output->h * d_output->w];


	for (int h = 0; h < d_output->h; ++h) {
		uint* p_indices = h_indices + (h * d_output->w);
		for (int w = 0; w < d_output->w; ++w) {
			p_indices[w] = h * input->w + w;
		}
	}

	checkCuda(cudaMalloc(&d_indices, sizeof(uint) * (d_output->h * d_output->w)));
	checkCuda(cudaMemcpy(d_indices, h_indices, sizeof(uint) * (d_output->h * d_output->w), cudaMemcpyHostToDevice));

	checkCuda(cudaMemset(gradient->data, 0, GetTotalSize(gradient)));

	dim3 threads(BLOCK_SIZE * BLOCK_SIZE);
	dim3 blocks(gradient->h * gradient->w * gradient->c, gradient->n);

	for (int i = 0; i < stream->st_size; ++i) {
		float* d_in = input->data + (i * input->h * input->w * input->c);
		float* d_dout = d_output->data + (i * d_output->h * d_output->w * d_output->c);

		__kernel_conv_2d_1x1024_g_ind<<<blocks, threads, 0, stream->st[i]>>>(
			d_in,
			d_dout,
			gradient->data,
			d_indices,
			input->h,
			input->w,
			input->c,
			d_output->h,
			d_output->w,
			d_output->c,
			gradient->h,
			gradient->w
		);
		checkCuda(cudaStreamSynchronize(stream->st[i]));
	}

	checkCuda(cudaFree(d_indices));
	delete[] h_indices;
}

void kernel_conv_2d_32x32_c_ind(
	const Stream* stream,
	const Tensor* input,
	const Tensor* d_output,
	Tensor* gradient
) {
	check_kernel_conv_2d(
		*input,
		*d_output,
		*gradient
	);

	uint h_indices[CONST_SIZE] = { 0, };

	for (int h = 0; h < d_output->h; ++h) {
		for (int w = 0; w < d_output->w; ++w) {
			h_indices[h * d_output->w + w] = h * input->w + w;
		}
	}

	checkCuda(cudaMemcpyToSymbol(__indices, h_indices, sizeof(h_indices)));

	checkCuda(cudaMemset(gradient->data, 1, GetTotalSize(gradient)));

	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocks(GetBlockSize(gradient->h * gradient->w * gradient->c), GetBlockSize(gradient->n));

	printf("threads = [%d, %d, %d]\n",
		threads.x, threads.y, threads.z);
	printf("threads = [%d, %d, %d]\n",
		blocks.x, blocks.y, blocks.z);

	for (int i = 0; i < stream->st_size; ++i) {
		float* d_in = input->data + (i * input->h * input->w * input->c);
		float* d_dout = d_output->data + (i * d_output->h * d_output->w * d_output->c);

		__kernel_conv_2d_32x32_c_ind<<<blocks, threads, 0, stream->st[i]>>>(
			d_in,
			d_dout,
			gradient->data,
			input->h,
			input->w,
			input->c,
			d_output->h,
			d_output->w,
			d_output->c,
			gradient->h,
			gradient->w
			);
		checkCuda(cudaStreamSynchronize(stream->st[i]));
	}
}

void kernel_conv_2d_32x32_g_ind(
	const Stream* stream,
	const Tensor* input,
	const Tensor* d_output,
	Tensor* gradient
) {
	check_kernel_conv_2d(
		*input,
		*d_output,
		*gradient
	);

	uint* d_indices = NULL;
	uint* h_indices = new uint[d_output->h * d_output->w];

	for (int c = 0; c < d_output->c; ++c) {
		for (int h = 0; h < d_output->h; ++h) {
			uint* p_indices = h_indices + (h * d_output->w);
			for (int w = 0; w < d_output->w; ++w) {
				p_indices[w] = h * input->w + w;
			}
		}
	}

	checkCuda(cudaMalloc(&d_indices, sizeof(uint) * (d_output->h * d_output->w)));
	checkCuda(cudaMemcpy(d_indices, h_indices, sizeof(uint) * (d_output->h * d_output->w), cudaMemcpyHostToDevice));

	checkCuda(cudaMemset(gradient->data, 0, GetTotalSize(gradient)));

	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocks(
		GetBlockSize(gradient->h * gradient->w * gradient->c),
		GetBlockSize(gradient->n)
	);

	for (int i = 0; i < stream->st_size; ++i) {
		float* d_in = input->data + (i * input->h * input->w * input->c);
		float* d_dout = d_output->data + (i * d_output->h * d_output->w * d_output->c);

		__kernel_conv_2d_32x32_g_ind<<<blocks, threads, 0, stream->st[i]>>>(
			d_in,
			d_dout,
			gradient->data,
			d_indices,
			input->h,
			input->w,
			input->c,
			d_output->h,
			d_output->w,
			d_output->c,
			gradient->h,
			gradient->w
			);
		checkCuda(cudaStreamSynchronize(stream->st[i]));
	}

	checkCuda(cudaFree(d_indices));
	delete[] h_indices;
}

#else

void kernel_conv_2d(
	const Stream* stream,
	const Tensor* input,
	const Tensor* d_output,
	Tensor* gradient
) {
	check_kernel_conv_2d(
		*input,
		*d_output,
		*gradient
	);

	uint* h_indices = new uint[d_output->h * d_output->w];

	for (int h = 0; h < d_output->h; ++h) {
		for (int w = 0; w < d_output->w; ++w) {
			h_indices[h * d_output->w + w] = h * input->w + w;
		}
	}

	checkCuda(cudaMemset(gradient->data, 0, GetTotalSize(gradient)));

	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocks(
		GetBlockSize(gradient->h * gradient->w * gradient->c),
		GetBlockSize(gradient->n)
	);

	if (d_output->h * d_output->w < CONST_SIZE) {

		checkCuda(cudaMemcpyToSymbol(__indices, h_indices, sizeof(uint) * d_output->h * d_output->w));

		for (uint i = 0; i < stream->st_size; ++i) {
			float* d_in = input->data + (i * input->h * input->w * input->c);
			float* d_dout = d_output->data + (i * d_output->h * d_output->w * d_output->c);

			__kernel_conv_2d_32x32_c_ind << <blocks, threads, 0, stream->st[i] >> > (
				d_in,
				d_dout,
				gradient->data,
				input->h,
				input->w,
				input->c,
				d_output->h,
				d_output->w,
				d_output->c,
				gradient->h,
				gradient->w
				);
			checkCuda(cudaStreamSynchronize(stream->st[i]));
		}
	}
	else {
		uint* d_indices = NULL;

		checkCuda(cudaMalloc(&d_indices, sizeof(uint) * d_output->h * d_output->w));
		checkCuda(cudaMemcpy(d_indices, h_indices, sizeof(uint) * d_output->h * d_output->w, cudaMemcpyHostToDevice));

		for (uint i = 0; i < stream->st_size; ++i) {
			float* d_in = input->data + (i * input->h * input->w * input->c);
			float* d_dout = d_output->data + (i * d_output->h * d_output->w * d_output->c);

			__kernel_conv_2d_32x32_g_ind << <blocks, threads, 0, stream->st[i] >> > (
				d_in,
				d_dout,
				gradient->data,
				d_indices,
				input->h,
				input->w,
				input->c,
				d_output->h,
				d_output->w,
				d_output->c,
				gradient->h,
				gradient->w
				);
			checkCuda(cudaStreamSynchronize(stream->st[i]));
		}

		checkCuda(cudaFree(d_indices));
	}

	delete[] h_indices;
}

#endif