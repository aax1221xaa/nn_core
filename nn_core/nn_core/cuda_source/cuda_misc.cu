#include "cuda_misc.cuh"

#ifndef __CUDACC__
#define __CUDACC__
#endif

#include <device_functions.h>
#include <device_launch_parameters.h>


/*******************************************
											  
			   kernel functions			  

*******************************************/

__global__ void __transpose(
	const float* input,
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

__global__ void __padding_dilation_2d(
	const float* input,
	float* output,
	cuint in_w,
	cuint in_h,
	cuint in_c,
	cuint out_w,
	cuint out_h,
	cuint stride_x,
	cuint stride_y,
	cuint offset_x,
	cuint offset_y
) {
	cuint in_cx = blockIdx.x * blockDim.x + threadIdx.x;
	cuint in_cy = blockIdx.y * blockDim.y + threadIdx.y;
	cuint in_cz = blockIdx.z;

	cuint out_cx = in_cx * stride_x + offset_x;
	cuint out_cy = in_cy * stride_y + offset_y;

	if (in_cx < in_w && in_cy < in_h) {
		output[in_cz * (out_w * out_h) + out_cy * out_w + out_cx] = input[in_cz * (in_w * in_h) + in_cy * in_w + in_cx];
	}
}

__global__ void __add_bias_32x32(
	const float* data_a,
	const float* data_b,
	float* data_c,
	cuint n,
	cuint c
) {
	cuint cx = blockIdx.x * blockDim.x + threadIdx.x;
	cuint cy = blockIdx.y * blockDim.y + threadIdx.y;

	cuint addr = cy * c + cx;

	__shared__ float share_b[BLOCK_32];

	if (threadIdx.y == 0) share_b[threadIdx.x] = cx < c ? data_b[cx] : 0.f;
	__syncthreads();

	if (cx < c && cy < n) {
		data_c[addr] = data_a[addr] + share_b[threadIdx.x];
	}
}

__global__ void __add_bias_16x16x4(
	const float* data_a,
	const float* data_b,
	float* data_c,
	cuint c,
	cuint h,
	cuint w
) {
	cuint cx = blockIdx.x * blockDim.x + threadIdx.x;
	cuint cy = blockIdx.y * blockDim.y + threadIdx.y;
	cuint cz = blockIdx.z * blockDim.z + threadIdx.z;

	cuint addr = cz * (h * w) + cy * w + cx;

	__shared__ float share_b[BLOCK_4];

	if (threadIdx.x == 0 && threadIdx.y == 0) share_b[threadIdx.z] = cz < c ? data_b[cz] : 0.f;
	__syncthreads();

	if (cx < w && cy < h && cz < c) {
		data_c[addr] = data_a[addr] + share_b[threadIdx.z];
	}
}

__global__ void __add_bias_8x8x16(
	const float* data_a,
	const float* data_b,
	float* data_c,
	cuint c,
	cuint h,
	cuint w
) {
	cuint cx = blockIdx.x * blockDim.x + threadIdx.x;
	cuint cy = blockIdx.y * blockDim.y + threadIdx.y;
	cuint cz = blockIdx.z * blockDim.z + threadIdx.z;

	cuint addr = cz * (h * w) + cy * w + cx;

	__shared__ float share_b[BLOCK_16];

	if (threadIdx.x == 0 && threadIdx.y == 0) share_b[threadIdx.z] = cz < c ? data_b[cz] : 0.f;
	__syncthreads();

	if (cx < w && cy < h && cz < c) {
		data_c[addr] = data_a[addr] + share_b[threadIdx.z];
	}
}

__global__ void __sum_gradient_1d(
	const float* a,
	float* b,
	cuint n,
	cuint c
) {
	cuint cx = blockIdx.x * blockDim.x + threadIdx.x;
	cuint tidx = threadIdx.y * BLOCK_32 + threadIdx.x;

	__shared__ float sm[BLOCK_32 * BLOCK_32];
	
	sm[tidx] = 0.f;
	__syncthreads();

	for (uint i = 0; i < n; i += BLOCK_32) {
		cuint cy = n + threadIdx.y;

		if (cx < c && cy < n) {
			sm[tidx] += a[cy * c + cx];
		}
	}

#pragma unroll
	for (uint i = BLOCK_32 / 2; i > 0; i /= 2) {
		cuint half_side = (threadIdx.y + i) * BLOCK_32 + threadIdx.x;

		__syncthreads();
		if (threadIdx.y < i) sm[tidx] += sm[half_side];
	}

	if (cx < c && threadIdx.y == 0) b[cx] = sm[threadIdx.x];
}

__global__ void __sum_gradient_2d(
	const float* a,
	float* b,
	cuint n,
	cuint c,
	cuint h,
	cuint w
) {
	/*
	threads = [4, 16, 16]
	blocks = [c]
	*/

	__shared__ float sm[BLOCK_4 * BLOCK_16 * BLOCK_16];

	cuint tidx = threadIdx.z * BLOCK_16 * BLOCK_16 + threadIdx.y * BLOCK_16 + threadIdx.x;
	cuint cidx = blockIdx.x * (w * h);

	sm[tidx] = 0.f;
	__syncthreads();

	for (uint z = 0; z < n; z += BLOCK_4) {
		cuint tz = z + threadIdx.z;
		cuint nidx = tz * (c * h * w);

		for (uint y = 0; y < h; y += BLOCK_16) {
			cuint ty = y + threadIdx.y;
			cuint yidx = ty * w;

			for (uint x = 0; x < w; x += BLOCK_16) {
				cuint tx = x + threadIdx.x;

				if (tz < n && ty < h && tx < w) sm[tidx] += a[nidx + cidx + yidx + tx];
			}
		}
	}

#pragma unroll
	for (uint i = BLOCK_1024 / 2; i > 0; i /= 2) {
		__syncthreads();

		if (tidx < i) sm[tidx] += sm[tidx + i];
	}

	b[blockIdx.x] = sm[0];
}

/*******************************************

			    host functions

*******************************************/

void transpose(
	const nn_type* input,
	nn_type* output,
	const nn_shape& in_shape
) {
	dim3 threads(BLOCK_1024);
	dim3 blocks = get_grid_size(threads, in_shape[0] * in_shape[1] * in_shape[2] * in_shape[3]);

	__transpose<<<blocks, threads>>>(
		input,
		output,
		in_shape[0],
		in_shape[1],
		in_shape[2],
		in_shape[3]
	);
}

void padding_dilation(
	cudaStream_t s,
	const nn_type* input,
	nn_type* output,
	const nn_shape& in_shape,
	const nn_shape& out_shape,
	int offset_x,
	int offset_y,
	int stride_x,
	int stride_y
) {
	dim3 threads(BLOCK_32, BLOCK_32);
	dim3 blocks = get_grid_size(threads, in_shape[1], in_shape[2], in_shape[3]);

	__padding_dilation_2d<<<blocks, threads, 0, s>>>(
		input,
		output,
		in_shape[3],
		in_shape[2],
		in_shape[1],
		out_shape[3],
		out_shape[2],
		stride_x,
		stride_y,
		offset_x,
		offset_y
	);
}

void add_bias_1d(
	const nn_type* input,
	const nn_type* bias,
	nn_type* output,
	const nn_shape& in_shape
) {
	dim3 threads(BLOCK_32, BLOCK_32);
	dim3 blocks = get_grid_size(threads, in_shape[1], in_shape[0]);

	__add_bias_32x32<<<blocks, threads>>>(
		input,
		bias,
		output,
		in_shape[0],
		in_shape[1]
	);
}

void add_bias_2d(
	cudaStream_t* s,
	const nn_type* input,
	const nn_type* bias,
	nn_type* output,
	const nn_shape& in_shape,
	const nn_shape& b_shape,
	const nn_shape& out_shape
) {

	if (in_shape[2] >= BLOCK_16 && in_shape[3] >= BLOCK_16 || in_shape[1] <= BLOCK_4) {
		dim3 threads(BLOCK_16, BLOCK_16, BLOCK_4);
		dim3 blocks = get_grid_size(threads, in_shape[3], in_shape[2], in_shape[1]);

		for (uint i = 0; i < in_shape[0]; ++i) {
			const nn_type* d_in = input + (i * in_shape[1] * in_shape[2] * in_shape[3]);
			nn_type* d_out = output + (i * in_shape[1] * in_shape[2] * in_shape[3]);

			__add_bias_16x16x4<<<blocks, threads, 0, s[i % STREAMS]>>>(
				d_in,
				bias,
				d_out,
				in_shape[1],
				in_shape[2],
				in_shape[3]
			);
		}
	}
	else {
		dim3 threads(BLOCK_8, BLOCK_8, BLOCK_16);
		dim3 blocks = get_grid_size(threads, in_shape[3], in_shape[2], in_shape[1]);

		for (uint i = 0; i < in_shape[0]; ++i) {
			const nn_type* d_in = input + (i * in_shape[1] * in_shape[2] * in_shape[3]);
			nn_type* d_out = output + (i * in_shape[1] * in_shape[2] * in_shape[3]);

			__add_bias_8x8x16<<<blocks, threads, 0, s[i % STREAMS]>>>(
				d_in,
				bias,
				d_out,
				in_shape[1],
				in_shape[2],
				in_shape[3]
			);
		}
	}
}

void sum_gradient_1d(
	const nn_type* input,
	nn_type* output,
	const nn_shape& in_shape
) {
	dim3 threads(BLOCK_32, BLOCK_32);
	dim3 blocks = get_grid_size(threads, in_shape[1]);

	__sum_gradient_1d<<<blocks, threads>>>(
		input,
		output,
		in_shape[0],
		in_shape[1]
	);
}

void sum_gradient_2d(
	const nn_type* input,
	nn_type* output,
	const nn_shape& in_shape
) {
	dim3 threads(BLOCK_16, BLOCK_16, BLOCK_4);
	dim3 blocks(in_shape[0]);

	__sum_gradient_2d<<<blocks, threads>>>(
		input,
		output,
		in_shape[0],
		in_shape[1],
		in_shape[2],
		in_shape[3]
	);
}