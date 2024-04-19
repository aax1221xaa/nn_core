#include "maxpool.cuh"
#include "cuda_misc.cuh"

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

__global__ void __maxpool_2d(
	const nn_type* a,
	nn_type* b,
	uint* mark,
	cuint a_h,
	cuint a_w,
	cuint b_h,
	cuint b_w,
	cuint k_h,
	cuint k_w,
	cuint st_h,
	cuint st_w,
	cuint tile_h,
	cuint tile_w
) {
	extern __shared__ nn_type sm[];

	cuint out_x = blockIdx.x * blockDim.x + threadIdx.x;
	cuint out_y = blockIdx.y * blockDim.y + threadIdx.y;
	cuint x0 = blockIdx.x * blockDim.x * st_w;
	cuint y0 = blockIdx.y * blockDim.y * st_h;
	cuint z0 = blockIdx.z;

	const nn_type* pa = a + ((z0 * a_w * a_h) + (y0 * a_w) + x0);
	nn_type* pb = b + (b_w * b_h * z0);
	uint* pmark = mark + (b_w * b_h * z0);

	for (uint h = 0; h < tile_h; h += blockDim.y) {
		cuint ty = threadIdx.y + h;
		cuint in_y = ty + y0;

		for (uint w = 0; w < tile_w; w += blockDim.x) {
			cuint tx = threadIdx.x + w;
			cuint in_x = tx + x0;

			if (tx < tile_w && ty < tile_h && in_x < a_w && in_y < a_h) {
				sm[ty * tile_w + tx] = pa[ty * a_w + tx];
			}
		}
	}
		
	__syncthreads();

	if (out_x < b_w && out_y < b_h) {
		nn_type val = -FLT_MAX;
		uint index = 0;

		for (uint h = 0; h < k_h; ++h) {
			cuint ty = threadIdx.y * st_h + h;
			for (uint w = 0; w < k_w; ++w) {
				cuint tx = threadIdx.x * st_w + w;
				nn_type sm_val = sm[ty * tile_w + tx];

				if (sm_val > val) {
					val = sm_val;
					index = h * k_w + w;
				}
			}
		}

		pb[out_y * b_w + out_x] = val;
		pmark[out_y * b_w + out_x] = index;
	}
}

/*
__global__ void __d_maxpool(
	const float* a,
	const uint* indice,
	float* b,
	cuint a_h,
	cuint a_w,
	cuint b_h,
	cuint b_w,
	cuint k_w,
	cuint st_h,
	cuint st_w
) {
	cuint cx = blockIdx.x * blockDim.x + threadIdx.x;
	cuint cy = blockIdx.y * blockDim.y + threadIdx.y;
	cuint cz = blockIdx.z;

	const float* p_a = a + (cz * (a_h * a_w) + (cy * st_h * a_w) + (cx * st_w));
	const uint* p_indice = indice + (cz * (b_h * b_w) + (cy * b_w) + cx);
	float* p_b = b + (cz * (b_h * b_w) + (cy * b_w) + cx);

	if (cx < b_w && cy < b_h) {
		cuint index = *p_indice;
		cuint x = index % k_w;
		cuint y = index / k_w;

		*p_b = p_a[y * a_w + x];
	}
}
*/

/**********************************************

				    Maxpool2d

**********************************************/

void maxpool2d(
	cudaStream_t s,
	const nn_type* input,
	nn_type* output,
	uint* max_indice,
	cuint in_h,
	cuint in_w,
	cuint out_h,
	cuint out_w,
	cuint h_kernel,
	cuint w_kernel,
	cuint h_stride,
	cuint w_stride,
	cuint h_tile,
	cuint w_tile
) {
	size_t smem_size = sizeof(nn_type) * h_tile * w_tile;
	dim3 threads(BLOCK_32, BLOCK_32);
	dim3 blocks = get_grid_size(threads, in_w, in_h);

	__maxpool_2d<<<blocks, threads, smem_size, s>>>(
		input,
		output,
		max_indice,
		in_h,
		in_w,
		out_h,
		out_w,
		h_kernel,
		w_kernel,
		h_stride,
		w_stride,
		h_tile,
		w_tile
	);
}

/**********************************************

				  D_Maxpool2d

**********************************************/
/*
void d_maxpool2d(
	cudaStream_t* s,
	const nn_type* d_output,
	nn_type* d_input,
	const nn_shape& out_shape,
	const nn_shape& in_shape,
	cuint* max_indice,
	cuint w_kernel,
	cuint h_stride,
	cuint w_stride
) {
	dim3 threads(BLOCK_32, BLOCK_32);
	dim3 blocks = get_grid_size(threads, out_shape[3], out_shape[2], out_shape[1]);

	for (int i = 0; i < out_shape[0]; ++i) {
		const nn_type* d_doutput = d_output + (i * out_shape[1] * out_shape[2] * out_shape[3]);
		cuint* d_indice = max_indice + (i * out_shape[1] * out_shape[2] * out_shape[3]);
		nn_type* d_dinput = d_input + (i * in_shape[1] * in_shape[2] * in_shape[3]);

		__d_maxpool<<<blocks, threads, 0, s[i % STREAMS]>>>(
			d_doutput,
			d_indice,
			d_dinput,
			out_shape[2],
			out_shape[3],
			in_shape[2],
			in_shape[3],
			w_kernel,
			h_stride,
			w_stride
		);
	}
}
*/