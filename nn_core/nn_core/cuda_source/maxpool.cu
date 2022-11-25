#include "maxpool.cuh"

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
	float* a,
	float* b,
	const uint a_w,
	const uint a_h,
	const uint b_w,
	const uint b_h,
	const uint ch,
	const uint k_w,
	const uint k_h,
	const uint st_w,
	const uint st_h,
	const uint sh_w,
	const uint sh_h
) {
	extern __shared__ float smem[];

	uint ocx = blockIdx.x * blockDim.x + threadIdx.x;
	uint ocy = blockIdx.y * blockDim.y + threadIdx.y;
	uint block_x = blockIdx.x * blockDim.x;
	uint block_y = blockIdx.y * blockDim.y;

	for (int c = 0; c < ch; ++c) {
		float* pa = a + (a_w * a_h) * c + (sh_h * block_y) * a_w + (sh_w * block_x);
		float* pb = b + (b_w * b_h) * c;
		
		__syncthreads();

		for (uint h = 0; h < sh_h; h += blockDim.y) {
			uint icy = ocy + h;
			uint sy = threadIdx.y + h;
			for (int w = 0; w < sh_w; w += blockDim.x) {
				uint icx = ocx + w;
				uint sx = threadIdx.x + w;

				if (sx < sh_w && sy < sh_h) {
					smem[sy * sh_w + sx] = icy < a_h && icx < a_w ? pa[sy * a_w + sx] : 0.f;
				}
			}
		}
		
		__syncthreads();

		if (ocx < b_w && ocy < b_h) {
			float val = 0.f;

			for (uint h = 0; h < k_h; ++h) {
				uint sy = threadIdx.y * st_h + h;
				for (uint w = 0; w < k_w; ++w) {
					uint sx = threadIdx.x * st_w + w;
					val = __max(val, smem[sy * sh_w + sx]);
				}
			}

			pb[ocy * b_w + ocx] = val;
		}
	}
}



/**********************************************/
/*											  */
/*				  host function 			  */
/*										      */
/**********************************************/

int calc_shared_mem_size(
	int kernel_size,
	int strides
) {
	return (BLOCK_SIZE_32 - 1) * strides + kernel_size;
}

int calc_output_size(
	int input_size,
	int k_size,
	int strides
) {
	return (input_size - k_size) / strides + 1;
}

void maxpool_2d(
	Stream& stream,
	Tensor& input,
	Tensor& output,
	int kernel_w,
	int kernel_h,
	int stride_w,
	int stride_h
) {
	int out_w = calc_output_size(input.w, kernel_w, stride_w);
	int out_h = calc_output_size(input.h, kernel_h, stride_h);

	if (input.n != output.n || out_w != output.w || out_h != output.h || input.c != output.c) {
		ErrorExcept(
			"[maxpool_2d] invalid output size. %s",
			dim_to_str(output)
		);
	}

	int share_w = calc_shared_mem_size(kernel_w, stride_w);
	int share_h = calc_shared_mem_size(kernel_h, stride_h);

	size_t smem_size = sizeof(float) * share_w * share_h;

	dim3 threads(BLOCK_SIZE_32, BLOCK_SIZE_32);
	dim3 blocks = get_grid_size(threads, output.w, output.h);
	
	for (int i = 0; i < stream.str_size; ++i) {
		float* d_in = input.data + (i * input.h * input.w * input.c);
		float* d_out = output.data + (i * output.h * output.w * output.c);

		__maxpool_2d << <blocks, threads, smem_size, stream.str[i] >> > (
			d_in,
			d_out,
			input.w,
			input.h,
			output.w,
			output.h,
			input.c,
			kernel_w,
			kernel_h,
			stride_w,
			stride_h,
			share_w,
			share_h
			);
	}

	sync_streams(stream);
}