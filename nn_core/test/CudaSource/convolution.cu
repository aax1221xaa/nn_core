#include "convolution.cuh"
#include "cuda_misc.cuh"

/*
#ifndef __CUDACC__
#define __CUDACC__
#endif

#include <device_functions.h>
#include <device_launch_parameters.h>
*/

#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>


/**********************************************/
/*											  */
/*				 kernel function			  */
/*										      */
/**********************************************/
#if 0
__global__ void __conv2d(
	const nn_type* input,
	const nn_type* kernel,
	nn_type* output,
	cuint* indice,
	cuint in_h,
	cuint in_w,
	cuint k_n,
	cuint k_c,
	cuint k_h,
	cuint k_w,
	cuint out_h,
	cuint out_w,
	cuint st_h,
	cuint st_w
) {
	cuint cx = blockIdx.x * blockDim.x + threadIdx.x;
	cuint cy = blockIdx.y * blockDim.y + threadIdx.y;
	cuint sidx = threadIdx.y * BLOCK_32 + threadIdx.x;

	cuint x0 = (cx % out_w) * st_w;
	cuint y0 = (cx / out_w) * st_h;

	cuint n = k_w * k_h * k_c;
	cuint k = out_w * out_h;

	__shared__ nn_type share_in[BLOCK_32 * BLOCK_32];
	__shared__ nn_type share_k[BLOCK_32 * BLOCK_32];

	const nn_type* p_input = input + (y0 * in_w + x0);
	const nn_type* p_kernel = kernel + (cy * k_w * k_h * k_c);

	nn_type sum = 0.f;

	for (uint i = 0; i < n; i += BLOCK_32) {
		uint th_x = i + threadIdx.x;
		uint th_y = i + threadIdx.y;

		__syncthreads();

		share_k[sidx] = th_x < n && cy < k_n ? p_kernel[th_x] : 0.f;
		share_in[sidx] = cx < k && th_y < n ? p_input[indice[th_y]] : 0.f;

		__syncthreads();

#pragma unroll
		for (uint e = 0; e < BLOCK_32; ++e) {
			sum += share_in[e * BLOCK_32 + threadIdx.x] * share_k[threadIdx.y * BLOCK_32 + e];
		}
	}

	if (cx < k && cy < k_n) {
		output[cy * k + cx] = sum;
	}
}

#else

__global__ void __conv2d(
	const nn_type* input,
	const nn_type* kernel,
	nn_type* output,
	cuint* indice,
	cuint in_h,
	cuint in_w,
	cuint k_h,
	cuint k_w,
	cuint k_ic,
	cuint k_oc,
	cuint out_h,
	cuint out_w,
	cuint st_h,
	cuint st_w
) {
	cuint cx = blockIdx.x * blockDim.x + threadIdx.x;
	cuint cy = blockIdx.y * blockDim.y + threadIdx.y;												
	cuint tidx = threadIdx.y * BLOCK_32 + threadIdx.x;

	cuint out_x = cx % out_w;
	cuint out_y = cx / out_w;
	cuint in_x = out_x * st_w;
	cuint in_y = out_y * st_h;

	cuint k = k_h * k_w * k_ic;
	cuint n = out_w * out_h;

	__shared__ nn_type share_in[BLOCK_32 * BLOCK_32];
	__shared__ nn_type share_k[BLOCK_32 * BLOCK_32];

	const nn_type* p_input = input + ((in_y * in_w * k_ic) + (in_x * k_ic));
	const nn_type* p_kernel = kernel + cy;
	nn_type* p_output = output + ((out_y * out_w * k_oc) + (out_x * k_oc) + cy);

	nn_type sum = 0.f;

	for (uint i = 0; i < k; i += BLOCK_32) {
		uint th_x = i + threadIdx.x;
		uint th_y = i + threadIdx.y;

		__syncthreads();

		//share_k[tidx] = th_y < k && cy < k_oc ? p_kernel[th_y * k_oc] : 0.f;
		//share_in[tidx] = th_x < k && cx < n ? p_input[indice[th_x]] : 0.f;
		share_k[tidx] = th_x < k && cy < k_oc ? p_kernel[th_x * k_oc] : 0.f;
		share_in[tidx] = th_y < k && cx < n ? p_input[indice[th_y]] : 0.f;

		__syncthreads();

#pragma unroll
		for (uint e = 0; e < BLOCK_32; ++e) {
			//sum += share_in[threadIdx.y * BLOCK_32 + e] * share_k[e * BLOCK_32 + threadIdx.x];
			sum += share_in[e * BLOCK_32 + threadIdx.x] * share_k[threadIdx.y * BLOCK_32 + e];
		}
	}

	if (cx < n && cy < k_oc) *p_output = sum;
}

#endif


/**********************************************/
/*                                            */
/*                 NN_Conv2D                  */
/*                                            */
/**********************************************/

cuint* _set_indice(const NN_Tensor4dShape& in, const NN_Filter4dShape& k) {
	uint* h_idx = new uint[k._h * k._w * in._c];

	for (int h = 0; h < k._h; ++h) {
		cuint kh = k._w * in._c * h;
		cuint in_h = in._w * in._c * h;
		for (int w = 0; w < k._w; ++w) {
			cuint kw = in._c * w;
			cuint in_w = in._c * w;
			for (int c = 0; c < in._c; ++c) {
				h_idx[kh + kw + c] = (uint)(in_h + in_w + c);
			}
		}
	}

	cuint* g_idx = set_const_mem(h_idx, k._h * k._w * in._c, 0);

	delete[] h_idx;

	return g_idx;
}

NN_Shape get_out_shape(
	const NN_Shape& in_shape,
	int amounts,
	const NN_Shape&& f_shape,
	const NN_Shape&& stride,
	const std::string&& pad
) {
	int n = 0;
	int h = 0;
	int w = 0;
	int c = 0;

	if (pad == "same") {
		n = in_shape[0];
		h = (int)ceil((float)in_shape[1] / stride[1]);
		w = (int)ceil((float)in_shape[2] / stride[0]);
		c = amounts;
	}
	else {
		n = in_shape[0];
		h = (int)floorf((float)(in_shape[1] - f_shape[1]) / stride[1] + 1);
		w = (int)floorf((float)(in_shape[2] - f_shape[0]) / stride[0] + 1);
		c = amounts;
	}

	return NN_Shape({ n, h, w, c });
}

void conv2d(
	NN_Stream& st,
	const GpuTensor<nn_type>& src,
	const GpuTensor<nn_type>& weight,
	GpuTensor<nn_type>& dst,
	const std::string&& pad,
	const NN_Shape&& stride
) {
	//std::cout << "input: " << m_input.get_shape();
	//std::cout << "kernel: " << _filter.get_shape();
	//std::cout << "output: " << m_output.get_shape();

	const NN_Tensor4dShape in = src.get_shape().get_4d_shape();
	const NN_Tensor4dShape out = dst.get_shape().get_4d_shape();
	const NN_Filter4dShape k = weight.get_shape().get_filter_shape();

	//printf("input: [%d, %d, %d, %d]\n", in.n, in.c, in.h, in.w);
	//printf("kernel: [%d, %d, %d, %d]\n", k.n, k.c, k.h, k.w);
	//printf("output: [%d, %d, %d, %d]\n", out.n, out.c, out.h, out.w);

	dim3 threads(BLOCK_32, BLOCK_32);
	dim3 blocks = get_grid_size(threads, out._w * out._h, out._c);

	const nn_type* input_data = src.get_ptr();
	nn_type* output_data = dst.get_ptr();
	const nn_type* filter_data = weight.get_ptr();

	cudaStream_t* p_st = st.get_stream();

	if (pad == "same") {
		NN_Tensor4dShape pad_shape = in;

		if (stride[0] == 1) pad_shape._w = in._w - 1 + k._w;
		else pad_shape._w = (in._w / stride[0]) + k._w;

		if (stride[1] == 1) pad_shape._h = in._h - 1 + k._h;
		else pad_shape._h = (in._h / stride[1]) + k._h;

		cuint* c_indice = _set_indice(pad_shape, k);

		for (uint n = 0; n < (uint)in._n; ++n) {
			const nn_type* in_data = input_data + (n * in._h * in._w * in._c);
			nn_type* out_data = output_data + (n * out._h * out._w * out._c);

			if (pad_shape._h != in._h || pad_shape._w != in._w) {
				nn_type* pad_space = NULL;
				cuint pad_size = pad_shape._h * pad_shape._w * pad_shape._c;

				check_cuda(cudaMallocAsync((void**)&pad_space, sizeof(nn_type) * pad_size, p_st[n % STREAMS]));
				check_cuda(cudaMemsetAsync(pad_space, 0, sizeof(nn_type) * pad_size, p_st[n % STREAMS]));

				padding_dilation(
					p_st[n % STREAMS],
					in_data,
					pad_space,
					in,
					pad_shape,
					stride[1] == 1 ? (pad_shape._w - in._w) / 2 : 0,
					stride[0] == 1 ? (pad_shape._h - in._h) / 2 : 0,
					stride[1],
					stride[0]
				);
#if _DEBUG
				check_cuda(cudaStreamSynchronize(st[n % STREAMS]));
				check_cuda(cudaGetLastError());
#endif
				__conv2d<<<blocks, threads, 0, p_st[n % STREAMS]>>>(
					pad_space,
					filter_data,
					out_data,
					c_indice,
					(uint)pad_shape._h,
					(uint)pad_shape._w,
					(uint)k._h,
					(uint)k._w,
					(uint)k._in_c,
					(uint)k._out_c,
					(uint)out._h,
					(uint)out._w,
					(uint)stride[1],
					(uint)stride[0]
					);
#if _DEBUG
				check_cuda(cudaStreamSynchronize(st[n % STREAMS]));
				check_cuda(cudaGetLastError());
#endif
				cudaFreeAsync(pad_space, p_st[n % STREAMS]);
			}
			else {
				__conv2d<<<blocks, threads, 0, p_st[n % STREAMS]>>>(
					in_data,
					filter_data,
					out_data,
					c_indice,
					(uint)in._h,
					(uint)in._w,
					(uint)k._h,
					(uint)k._w,
					(uint)k._in_c,
					(uint)k._out_c,
					(uint)out._h,
					(uint)out._w,
					(uint)stride[1],
					(uint)stride[0]
				);
#if _DEBUG
				check_cuda(cudaStreamSynchronize(st[n % STREAMS]));
				check_cuda(cudaGetLastError());
#endif
			}
		}
	}
	else {
		cuint* c_indice = _set_indice(in, k);

		for (uint n = 0; n < (uint)in._n; ++n) {
			const nn_type* in_data = input_data + (n * in._c * in._h * in._w);
			nn_type* out_data = output_data + (n * out._c * out._h * out._w);

			__conv2d<<<blocks, threads, 0, p_st[n % STREAMS]>>>(
				in_data,
				filter_data,
				out_data,
				c_indice,
				(uint)in._h,
				(uint)in._w,
				(uint)k._h,
				(uint)k._w,
				(uint)k._in_c,
				(uint)k._out_c,
				(uint)out._h,
				(uint)out._w,
				(uint)stride[1],
				(uint)stride[0]
				);
#if _DEBUG
			check_cuda(cudaStreamSynchronize(p_st[n % STREAMS]));
			check_cuda(cudaGetLastError());
#endif
		}
		//Tensor<nn_type> tmp(_filter.get_shape());
		//tmp = _filter;

		//std::cout << std::endl << tmp;
	}
}