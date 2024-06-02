#include "convolution.cuh"
#include "cuda_indice.cuh"
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

__global__ void __conv2d(
	const uint* indice,
	const nn_type* input,
	const nn_type* kernel,
	nn_type* output,
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

/*
__global__ void __kernel_conv2d(
	const uint* indice,
	const float* d_output,
	const float* input,
	float* gradient,
	cuint d_output_c,
	cuint d_output_h,
	cuint d_output_w,
	cuint input_c,
	cuint input_h,
	cuint input_w,
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

	cuint sidx = threadIdx.y * BLOCK_32 + threadIdx.x;

	__shared__ float sm_in[BLOCK_32 * BLOCK_32];
	__shared__ float sm_dout[BLOCK_32 * BLOCK_32];

	const float* p_dout = d_output + (cy * d_output_h * d_output_w);
	const float* p_in = input + (c0 * (input_h * input_w) + y0 * input_w + x0);

	float sum = 0.f;

	for (int i = 0; i < n; i += BLOCK_32) {
		cuint th_x = threadIdx.x + i;
		cuint th_y = threadIdx.y + i;

		__syncthreads();

		sm_dout[sidx] = th_x < n && cy < d_output_c ? p_dout[th_x] : 0.f;
		sm_in[sidx] = cx < k && th_y < n ? p_in[indice[th_y]] : 0.f;

		__syncthreads();

#pragma unroll
		for (int e = 0; e < BLOCK_32; ++e) {
			sum += sm_dout[threadIdx.y * BLOCK_32 + e] * sm_in[e * BLOCK_32 + threadIdx.x];
		}
	}

	if (cx < k && cy < d_output_c) {
		gradient[cy * k + cx] += sum;
	}
}
*/

/**********************************************/
/*                                            */
/*                 NN_Conv2D                  */
/*                                            */
/**********************************************/

cuint* NN_Conv2D::get_indice(const NCHW& in, const NCHW& k) {
	uint* h_idx = new uint[k.c * k.h * k.w];

	for (int c = 0; c < k.c; ++c) {
		cuint kc = (uint)(k.h * k.w * c);
		cuint pc = (uint)(in.h * in.w * c);
		for (int h = 0; h < k.h; ++h) {
			cuint kh = (uint)(k.w * h);
			cuint ph = (uint)(in.w * h);
			for (int w = 0; w < k.w; ++w) {
				h_idx[kc + kh + w] = pc + ph + w;
			}
		}
	}

	set_indice(h_idx, sizeof(uint) * (k.c * k.h * k.w), 0);
	delete[] h_idx;

	return get_indice_ptr();
}

NN_Conv2D::NN_Conv2D(int amounts, const NN_Shape& filter_size, const NN_Shape& stride, Pad pad, const char* name) :
	_amounts(amounts),
	_filter_size(filter_size),
	_stride(stride),
	_pad(pad),
	NN_Layer(name)
{
}

void NN_Conv2D::get_output_shape(const std::vector<NN_Shape>& input_shape, std::vector<NN_Shape>& output_shape) {
	const NN_Shape& shape = input_shape[0];

	if (_pad == Pad::SAME) {
		int n = shape[0];
		int c = _amounts;
		int h = (int)ceil((float)shape[2] / _stride[0]);
		int w = (int)ceil((float)shape[3] / _stride[1]);

		output_shape.push_back({ n, c, h, w });
	}
	else {
		int n = shape[0];
		int c = _amounts;
		int h = (int)floorf((float)(shape[2] - _filter_size[0]) / _stride[0] + 1);
		int w = (int)floorf((float)(shape[3] - _filter_size[1]) / _stride[1] + 1);

		output_shape.push_back({ n, c, h, w });
	}
}

void NN_Conv2D::build(const std::vector<NN_Shape>& input_shape) {
	const NN_Shape& shape = input_shape[0];

	_filter.resize({ _amounts, shape[1], _filter_size[0], _filter_size[1] });
	_bias = GpuTensor<nn_type>::zeros({ _amounts });

	set_random_uniform(_filter, -0.1f, 0.1f);
}

void NN_Conv2D::run_forward(NN_Stream& st, const std::vector<GpuTensor<nn_type>>& input, std::vector<GpuTensor<nn_type>>& output) {
	const GpuTensor<nn_type>& m_input = input[0];
	GpuTensor<nn_type>& m_output = output[0];

	const NCHW in = m_input.get_shape().get_nchw();
	const NCHW out = m_output.get_shape().get_nchw();
	const NCHW k = _filter.get_shape().get_nchw();

	dim3 threads(BLOCK_32, BLOCK_32);
	dim3 blocks = get_grid_size(threads, out.h * out.w, out.c);

	const nn_type* input_data = m_input.get_ptr();
	nn_type* output_data = m_output.get_ptr();
	const nn_type* filter_data = _filter.get_ptr();

	if (_pad == Pad::SAME) {
		NCHW pad = { in.n, in.c, 0, 0 };

		if (_stride[0] == 1) {
			pad.h = in.h - 1 + k.h;
		}
		else {
			pad.h = (in.h / _stride[0]) + k.h;
		}

		if (_stride[1] == 1) {
			pad.w = in.w - 1 + k.w;
		}
		else {
			pad.w = (in.w / _stride[1]) + k.w;
		}

		printf("input=[%d, %d, %d, %d]\n", in.n, in.c, in.h, in.w);
		printf("pad=[%d, %d, %d, %d]\n", pad.n, pad.c, pad.h, pad.w);

		cuint* g_indice = get_indice(pad, k);

		for (int n = 0; n < in.n; ++n) {
			const nn_type* in_data = input_data + (n * in.c * in.h * in.w);
			nn_type* out_data = output_data + (n * out.c * out.h * out.w);

			if (pad.h != in.h || pad.w != in.w) {
				nn_type* pad_space = NULL;

				cudaMallocAsync(&pad_space, sizeof(nn_type) * pad.c * pad.h * pad.w, st[n % STREAMS]);
				cudaMemsetAsync(pad_space, 0, sizeof(nn_type) * pad.c * pad.h * pad.w, st[n % STREAMS]);

				padding_dilation(
					st[n % STREAMS],
					in_data,
					pad_space,
					in,
					pad,
					_stride[1] == 1 ? (pad.w - in.w) / 2 : 0,
					_stride[0] == 1 ? (pad.h - in.h) / 2 : 0,
					1, 1
				);

				//check_cuda(cudaStreamSynchronize(st[n % STREAMS]));
				//check_cuda(cudaGetLastError());

				__conv2d<<<blocks, threads, 0, st[n % STREAMS]>>>(
					g_indice,
					pad_space,
					filter_data,
					out_data,
					pad.h,
					pad.w,
					k.n,
					k.c,
					k.h,
					k.w,
					out.h,
					out.w,
					_stride[0],
					_stride[1]
				);

				//check_cuda(cudaStreamSynchronize(st[n % STREAMS]));
				//check_cuda(cudaGetLastError());

				cudaFreeAsync(pad_space, st[n % STREAMS]);
			}
			else {
				__conv2d<<<blocks, threads, 0, st[n % STREAMS]>>>(
					g_indice,
					in_data,
					filter_data,
					out_data,
					in.h,
					in.w,
					k.n,
					k.c,
					k.h,
					k.w,
					out.h,
					out.w,
					_stride[0],
					_stride[1]
				);

				//check_cuda(cudaStreamSynchronize(st[n % STREAMS]));
				//check_cuda(cudaGetLastError());
			}
		}
	}
	else {
		cuint* g_indice = get_indice(in, k);

		for (int n = 0; n < in.n; ++n) {
			const nn_type* in_data = input_data + (n * in.c * in.h * in.w);
			nn_type* out_data = output_data + (n * out.c * out.h * out.w);

			__conv2d<<<blocks, threads, 0, st[n % STREAMS]>>>(
				g_indice,
				in_data,
				filter_data,
				out_data,
				in.h,
				in.w,
				k.n,
				k.c,
				k.h,
				k.w,
				out.h,
				out.w,
				_stride[0],
				_stride[1]
			);
		}
	}

	add_bias_2d(st, m_output, _bias, m_output);
}

/**********************************************

			     KernelConv2d

**********************************************/
/*
void kernel_conv2d(
	const nn_type* d_output,
	const nn_type* input,
	nn_type* grad,
	const nn_shape& out_shape,
	const nn_shape& in_shape,
	const nn_shape& grad_shape
) {
	cint hw = out_shape[2] * out_shape[3];
	uint* indice = NULL;
	uint* _indice = new uint[hw];

	for (int h = 0; h < out_shape[2]; ++h) {
		for (int w = 0; w < out_shape[3]; ++w) {
			_indice[h * out_shape[3] + w] = h * in_shape[3] + w;
		}
	}

	if (hw > CONST_ELEM_SIZE) {
		check_cuda(cudaMalloc(&indice, sizeof(uint) * hw));
		check_cuda(cudaMemcpy(indice, _indice, sizeof(uint) * hw, cudaMemcpyHostToDevice));
	}
	else {
		indice = get_indice_ptr();
		set_indice(_indice, sizeof(uint) * hw, 0);
	}
	delete[] _indice;

	dim3 threads(BLOCK_32, BLOCK_32);
	dim3 blocks = get_grid_size(threads, grad_shape[1] * grad_shape[2] * grad_shape[3], grad_shape[0]);

	for (int i = 0; i < out_shape[0]; ++i) {
		const nn_type* d_dout = d_output + (i * out_shape[1] * out_shape[2] * out_shape[3]);
		const nn_type* d_input = input + (i * in_shape[1] * in_shape[2] * in_shape[3]);

		__kernel_conv2d<<<blocks, threads>>>(
			indice,
			d_dout,
			d_input,
			grad,
			out_shape[1],
			out_shape[2],
			out_shape[3],
			in_shape[1],
			in_shape[2],
			in_shape[3],
			grad_shape[2],
			grad_shape[3]
		);
	}

	if (hw > CONST_ELEM_SIZE) check_cuda(cudaFree(indice));
}
*/