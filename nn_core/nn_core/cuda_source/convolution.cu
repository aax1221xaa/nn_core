#define CUDA_API_PER_THREAD_DEFAULT_STEAM 

#include "convolution.cuh"
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
	cuint in_c,
	cuint k_h,
	cuint k_w,
	cuint out_h,
	cuint out_w,
	cuint out_c,
	cuint st_h,
	cuint st_w
) {
	cuint cx = blockIdx.x * blockDim.x + threadIdx.x;
	cuint cy = blockIdx.y * blockDim.y + threadIdx.y;
	cuint sidx = threadIdx.y * BLOCK_32 + threadIdx.x;

	cuint x0 = (cx % out_w) * st_w;
	cuint y0 = (cx / out_w) * st_h;

	cuint k = k_w * k_h * in_c;
	cuint n = out_w * out_h;

	__shared__ nn_type share_in[BLOCK_32 * BLOCK_32];
	__shared__ nn_type share_k[BLOCK_32 * BLOCK_32];

	const nn_type* p_input = input + (y0 * in_w + x0);
	const nn_type* p_kernel = kernel + (cy * k_w * k_h * in_c);

	nn_type sum = 0.f;

	for (uint i = 0; i < k; i += BLOCK_32) {
		uint th_x = i + threadIdx.x;
		uint th_y = i + threadIdx.y;

		__syncthreads();

		share_k[sidx] = th_x < k && cy < out_c ? p_kernel[th_x] : 0.f;
		share_in[sidx] = cx < n && th_y < k ? p_input[indice[th_y]] : 0.f;

		__syncthreads();

#pragma unroll
		for (uint e = 0; e < BLOCK_32; ++e) {
			sum += share_in[e * BLOCK_32 + threadIdx.x] * share_k[threadIdx.y * BLOCK_32 + e];
		}
	}

	if (cx < n && cy < out_c) {
		output[cy * k + cx] = sum;
	}
}

#endif
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

void NN_Conv2D::set_indice(const NN_Tensor4dShape& in, const NN_Filter4dShape& k) {
	uint* h_idx = new uint[k._h * k._w * k._in_c];

	for (int h = 0; h < k._h; ++h) {
		cuint kh = k._w * k._in_c * h;
		cuint in_h = in._w * in._c * h;
		for (int w = 0; w < k._w; ++w) {
			cuint kw = k._in_c * w;
			cuint in_w = in._c * w;
			for (int c = 0; c < k._in_c; ++c) {
				h_idx[kh + kw + c] = in_h + in_w + c;
			}
		}
	}

	set_const_mem(h_idx, k._h * k._w * k._in_c, 0);

	delete[] h_idx;
}

NN_Conv2D::NN_Conv2D(cuint amounts, const NN_Shape& filter_size, const NN_Shape& stride, const std::string& pad, const std::string& name) :
	_amounts(amounts),
	_filter_size(filter_size),
	_stride(stride),
	_pad(pad),
	NN_Layer(name)
{
}

void NN_Conv2D::get_output_shape(const NN_List<NN_Shape>& input_shape, NN_List<NN_Shape>& output_shape) {
	const NN_Shape& shape = input_shape[0].val();

	if (_pad == "same") {
		int n = shape[0];
		int h = (int)ceil((float)shape[1] / _stride[1]);
		int w = (int)ceil((float)shape[2] / _stride[0]);
		int c = _amounts;

		output_shape.append(NN_Shape({ n, h, w, c }));
	}
	else {
		int n = shape[0];
		int h = (int)floorf((float)(shape[1] - _filter_size[1]) / _stride[1] + 1);
		int w = (int)floorf((float)(shape[2] - _filter_size[0]) / _stride[0] + 1);
		int c = _amounts;

		output_shape.append(NN_Shape({ n, h, w, c }));
	}
}

void NN_Conv2D::build(const NN_List<NN_Shape>& input_shape, NN_List<GpuTensor<nn_type>>& weights) {
	const NN_Shape& shape = input_shape[0].val();
	

	_filter.resize(NN_Shape({ _filter_size[1], _filter_size[0], shape[3], _amounts }));
	_bias = GpuTensor<nn_type>::zeros({ _amounts });

	set_random_uniform(_filter, -0.1f, 0.1f);

	weights.append(_filter);
	weights.append(_bias);
}

void NN_Conv2D::run(NN_Stream& st, const NN_List<GpuTensor<nn_type>>& input, NN_List<GpuTensor<nn_type>>& output) {
	const GpuTensor<nn_type>& m_input = input[0].val();
	GpuTensor<nn_type>& m_output = output[0].val();

	//std::cout << "input: " << m_input.get_shape();
	//std::cout << "kernel: " << _filter.get_shape();
	//std::cout << "output: " << m_output.get_shape();

	const NN_Tensor4dShape& in = m_input.get_shape().get_4d_shape();
	const NN_Tensor4dShape& out = m_output.get_shape().get_4d_shape();
	const NN_Filter4dShape& k = _filter.get_shape().get_filter_shape();

	//printf("input: [%d, %d, %d, %d]\n", in.n, in.c, in.h, in.w);
	//printf("kernel: [%d, %d, %d, %d]\n", k.n, k.c, k.h, k.w);
	//printf("output: [%d, %d, %d, %d]\n", out.n, out.c, out.h, out.w);

	dim3 threads(BLOCK_32, BLOCK_32);
	dim3 blocks = get_grid_size(threads, out._h * out._w * out._c);

	const nn_type* input_data = m_input.get_ptr();
	nn_type* output_data = m_output.get_ptr();
	const nn_type* filter_data = _filter.get_ptr();

	cudaStream_t* p_st = st.get_stream();

	if (_pad == "same") {
		NN_Tensor4dShape pad = in;

		pad._h = 0;
		pad._w = 0;

		if (_stride[0] == 1) {
			pad._w = in._w - 1 + k._w;
		}
		else {
			pad._w = (in._w / _stride[0]) + k._w;
		}

		if (_stride[1] == 1) {
			pad._h = in._h - 1 + k._h;
		}
		else {
			pad._h = (in._h / _stride[1]) + k._h;
		}

		set_indice(pad, k);
		cuint* c_indice = get_const_mem(k._in_c * k._h * k._w, 0);

		for (uint n = 0; n < (uint)in._n; ++n) {
			cuint in_index = n * in._h * in._w * in._c;
			cuint out_index = n * out._h * out._w * out._c;

			const nn_type* in_data = input_data + in_index;
			nn_type* out_data = output_data + out_index;

			if (pad._h != in._h || pad._w != in._w) {
				nn_type* pad_space = NULL;
				cuint pad_size = pad._h * pad._w * pad._c;

				cudaMallocAsync(&pad_space, sizeof(nn_type) * pad_size, p_st[n % STREAMS]);
				cudaMemsetAsync(pad_space, 0, sizeof(nn_type) * pad_size, p_st[n % STREAMS]);

				padding_dilation(
					p_st[n % STREAMS],
					in_data,
					pad_space,
					in,
					pad,
					_stride[1] == 1 ? (pad._w - in._w) / 2 : 0,
					_stride[0] == 1 ? (pad._h - in._h) / 2 : 0,
					1, 1
				);

				//check_cuda(cudaStreamSynchronize(st[n % STREAMS]));
				//check_cuda(cudaGetLastError());

				__conv2d<<<blocks, threads, 0, p_st[n % STREAMS]>>>(
					pad_space,
					filter_data,
					out_data,
					c_indice,
					(uint)in._h,
					(uint)in._w,
					(uint)in._c,
					(uint)k._h,
					(uint)k._w,
					(uint)out._h,
					(uint)out._w,
					(uint)out._c,
					(uint)_stride[1],
					(uint)_stride[0]
					);

				//check_cuda(cudaStreamSynchronize(st[n % STREAMS]));
				//check_cuda(cudaGetLastError());

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
					(uint)in._c,
					(uint)k._h,
					(uint)k._w,
					(uint)out._h,
					(uint)out._w,
					(uint)out._c,
					(uint)_stride[1],
					(uint)_stride[0]
				);

				//check_cuda(cudaStreamSynchronize(st[n % STREAMS]));
				//check_cuda(cudaGetLastError());
			}
		}
	}
	else {
		set_indice(in, k);
		cuint* c_indice = get_const_mem(k._in_c * k._h * k._w, 0);

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
				(uint)in._c,
				(uint)k._h,
				(uint)k._w,
				(uint)out._h,
				(uint)out._w,
				(uint)out._c,
				(uint)_stride[1],
				(uint)_stride[0]
				);
		}

		//Tensor<nn_type> tmp(_filter.get_shape());
		//tmp = _filter;

		//std::cout << std::endl << tmp;
	}

	add_bias_2d(st, m_output, _bias, m_output);
}

NN_Backward* NN_Conv2D::create_backward(std::vector<bool>& mask) {
	return new NN_dConv2D(*this);
}

NN_List<GpuTensor<nn_type>> NN_Conv2D::get_weight() {
	return { _filter, _bias };
}


/**********************************************/
/*                                            */
/*                 NN_dConv2D                 */
/*                                            */
/**********************************************/

NN_dConv2D::NN_dConv2D(NN_Conv2D& layer) :
	NN_Backward_t(layer)
{
}

void NN_dConv2D::run(
	NN_Stream& st,
	const NN_List<GpuTensor<nn_type>>& input,
	const NN_List<GpuTensor<nn_type>>& doutput,
	NN_List<GpuTensor<nn_type>>& dinput
) {

}

NN_Optimizer* NN_dConv2D::create_optimizer(const NN_Optimizer& optimizer) {
	return optimizer.create({ _layer._filter, _layer._bias });
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