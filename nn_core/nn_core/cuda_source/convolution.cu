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

__global__ void __conv_2d(
	const uint* indice,
	const float* input,
	const float* kernel,
	float* output,
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
	cuint sidx = threadIdx.y * BLOCK_SIZE + threadIdx.x;

	cuint x0 = (cx % out_w) * st_w;
	cuint y0 = (cx / out_w) * st_h;

	cuint n = k_w * k_h * k_c;
	cuint k = out_w * out_h;

	__shared__ float share_in[BLOCK_SIZE * BLOCK_SIZE];
	__shared__ float share_k[BLOCK_SIZE * BLOCK_SIZE];

	const float* p_input = input + (y0 * in_w + x0);
	const float* p_kernel = kernel + (cy * k_w * k_h * k_c);

	float sum = 0.f;

	for (uint i = 0; i < n; i += BLOCK_SIZE) {
		uint th_x = i + threadIdx.x;
		uint th_y = i + threadIdx.y;

		__syncthreads();

		share_k[sidx] = th_x < n && cy < k_n ? p_kernel[th_x] : 0.f;
		share_in[sidx] = cx < k && th_y < n ? p_input[indice[th_y]] : 0.f;

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

__global__ void __kernel_conv_2d(
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

	cuint sidx = threadIdx.y * BLOCK_SIZE + threadIdx.x;

	__shared__ float sm_in[BLOCK_SIZE * BLOCK_SIZE];
	__shared__ float sm_dout[BLOCK_SIZE * BLOCK_SIZE];

	const float* p_dout = d_output + (cy * d_output_h * d_output_w);
	const float* p_in = input + (c0 * (input_h * input_w) + y0 * input_w + x0);

	float sum = 0.f;

	for (int i = 0; i < n; i += BLOCK_SIZE) {
		cuint th_x = threadIdx.x + i;
		cuint th_y = threadIdx.y + i;

		__syncthreads();

		sm_dout[sidx] = th_x < n && cy < d_output_c ? p_dout[th_x] : 0.f;
		sm_in[sidx] = cx < k && th_y < n ? p_in[indice[th_y]] : 0.f;

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

/**********************************************

				  conv2dParam

**********************************************/

conv2dParam::conv2dParam() :
	_w_stride(1),
	_h_stride(1),
	_mode(Pad::VALID)
{
}

conv2dParam::conv2dParam(int w_stride, int h_stride, Pad mode) :
	_w_stride(w_stride),
	_h_stride(h_stride),
	_mode(mode)
{
}

void conv2dParam::set(int w_stride, int h_stride, Pad mode) {
	_w_stride = w_stride;
	_h_stride = h_stride;
	_mode = mode;
}

const bool conv2dParam::is_valid() const {
	return (_w_stride > 0 || _h_stride > 0);
}

/**********************************************

				 conv2dSolution

**********************************************/

conv2dSolution::conv2dSolution(const tensor4d& input, const tensor4d& kernel, const conv2dParam& param) :
	_input(input),
	_kernel(kernel),
	_param(param)
{
}

const tensor4d conv2dSolution::calculate_size() {
	_is_calculated = false;

	if (!_input.is_valid() || !_kernel.is_valid() || _input._c != _kernel._c) {
		ErrorExcept(
			"[conv2dSolution::calculate_size()] invalid input, kernel arguments. input: %s, kernel: %s",
			tensor4d::shape_to_str(_input),
			tensor4d::shape_to_str(_kernel)
		);
	}

	if (_param._mode == Pad::SAME) {
		int out_n = _input._n;
		int out_c = _kernel._n;
		int out_h = _input._h;
		int out_w = _input._w;

		_output.set(out_n, out_c, out_h, out_w);

		int pad_n = __min(STREAMS, _input._n);
		int pad_c = _input._c;
		int pad_h = (_input._h - 1) * _param._h_stride + _kernel._h;
		int pad_w = (_input._w - 1) * _param._w_stride + _kernel._w;

		_pad.set(pad_n, pad_c, pad_h, pad_w);
	}
	else {
		int out_n = _input._n;
		int out_c = _kernel._n;
		int out_h = (_input._h - _kernel._h) / _param._h_stride + 1;
		int out_w = (_input._w - _kernel._w) / _param._w_stride + 1;

		_output.set(out_n, out_c, out_h, out_w);
		_pad = tensor4d();
	}

	if (!_output.is_valid()) {
		ErrorExcept(
			"[conv2dSolution::calculate_size()] calculated invalid output size. output: %s",
			tensor4d::shape_to_str(_output)
		);
	}
	else if (_param._mode == Pad::VALID && _pad.is_valid()) {
		ErrorExcept(
			"[conv2dSolution::calculate_size()] calculated invalid pad size. pad: %s",
			tensor4d::shape_to_str(_pad)
		);
	}

	_is_calculated = true;

	return _output;
}

const size_t conv2dSolution::get_workspace_size() {
	if (!_is_calculated) {
		ErrorExcept(
			"[conv2dSolution::get_workspace()] not calculated sizes."
		);
	}

	return _pad.get_size();
}

void conv2dSolution::operator()(cudaStream_t* s, const nn_type* input, const nn_type* kernel, nn_type* output, void* workspace) {
	if (!_is_calculated) {
		ErrorExcept(
			"[cudaConv2d::get_output_param()] not calculated convolution sizes."
		);
	}

	uint* indice = (uint*)malloc(_kernel.get_size());
	const uint* c_indice = get_indice_ptr();

	if (_param._mode == Pad::SAME) {
		for (int c = 0; c < _kernel._c; ++c) {
			for (int h = 0; h < _kernel._h; ++h) {
				for (int w = 0; w < _kernel._w; ++w) {
					indice[c * (_kernel._h * _kernel._w) + h * _kernel._w + w] = (uint)(c * (_pad._h * _pad._w) + h * _pad._w + w);
				}
			}
		}
		set_indice(indice, _kernel.get_size(), 0);
		free(indice);

		check_cuda(cudaMemset(workspace, 0, _pad.get_size()));

		for (int n = 0; n < _input._n; ++n) {
			const nn_type* d_input = input + (n * _input._c * _input._h * _input._w);
			nn_type* d_output = output + (n * _output._c * _output._h * _output._w);
			nn_type* d_pad = (nn_type*)workspace + ((n % STREAMS) * _pad._c * _pad._h * _pad._w);

			dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
			dim3 blocks = get_grid_size(threads, _input._w, _input._h, _input._c);

			__padding_dilation_2d<<<blocks, threads, 0, s[n % STREAMS]>>>(
				d_input,
				d_pad,
				_input._w,
				_input._h,
				_input._c,
				_pad._w,
				_pad._h,
				1, 1,
				(_pad._w - _input._w) / 2,
				(_pad._h - _input._h) / 2
			);

			blocks = get_grid_size(threads, _output._h * _output._w, _output._c);

			__conv_2d<<<blocks, threads, 0, s[n % STREAMS]>>>(
				c_indice,
				d_pad,
				kernel,
				d_output,
				_pad._h,
				_pad._w,
				_kernel._n,
				_kernel._c,
				_kernel._h,
				_kernel._w,
				_output._h,
				_output._w,
				_conv._h_st,
				_conv._w_st
			);
		}
	}
	else {
		for (int c = 0; c < _kernel._c; ++c) {
			for (int h = 0; h < _kernel._h; ++h) {
				for (int w = 0; w < _kernel._w; ++w) {
					indice[c * (_kernel._h * _kernel._w) + h * _kernel._w + w] = (uint)(c * (_input._h * _input._w) + h * _input._w + w);
				}
			}
		}
		set_indice(indice, _kernel.get_size(), 0);
		free(indice);

		for (int n = 0; n < _input._n; ++n) {
			const nn_type* d_input = input + (n * _input._c * _input._h * _input._w);
			nn_type* d_output = output + (n * _output._c * _output._h * _output._w);

			dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
			dim3 blocks = get_grid_size(threads, _output._h * _output._w, _output._c);

			__conv_2d<<<blocks, threads, 0, s[n % STREAMS]>>>(
				c_indice,
				d_input,
				kernel,
				d_output,
				_input._h,
				_input._w,
				_kernel._n,
				_kernel._c,
				_kernel._h,
				_kernel._w,
				_output._h,
				_output._w,
				_h_stride,
				_w_stride
			);
		}
	}
}

/**********************************************

				 dConv2dSolution

**********************************************/

dConv2dSolution::dConv2dSolution(const tensor4d& d_output, const conv2dSolution& conv) :
	_d_output(d_output),
	_conv(conv)
{
}

const tensor4d dConv2dSolution::calculate_size() {
	_is_calculated = false;

	if (_d_output._n != _conv._output._n ||
		_d_output._c != _conv._output._c ||
		_d_output._h != _conv._output._h ||
		_d_output._w != _conv._output._w) {
		ErrorExcept(
			"[dConv2dSolution::calculate_size()] invalid d_output size. %s",
			tensor4d::shape_to_str(_d_output)
		);
	}

	int in_n = _conv._input._n;
	int in_c = _conv._input._c;
	int in_h = _conv._input._h;
	int in_w = _conv._input._w;

	_d_input.set(in_n, in_c, in_h, in_w);

	int pad_n = __min(STREAMS, in_n);
	int pad_c = in_c;
	int pad_h = (_d_output._h * _conv._param._h_stride) + _conv._kernel._h - 1;
	int pad_w = (_d_output._w * _conv._param._w_stride) + _conv._kernel._w - 1;

	_d_pad.set(pad_n, pad_c, pad_h, pad_w);

	if (!_d_output.is_valid() || !_d_input.is_valid() || !_d_pad.is_valid()) {
		ErrorExcept(
			"[dConv2dSolution::calculate_size()] mismatched calcute sizes. d_output: %s, d_pad: %s, d_input: %s",
			tensor4d::shape_to_str(_d_output),
			tensor4d::shape_to_str(_d_pad),
			tensor4d::shape_to_str(_d_input)
		);
	}

	_is_calculated = true;

	return _d_input;
}

const size_t dConv2dSolution::get_workspace_size() {
	if (!_is_calculated) {
		ErrorExcept(
			"[dConv2dSolution::get_workspace()] not calculated sizes."
		);
	}

	return _conv._kernel.get_size() + _d_pad.get_size();
}

void dConv2dSolution::operator()(cudaStream_t* s, const nn_type* d_output, const nn_type* kernel, nn_type* d_input, void* workspace) {
	if (!_is_calculated) {
		ErrorExcept(
			"[dConv2dSolution::operator()] not calculated sizes."
		);
	}

	const tensor4d& _kernel = _conv._kernel;

	uint* indice = (uint*)malloc(sizeof(uint) * _kernel._n * _kernel._h * _kernel._w);
	const uint* c_indice = get_indice_ptr();

	for (int n = 0; n < _kernel._n; ++n) {
		for (int h = 0, _h = _kernel._h - 1; h < _kernel._h; ++h, --_h) {
			for (int w = 0, _w = _kernel._w - 1; w < _kernel._w; ++w, --_w) {
				indice[n * (_kernel._h * _kernel._w) + h * _kernel._w + w] = (uint)(n * (_d_pad._h * _d_pad._w) + _h * _d_pad._w + _w);
			}
		}
	}
	set_indice(indice, sizeof(uint) * _kernel._n * _kernel._h * _kernel._w, 0);
	free(indice);

	nn_type* p_kernel = (nn_type*)workspace;
	nn_type* p_pad = (nn_type*)((char*)workspace + _kernel.get_size());

	check_cuda(cudaMemset(p_pad, 0, _d_pad.get_size()));

	transpose(_kernel, kernel, p_kernel);

	for (uint n = 0; n < (uint)_d_output._n; ++n) {
		const nn_type* d_out = d_output + (n * _d_output._c * _d_output._h * _d_output._w);
		nn_type* d_pad = p_pad + ((n % STREAMS) * _d_pad._c * _d_pad._h * _d_pad._w);
		nn_type* d_in = d_input + (n * _d_input._c * _d_input._h * _d_input._w);

		dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
		dim3 blocks = get_grid_size(threads, _d_output._w, _d_output._h, _d_output._c);

		__padding_dilation_2d<<<blocks, threads, 0, s[n % STREAMS]>>>(
			d_out,
			d_pad,
			_d_output._w,
			_d_output._h,
			_d_output._c,
			_d_pad._w,
			_d_pad._h,
			_conv._w_stride,
			_conv._h_stride,
			(_d_pad._w - _d_output._w) / 2,
			(_d_pad._h - _d_output._h) / 2
		);

		blocks = get_grid_size(threads, _d_input._h * _d_input._w, _d_input._c);

		__conv_2d<<<blocks, threads, 0, s[n % STREAMS]>>>(
			c_indice,
			d_pad,
			p_kernel,
			d_in,
			_d_pad._h,
			_d_pad._w,
			_kernel._c,
			_kernel._n,
			_kernel._h,
			_kernel._w,
			_d_input._h,
			_d_input._w,
			1, 1
		);
	}
}

/**********************************************

			  kernelConv2dSolution

**********************************************/

kernelConv2dSolution::kernelConv2dSolution(const dConv2dSolution& d_conv) :
	_d_conv(d_conv)
{
}

const size_t kernelConv2dSolution::get_workspace_size() {
	size_t size = 0;

	if ((_d_conv._d_output._h * _d_conv._d_output._w) > CONST_ELEM_SIZE)
		size = sizeof(uint) * (_d_conv._d_output._h * _d_conv._d_output._w);

	return size;
}

void kernelConv2dSolution::operator()(const nn_type* d_output, nn_type* gradient, const nn_type* input, void* workspace) {
	const tensor4d& _d_output = _d_conv._d_output;
	const tensor4d& _input = _d_conv._conv._input;
	const tensor4d& _kernel = _d_conv._conv._kernel;

	const uint* p_indice = (_d_output._h * _d_output._w) > CONST_ELEM_SIZE ? (uint*)workspace : get_indice_ptr();
	uint* indice = (uint*)malloc(sizeof(uint) * _d_output._h * _d_output._w);

	for (int h = 0; h < _d_output._h; ++h) {
		for (int w = 0; w < _d_output._w; ++w) {
			indice[h * _d_output._w + w] = (uint)(h * _d_output._w + w);
		}
	}

	if ((_d_output._h * _d_output._w) > CONST_ELEM_SIZE) {
		check_cuda(cudaMemcpy(workspace, indice, sizeof(uint) * _d_output._h * _d_output._w, cudaMemcpyHostToDevice));
	}
	else {
		set_indice(indice, sizeof(uint) * _d_output._h * _d_output._w, 0);
	}

	free(indice);

	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocks = get_grid_size(threads, _kernel._c * _kernel._h * _kernel._w, _kernel._n);

	for (int n = 0; n < _d_output._n; ++n) {
		const nn_type* d_out = d_output + (n * _d_output._c * _d_output._h * _d_output._w);
		const nn_type* d_in = input + (n * _input._c * _input._h * _input._w);

		__kernel_conv_2d<<<blocks, threads>>>(
			p_indice,
			d_out,
			d_in,
			gradient,
			_d_output._c,
			_d_output._h,
			_d_output._w,
			_input._c,
			_input._h,
			_input._w,
			_kernel._h,
			_kernel._w
			);
	}
}
