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
	const float* a,
	float* b,
	uint* mark,
	cuint a_h,
	cuint a_w,
	cuint b_h,
	cuint b_w,
	cuint ch,
	cuint k_h,
	cuint k_w,
	cuint st_h,
	cuint st_w,
	cuint tile_h,
	cuint tile_w
) {
	extern __shared__ float sm[];

	cuint out_x = blockIdx.x * blockDim.x + threadIdx.x;
	cuint out_y = blockIdx.y * blockDim.y + threadIdx.y;
	cuint x0 = blockIdx.x * blockDim.x * st_w;
	cuint y0 = blockIdx.y * blockDim.y * st_h;
	cuint z0 = blockIdx.z;

	const float* pa = a + ((z0 * a_w * a_h) + (y0 * a_w) + x0);
	float* pb = b + (b_w * b_h * z0);
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
		float val = -FLT_MAX;
		uint index = 0;

		for (uint h = 0; h < k_h; ++h) {
			cuint ty = threadIdx.y * st_h + h;
			for (uint w = 0; w < k_w; ++w) {
				cuint tx = threadIdx.x * st_w + w;
				float sm_val = sm[ty * tile_w + tx];

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


/**********************************************

				  _maxpool2d

**********************************************/

int calc_shared_mem_size(
	int kernel_size,
	int strides
) {
	return (BLOCK_SIZE - 1) * strides + kernel_size;
}

maxpool2dParam::maxpool2dParam() :
	_kh(0),
	_kw(0),
	_stride_h(0),
	_stride_w(0)
{
}

maxpool2dParam::maxpool2dParam(int kh, int kw, int stride_h, int stride_w) :
	_kh(kh),
	_kw(kw),
	_stride_h(stride_h),
	_stride_w(stride_w)
{
}

void maxpool2dParam::set(int kh, int kw, int stride_h, int stride_w) {
	_kh = kh;
	_kw = kw;
	_stride_h = stride_h;
	_stride_w = stride_w;
}

const bool maxpool2dParam::is_valid() const {
	return (_kh > 0 && _kw > 0 && _stride_h > 0 && _stride_w > 0);
}


_maxpool2d::_maxpool2d(const tensor4d& input, const maxpool2dParam& param) :
	_input(input),
	_param(param)
{
}

const tensor4d _maxpool2d::calculate_size() {
	if (!_input.is_valid() || !_param.is_valid()) {
		ErrorExcept(
			"[_maxpool2d::calculate_size()] invalid arguments. input: %s, kernel_size: [%d, %d], strides: [%d, %d]",
			tensor4d::shape_to_str(_input),
			_param._kw, _param._kh,
			_param._stride_w, _param._stride_h
		);
	}

	int out_n = _input._n;
	int out_c = _input._c;
	int out_h = (_input._h - _param._kh) / _param._stride_h + 1;
	int out_w = (_input._w - _param._kw) / _param._stride_w + 1;

	_output.set(out_n, out_c, out_h, out_w);

	if (!_output.is_valid()) {
		ErrorExcept(
			"[_maxpool2d::calculate_size()] invalid output size. output: %s",
			tensor4d::shape_to_str(_output)
		);
	}

	return _output;
}

void _maxpool2d::operator()(cudaStream_t* s, const nn_type* input, nn_type* output) {
	int share_w = calc_shared_mem_size(_param._kw, _param._stride_w);
	int share_h = calc_shared_mem_size(_param._kh, _param._stride_h);

	size_t smem_size = sizeof(float) * share_w * share_h;

	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocks = get_grid_size(threads, _output._w, _output._h);

	for (uint i = 0; i < _input._n; ++i) {
		const float* d_in = input + (i * _input._h * _input._w * _input._c);
		float* d_out = output + (i * _output._h * _output._w * _output._c);

		__maxpool_2d<<<blocks, threads, smem_size, s[i % STREAMS]>>>(
			d_in,
			d_out,
			_input._h,
			_input._w,
			_output._h,
			_output._w,
			_input._c,
			_param._kh,
			_param._kw,
			_param._stride_h,
			_param._stride_w,
			share_h,
			share_w
		);
	}
}

/**********************************************

				  _dMaxpool2d

**********************************************/

_dMaxpool2d::_dMaxpool2d(const tensor4d& d_output, const _maxpool2d& maxpool) :
	_d_output(d_output),
	_maxpool(maxpool)
{
}

const tensor4d _dMaxpool2d::calculate_size() {
	if (!_d_output.is_valid()) {
		ErrorExcept(
			"[_dMaxpool2d::calculate_size()] invalid d_output. %s",
			tensor4d::shape_to_str(_d_output)
		);
	}

	return _maxpool._input;
}

void _dMaxpool2d::operator()(const nn_type* d_output, nn_type* d_input) {

}