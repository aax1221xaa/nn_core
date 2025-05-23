#include "nn_dw_conv.cuh"
#include "cuda_misc.cuh"


#ifndef __CUDACC__
#define __CUDACC__
#endif

#include <device_functions.h>
#include <device_launch_parameters.h>


__global__ void __dw_conv2d(
	const nn_type* input,
	const nn_type* kernel,
	nn_type* output,
	cuint* indice,
	cuint in_h,
	cuint in_w,
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
	cuint tidx = threadIdx.y * BLOCK_32 + threadIdx.x;

	cuint out_x = cx % out_w;
	cuint out_y = cx / out_w;
	cuint in_x = out_x * st_w;
	cuint in_y = out_y * st_h;

	cuint k = k_h * k_w;
	cuint n = out_w * out_h;

	__shared__ nn_type share_in[BLOCK_32 * BLOCK_32];
	__shared__ nn_type share_k[BLOCK_32 * BLOCK_32];

	const nn_type* p_input = input + ((in_y * in_w * out_c) + (in_x * out_c) + cy);
	const nn_type* p_kernel = kernel + cy;
	nn_type* p_output = output + ((out_y * out_w * out_c) + (out_x * out_c) + cy);

	nn_type sum = 0.f;

	for (uint i = 0; i < k; i += BLOCK_32) {
		cuint th_x = i + threadIdx.x;
		cuint th_y = i + threadIdx.y;

		__syncthreads();
		share_in[tidx] = th_y < k && cx < n ? p_input[indice[th_y]] : 0.f;
		share_k[tidx] = th_x < k && cy < out_c ? p_kernel[th_x] : 0.f;
		__syncthreads();

		for (uint e = 0; e < BLOCK_32; ++e) {
			sum += share_k[threadIdx.y * BLOCK_32 + e] * share_in[e * BLOCK_32 + threadIdx.x];
		}
	}

	if (cx < n && cy < out_c) *p_output = sum;
}


/**********************************************/
/*                                            */
/*                 NN_DwConv2D                */
/*                                            */
/**********************************************/

NN_DwConv2D::NN_DwConv2D(const NN_Shape& k_size, const NN_Shape& stride, const std::string& pad, bool use_bias, const std::string& layer_name) :
	NN_Layer(layer_name),
	_k_size(k_size),
	_stride(stride),
	_pad(pad),
	_use_bias(use_bias)
{

}

void NN_DwConv2D::get_output_shape(const NN_List<NN_Shape>& input_shape, NN_List<NN_Shape>& output_shape) {
	const NN_Shape& shape = input_shape[0].val();

	if (_pad == "same") {
		int n = shape[0];
		int h = (int)ceil((float)shape[1] / _stride[1]);
		int w = (int)ceil((float)shape[2] / _stride[0]);
		int c = shape[3];

		output_shape.append(NN_Shape({ n, h, w, c }));
	}
	else {
		int n = shape[0];
		int h = (int)floorf((float)(shape[1] - _k_size[1]) / _stride[1] + 1);
		int w = (int)floorf((float)(shape[2] - _k_size[0]) / _stride[0] + 1);
		int c = shape[3];

		output_shape.append(NN_Shape({ n, h, w, c }));
	}
}

void NN_DwConv2D::build(const NN_List<NN_Shape>& input_shape, NN_List<GpuTensor<nn_type>>& weights) {
	const NN_Shape& shape = input_shape[0].val();

	_kernel.resize(NN_Shape({ _k_size[1], _k_size[0], shape[3], 1 }));
	set_random_uniform(_kernel, -0.1f, 0.1f);

	weights.append(_kernel);

	if (_use_bias) {
		_bias = GpuTensor<nn_type>::zeros({ shape[3] });
		weights.append(_bias);
	}
}

void NN_DwConv2D::run(NN_Stream& st, const NN_List<GpuTensor<nn_type>>& input, NN_List<GpuTensor<nn_type>>& output) {
	
}

NN_List<GpuTensor<nn_type>> NN_DwConv2D::get_weight() {
	if (_use_bias) return { _kernel, _bias };
	else return { _kernel, };
}