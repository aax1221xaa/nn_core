#include "nn_upsample.cuh"

#ifndef __CUDACC__
#define __CUDACC__
#endif

#include <device_functions.h>
#include <device_launch_parameters.h>


__global__ void __upsample2d(
	const nn_type* input,
	nn_type* output,
	cuint in_h,
	cuint in_w,
	cuint c,
	cuint kh,
	cuint kw
) {
	cuint cx = blockIdx.x * blockDim.x + threadIdx.x;
	cuint cy = blockIdx.y * blockDim.y + threadIdx.y;
	cuint cz = blockIdx.z;

	cuint ox = cx * kw;
	cuint oy = cy * kh;

	const nn_type* p_input = input + ((cy * in_w * c) + (cx * c) + cz);
	nn_type* p_output = output + ((oy * in_w * kw * c) + (ox * c) + cz);

	if (cx < in_w && cy < in_h) {
		nn_type in_val = *p_input;

		for (uint i = 0; i < kh; ++i) {
			for (uint j = 0; j < kw; ++j) {
				p_output[i * (in_w * kw * c) + j * c] = in_val;
			}
		}
	}
}


/**********************************************/
/*                                            */
/*                NN_UpSample2D               */
/*                                            */
/**********************************************/

NN_UpSample2D::NN_UpSample2D(const NN_Shape& k_size, const std::string& name) :
	NN_Layer(name),
	_k_size(k_size)
{

}

void NN_UpSample2D::get_output_shape(const NN_List<NN_Shape>& input_shape, NN_List<NN_Shape>& output_shape) {
	const NN_Tensor4dShape shape = input_shape[0].val().get_4d_shape();
	NN_Shape out_shape(4);

	out_shape[0] = shape._n;
	out_shape[1] = shape._h * _k_size[0];
	out_shape[2] = shape._w * _k_size[1];
	out_shape[3] = shape._c;

	output_shape.append(out_shape);
}

void NN_UpSample2D::run(NN_Stream& st, const NN_List<GpuTensor<nn_type>>& input, NN_List<GpuTensor<nn_type>>& output) {
	const GpuTensor<nn_type>& p_input = input[0].val();
	GpuTensor<nn_type>& p_output = output[0].val();

	const NN_Tensor4dShape in_shape = p_input.get_shape().get_4d_shape();
	const NN_Tensor4dShape out_shape = p_output.get_shape().get_4d_shape();

	dim3 threads(BLOCK_32, BLOCK_32);
	dim3 blocks = get_grid_size(threads, in_shape._w, in_shape._h, in_shape._c);

	for (uint n = 0; n < in_shape._n; ++n) {
		const nn_type* in_data = p_input.get_ptr() + (in_shape._h * in_shape._w * in_shape._c * n);
		nn_type* out_data = p_output.get_ptr() + (out_shape._h * out_shape._w * out_shape._c * n);

		__upsample2d<<<blocks, threads, 0, st[n % STREAMS]>>>(
			in_data,
			out_data,
			in_shape._h,
			in_shape._w,
			in_shape._c,
			_k_size[0],
			_k_size[1]
		);
	}
}