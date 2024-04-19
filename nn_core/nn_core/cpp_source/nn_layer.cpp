#include "nn_layer.h"
#include "../cuda_source/matmul.cuh"
#include "../cuda_source/relu.cuh"
#include "../cuda_source/cuda_misc.cuh"
#include "../cuda_source/cuda_indice.cuh"
#include "../cuda_source/convolution.cuh"
#include "../cuda_source/maxpool.cuh"



/**********************************************/
/*                                            */
/*                   NN_Dense                 */
/*                                            */
/**********************************************/

NN_Dense::NN_Dense(const int amounts, const char* name) :
	NN_Layer(name),
	_amounts(amounts)
{
}

void NN_Dense::get_output_shape(const std::vector<NN_Shape>& input_shape, std::vector<NN_Shape>& output_shape) {
	const NN_Shape& shape = input_shape[0];

	output_shape.push_back({ shape[0], _amounts });
}

void NN_Dense::build(const std::vector<NN_Shape>& input_shape) {
	const NN_Shape& shape = input_shape[0];

	_weight.set({ shape[1], _amounts });
	_bias = gpu_zeros<nn_type>({ _amounts });
	set_random_uniform(_weight, 0.1f, -0.1f);

}

void NN_Dense::run_forward(NN_Stream& st, const std::vector<GpuTensor<nn_type>>& input, std::vector<GpuTensor<nn_type>>& output) {
	const GpuTensor<nn_type>& m_input = input[0];
	GpuTensor<nn_type>& m_output = output[0];

	matmul(
		m_input.get_shape()[0],
		m_input.get_shape()[1],
		m_output.get_shape()[1],
		m_input.get_data(),
		_weight.get_data(),
		m_output.get_data()
	);
	add_bias_1d(
		m_output.get_data(),
		_bias.get_data(),
		m_output.get_data(),
		m_output.get_shape()[0],
		m_output.get_shape()[1]
	);
}


/**********************************************/
/*                                            */
/*                   NN_ReLU                  */
/*                                            */
/**********************************************/

NN_ReLU::NN_ReLU(const char* name) :
	NN_Layer(name)
{
}

void NN_ReLU::get_output_shape(const std::vector<NN_Shape>& input_shape, std::vector<NN_Shape>& output_shape) {
	output_shape.push_back(input_shape[0]);
}

void NN_ReLU::build(const std::vector<NN_Shape>& input_shape) {

}

void NN_ReLU::run_forward(NN_Stream& st, const std::vector<GpuTensor<nn_type>>& input, std::vector<GpuTensor<nn_type>>& output) {
	relu(
		input[0].get_data(),
		output[0].get_data(),
		calculate_length(input[0].get_shape())
	);
}


/**********************************************/
/*                                            */
/*                   NN_Flat                  */
/*                                            */
/**********************************************/

NN_Flat::NN_Flat(const char* name) :
	NN_Layer(name)
{
}

void NN_Flat::get_output_shape(const std::vector<NN_Shape>& input_shape, std::vector<NN_Shape>& output_shape) {
	int c = 1;

	for (int i = 1; i < input_shape[0].get_size(); ++i) c *= input_shape[0][i];

	output_shape.push_back({ input_shape[0][0], c });
}

void NN_Flat::build(const std::vector<NN_Shape>& input_shape) {

}

void NN_Flat::run_forward(NN_Stream& st, const std::vector<GpuTensor<nn_type>>& input, std::vector<GpuTensor<nn_type>>& output) {
	gpu_to_gpu(input[0], output[0]);
}


/**********************************************/
/*                                            */
/*                 NN_Conv2D                  */
/*                                            */
/**********************************************/

NN_Conv2D::NN_Conv2D(int amounts, NN_Shape filter_size, NN_Shape stride, Pad pad, const char* name) :
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

	_filter.set({ _amounts, shape[1], _filter_size[0], _filter_size[1] });
	_bias = gpu_zeros<nn_type>({ _amounts });

	set_random_uniform(_filter, -0.1, 0.1);
}

void NN_Conv2D::run_forward(NN_Stream& st, const std::vector<GpuTensor<nn_type>>& input, std::vector<GpuTensor<nn_type>>& output) {
	const GpuTensor<nn_type>& m_input = input[0];
	GpuTensor<nn_type>& m_output = output[0];

	const NN_Shape& input_shape = m_input.get_shape();
	const NN_Shape& output_shape = m_output.get_shape();
	const NN_Shape& k_size = _filter.get_shape();

	if (_pad == Pad::SAME) {
		int pad_c = input_shape[1];
		int pad_h = 0;
		int pad_w = 0;

		if (_stride[0] == 1) {
			pad_h = input_shape[2] - 1 + k_size[2];
		}
		else {
			pad_h = (input_shape[2] / _stride[0]) + k_size[2];
		}

		if (_stride[1] == 1) {
			pad_w = input_shape[3] - 1 + k_size[3];
		}
		else {
			pad_w = (input_shape[3] / _stride[1]) + k_size[3];
		}

		uint* indice = new uint[k_size[1] * k_size[2] * k_size[3]];

		for (uint kc = 0; kc < k_size[1]; ++kc) {
			for (uint kh = 0; kh < k_size[2]; ++kh) {
				for (uint kw = 0; kw < k_size[3]; ++kw) {
					indice[kc * (k_size[2] * k_size[3]) + kh * k_size[3] + kw] =
						kc * (pad_h * pad_w) + kh * pad_w + kw;
				}
			}
		}

		set_indice(indice, sizeof(uint) * (k_size[1] * k_size[2] * k_size[3]), 0);
		delete[] indice;

		cuint* g_indice = get_indice_ptr();

		for (uint n = 0; n < input_shape[0]; ++n) {
			const nn_type* input_data = m_input.get_data() + (n * input_shape[1] * input_shape[2] * input_shape[3]);
			nn_type* output_data = m_output.get_data() + (n * output_shape[1] * output_shape[2] * output_shape[3]);

			if (pad_h != input_shape[2] || pad_w != input_shape[3]) {
				nn_type* pad_space = NULL;
				
				check_cuda(cudaMallocAsync(&pad_space, sizeof(nn_type) * pad_c * pad_h * pad_w, st[n % STREAMS]));
				check_cuda(cudaMemsetAsync(pad_space, 0, sizeof(nn_type) * pad_c * pad_h * pad_w, st[n % STREAMS]));

				padding_dilation(
					st[n % STREAMS],
					input_data,
					pad_space,
					input_shape[1],
					input_shape[2],
					input_shape[3],
					pad_h,
					pad_w,
					_stride[1] == 1 ? (input_shape[3] - pad_w) / 2 : 0,
					_stride[0] == 1 ? (input_shape[2] - pad_h) / 2 : 0,
					0, 0
				);
				conv2d(
					st[n % STREAMS],
					g_indice,
					pad_space,
					_filter.get_data(),
					output_data,
					pad_c,
					pad_h,
					pad_w,
					k_size[2],
					k_size[3],
					output_shape[1],
					output_shape[2],
					output_shape[3],
					_stride[0],
					_stride[1]
				);

				check_cuda(cudaFreeAsync(pad_space, st[n % STREAMS]));
			}
			else {
				conv2d(
					st[n % STREAMS],
					g_indice,
					input_data,
					_filter.get_data(),
					output_data,
					input_shape[1],
					input_shape[2],
					input_shape[3],
					k_size[2],
					k_size[3],
					output_shape[1],
					output_shape[2],
					output_shape[3],
					_stride[0],
					_stride[1]
				);
			}
		}
	}
}


/**********************************************/
/*                                            */
/*                NN_Maxpool2D                */
/*                                            */
/**********************************************/

NN_Maxpool2D::NN_Maxpool2D(const NN_Shape k_size, const NN_Shape stride, const Pad pad, const char* name) :
	_pad(pad),
	_k_size(k_size),
	_stride(stride),
	NN_Layer(name)
{
}

void NN_Maxpool2D::get_output_shape(const std::vector<NN_Shape>& input_shape, std::vector<NN_Shape>& output_shape) {
	const NN_Shape& shape = input_shape[0];

	int n = shape[0];
	int c = shape[1];
	int h = 0;
	int w = 0;

	if (_pad == Pad::SAME) {
		h = (int)ceil((float)(shape[2] - _k_size[0]) / _stride[0] + 1);
		w = (int)ceil((float)(shape[3] - _k_size[1]) / _stride[1] + 1);
	}
	else {
		h = (int)floorf((float)(shape[2] - _k_size[0]) / _stride[0] + 1);
		w = (int)floorf((float)(shape[3] - _k_size[1]) / _stride[1] + 1);
	}

	output_shape.push_back({ n, c, h, w });
}

void NN_Maxpool2D::build(const std::vector<NN_Shape>& input_shape) {
	const NN_Shape& shape = input_shape[0];

	int n = shape[0];
	int c = shape[1];
	int h = 0;
	int w = 0;

	if (_pad == Pad::SAME) {
		h = (int)ceil((float)(shape[2] - _k_size[0]) / _stride[0] + 1);
		w = (int)ceil((float)(shape[3] - _k_size[1]) / _stride[1] + 1);
	}
	else {
		h = (int)floorf((float)(shape[2] - _k_size[0]) / _stride[0] + 1);
		w = (int)floorf((float)(shape[3] - _k_size[1]) / _stride[1] + 1);
	}

	_indice = gpu_zeros<uint>({ n, c, h, w });
}

void NN_Maxpool2D::run_forward(NN_Stream& st, const std::vector<GpuTensor<nn_type>>& input, std::vector<GpuTensor<nn_type>>& output) {
	const nn_type* m_input = input[0].get_data();
	nn_type* m_output = output[0].get_data();
	uint* m_indice = _indice.get_data();

	const NN_Shape& in_shape = input[0].get_shape();
	const NN_Shape& out_shape = output[0].get_shape();

	int tile_h = (BLOCK_32 - 1) * _stride[0] + _k_size[0];
	int tile_w = (BLOCK_32 - 1) * _stride[1] + _k_size[1];

	for (int n = 0; n < in_shape[0]; ++n) {
		for (int c = 0; c < in_shape[1]; ++c) {
			const nn_type* in_data = m_input + (n * in_shape[1] * in_shape[2] * in_shape[3]) + (c * in_shape[2] * in_shape[3]);
			nn_type* out_data = m_output + (n * out_shape[1] * out_shape[2] * out_shape[3]) + (c * out_shape[2] * out_shape[3]);
			uint* indice = m_indice + (n * out_shape[1] * out_shape[2] * out_shape[3]) + (c * out_shape[2] * out_shape[3]);

			maxpool2d(
				st[(n * in_shape[1] + c) % STREAMS],
				in_data,
				out_data,
				indice,
				in_shape[2],
				in_shape[3],
				out_shape[2],
				out_shape[3],
				_k_size[0],
				_k_size[1],
				_stride[0],
				_stride[1],
				tile_h,
				tile_w
			);
		}
	}
}