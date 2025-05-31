#include "special_act.cuh"
#include "cuda_misc.cuh"


#ifndef __CUDACC__
#define __CUDACC__
#endif

#include <device_functions.h>
#include <device_launch_parameters.h>


__global__ void __special_act(
	const nn_type* input,
	nn_type* output,
	cuint len,
	cuint ch_len,
	cuint* mark
) {
	cuint xidx = blockIdx.x * blockDim.x + threadIdx.x;
	cuint ch_idx = xidx % ch_len;
	cuint f_idx = xidx / ch_len;

	if (f_idx < len) {
		output[xidx] = mark[ch_idx] ? 1.f / (1.f + __expf(-input[xidx])) : input[xidx];
	}
}


SpecialAct::SpecialAct(int n_classes, const std::string& name) :
	NN_Layer(name, "SpecialAct"),
	_n_classes(n_classes)
{

}

void SpecialAct::get_output_shape(const NN_List<NN_Shape>& input_shape, NN_List<NN_Shape>& output_shape) {
	const NN_Shape& in = input_shape[0].val();
	NN_Shape out = { in[0], in[1], in[2], 3, 4 + 1 + _n_classes };

	output_shape.append(out);
}

void SpecialAct::run(NN_Stream& st, const NN_List<GpuTensor<nn_type>>& input, NN_List<GpuTensor<nn_type>>& output) {
	const GpuTensor<nn_type>& in_tensor = input.val();
	const NN_Shape in_shape = in_tensor.get_shape();
	const nn_type* in_data = in_tensor.get_ptr();

	GpuTensor<nn_type>& out_tensor = output.val();
	const NN_Shape out_shape = out_tensor.get_shape();
	nn_type* out_data = out_tensor.get_ptr();

	const int ch = out_shape[-1];
	uint* h_mark = new uint[ch];

	for (int i = 0; i < ch; ++i) h_mark[i] = 1;
	h_mark[2] = h_mark[3] = 0;

	cuint* g_mark = set_const_mem(h_mark, ch, 0);
	
	delete[] h_mark;

	uint len = 1;
	for (int i = 0; i < out_shape.ranks() - 1; ++i) len *= out_shape[i];

	dim3 threads(BLOCK_1024);
	dim3 blocks = get_grid_size(threads, out_shape.total_size());

	__special_act<<<blocks, threads>>>(
		in_data,
		out_data,
		len,
		out_shape[-1],
		g_mark
	);
}

void SpecialAct::set_output(const NN_List<NN_Shape>& output_shape, NN_List<GpuTensor<nn_type>>& input, NN_List<GpuTensor<nn_type>>& output) {
	GpuTensor<nn_type>& in_tensor = input.val();
	const NN_Shape& out_shape = output_shape.val();

	GpuTensor<nn_type> out_tensor(in_tensor, out_shape);

	output.append(out_tensor);
}