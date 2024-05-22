#ifndef _CONVOLUTION_CUH_
#define _CONVULUTION_CUH_

#include "../cpp_source/nn_base.h"


/**********************************************/
/*                                            */
/*                 NN_Conv2D                  */
/*                                            */
/**********************************************/

class NN_Conv2D : public NN_Layer {
public:
	const int _amounts;
	const NN_Shape _filter_size;
	const NN_Shape _stride;
	const Pad _pad;

	GpuTensor<nn_type> _filter;
	GpuTensor<nn_type> _bias;

	static cuint* get_indice(const NCHW& in, const NCHW& k);

	NN_Conv2D(int amounts, const NN_Shape& filter_size, const NN_Shape& stride, Pad pad, const char* name);

	void get_output_shape(const std::vector<NN_Shape>& input_shape, std::vector<NN_Shape>& output_shape);
	void build(const std::vector<NN_Shape>& input_shape);
	void run_forward(NN_Stream& st, const std::vector<GpuTensor<nn_type>>& input, std::vector<GpuTensor<nn_type>>& output);
};

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
);
*/
#endif // !_CONVOLUTION_CUH_