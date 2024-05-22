#ifndef MAXPOOL_CUH
#define MAXPOOL_CUH

#include "../cpp_source/nn_base.h"


/**********************************************/
/*                                            */
/*                NN_Maxpool2D                */
/*                                            */
/**********************************************/

class NN_Maxpool2D : public NN_Layer {
public:
	const Pad _pad;

	const NN_Shape _k_size;
	const NN_Shape _stride;

	GpuTensor<uint> _indice;

	NN_Maxpool2D(const NN_Shape& k_size, const NN_Shape& stride, const Pad pad, const char* name);

	void get_output_shape(const std::vector<NN_Shape>& input_shape, std::vector<NN_Shape>& output_shape);
	void build(const std::vector<NN_Shape>& input_shape);
	void run_forward(NN_Stream& st, const std::vector<GpuTensor<nn_type>>& input, std::vector<GpuTensor<nn_type>>& output);
};

/**********************************************

				  D_Maxpool2d

**********************************************/
/*
void d_maxpool2d(
	cudaStream_t* s,
	const nn_type* d_output,
	nn_type* d_input,
	const nn_shape& out_shape,
	const nn_shape& in_shape,
	cuint* max_indice,
	cuint w_kernel,
	cuint h_stride,
	cuint w_stride
);
*/

#endif // !MAXPOOL_CUH
