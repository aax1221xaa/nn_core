#ifndef MAXPOOL_CUH
#define MAXPOOL_CUH

#include "../cpp_source/nn_base.h"


/**********************************************/
/*                                            */
/*                NN_Maxpool2D                */
/*                                            */
/**********************************************/

class NN_Maxpool2D : public NN_Layer {
	static uint special_i;

public:
	const std::string _pad;

	const NN_Shape _k_size;
	const NN_Shape _stride;

	GpuTensor<uint> _indice;

	NN_Maxpool2D(const NN_Shape& k_size, const NN_Shape& stride, const std::string& pad, const std::string& name = "");

	void get_output_shape(const NN_List<NN_Shape>& input_shape, NN_List<NN_Shape>& output_shape);
	void build(const NN_List<NN_Shape>& input_shape, NN_List<GpuTensor<nn_type>>& weights);
	void run(NN_Stream& st, const NN_List<GpuTensor<nn_type>>& input, NN_List<GpuTensor<nn_type>>& output);
	NN_Backward* create_backward(std::vector<bool>& mask);
};


/**********************************************/
/*                                            */
/*                NN_dMaxpool2D               */
/*                                            */
/**********************************************/

class NN_dMaxpool2D : public NN_Backward_t<NN_Maxpool2D> {
public:
	NN_dMaxpool2D(NN_Maxpool2D& layer);

	void run(
		NN_Stream& st,
		const NN_List<GpuTensor<nn_type>>& input,
		const NN_List<GpuTensor<nn_type>>& doutput,
		NN_List<GpuTensor<nn_type>>& dinput
	);
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
