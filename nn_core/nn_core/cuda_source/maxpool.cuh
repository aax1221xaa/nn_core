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

	void get_output_shape(const NN_List<NN_Shape>& input_shape, NN_List<NN_Shape>& output_shape);
	void build(const NN_List<NN_Shape>& input_shape, NN_Link* p_node);
	void run(NN_Stream& st, const NN_List<GpuTensor<nn_type>>& input, NN_List<GpuTensor<nn_type>>& output);
	NN_Backward* create_backward(NN_Optimizer* optimizer);
};


/**********************************************/
/*                                            */
/*                NN_dMaxpool2D               */
/*                                            */
/**********************************************/

class NN_dMaxpool2D : public NN_Backward {
public:
	NN_Maxpool2D* _maxpool;

	NN_dMaxpool2D(NN_Maxpool2D* maxpool, NN_Optimizer* optimizer);

	void get_dinput_shape(const NN_List<NN_Shape>& dout_shape, NN_List<NN_Shape>& din_shape);
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
