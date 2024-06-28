#ifndef MATMUL_CUH
#define MATMUL_CUH

#include "../cpp_source/nn_base.h"



/**********************************************/
/*                                            */
/*                   NN_Dense                 */
/*                                            */
/**********************************************/

void test_matmul(
	const GpuTensor<nn_type>& input,
	const GpuTensor<nn_type>& weight,
	const GpuTensor<nn_type>& bias,
	GpuTensor<nn_type>& output
	);

class NN_Dense : public NN_Layer {
public:
	GpuTensor<nn_type> _weight;
	GpuTensor<nn_type> _bias;
	const int _amounts;

	NN_Dense(const int amounts, const char* name);

	void get_output_shape(const NN_List<NN_Shape>& input_shape, NN_List<NN_Shape>& output_shape);
	void build(const NN_List<NN_Shape>& input_shape);
	void run(NN_Stream& st, const NN_List<GpuTensor<nn_type>>& input, NN_List<GpuTensor<nn_type>>& output);
	NN_Backward* create_backward(NN_Optimizer* optimizer);

	NN_List<GpuTensor<nn_type>> get_weight();
};


/**********************************************/
/*                                            */
/*                   NN_dDense                */
/*                                            */
/**********************************************/

class NN_dDense : public NN_Backward {
public:
	NN_Dense* _dense;

	NN_dDense(NN_Dense* dense, NN_Optimizer* optimizer);

	void get_dinput_shape(const NN_List<NN_Shape>& dout_shape, NN_List<NN_Shape>& din_shape);
	void run(
		NN_Stream& st,
		const NN_List<GpuTensor<nn_type>>& input,
		const NN_List<GpuTensor<nn_type>>& doutput,
		NN_List<GpuTensor<nn_type>>& dinput
	);
};


#endif