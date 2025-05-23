#ifndef MATMUL_CUH
#define MATMUL_CUH

#include "../cpp_source/nn_base.h"



/**********************************************/
/*                                            */
/*                   NN_Dense                 */
/*                                            */
/**********************************************/

class NN_Dense : public NN_Layer {
public:
	GpuTensor<nn_type> _weight;
	GpuTensor<nn_type> _bias;
	const int _amounts;

	NN_Dense(const int amounts, const std::string& name = "");

	void get_output_shape(const NN_List<NN_Shape>& input_shape, NN_List<NN_Shape>& output_shape);
	void build(const NN_List<NN_Shape>& input_shape, NN_List<GpuTensor<nn_type>>& weights);
	void run(NN_Stream& st, const NN_List<GpuTensor<nn_type>>& input, NN_List<GpuTensor<nn_type>>& output);
	NN_Backward* create_backward(std::vector<bool>& mask);
	NN_List<GpuTensor<nn_type>> get_weight();
};


/**********************************************/
/*                                            */
/*                   NN_dDense                */
/*                                            */
/**********************************************/

class NN_dDense : public NN_Backward_t<NN_Dense> {
public:
	NN_dDense(NN_Dense& dense);

	void run(
		NN_Stream& st,
		const NN_List<GpuTensor<nn_type>>& input,
		const NN_List<GpuTensor<nn_type>>& doutput,
		NN_List<GpuTensor<nn_type>>& dinput
	);
	NN_Optimizer* create_optimizer(const NN_Optimizer& optimizer);
};


#endif