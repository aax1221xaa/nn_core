#ifndef SIGMOID_CUH
#define SIGMOID_CUH

#include "../cpp_source/nn_base.h"


/**********************************************/
/*                                            */
/*                  NN_Sigmoid                */
/*                                            */
/**********************************************/

class NN_Sigmoid : public NN_Layer {
public:
	NN_Sigmoid(const std::string& name = "");

	void get_output_shape(const NN_List<NN_Shape>& input_shape, NN_List<NN_Shape>& output_shape);
	void build(const NN_List<NN_Shape>& input_shape, NN_List<GpuTensor<nn_type>>& weights);
	void run(NN_Stream& st, const NN_List<GpuTensor<nn_type>>& input, NN_List<GpuTensor<nn_type>>& output);
	NN_Backward* create_backward(std::vector<bool>& mask);
};


/**********************************************/
/*                                            */
/*                  NN_dSigmoid               */
/*                                            */
/**********************************************/

class NN_dSigmoid : public NN_Backward_t<NN_Sigmoid> {
public:
	NN_dSigmoid(NN_Sigmoid& layer);

	void run(
		NN_Stream& st,
		const NN_List<GpuTensor<nn_type>>& input,
		const NN_List<GpuTensor<nn_type>>& doutput,
		NN_List<GpuTensor<nn_type>>& dinput
	);
};

#endif