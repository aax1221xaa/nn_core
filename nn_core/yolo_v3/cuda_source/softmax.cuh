#ifndef SOFTMAX_CUH
#define SOFTMAX_CUH

#include "../cpp_source/nn_base.h"


/**********************************************/
/*                                            */
/*                  NN_Softmax                */
/*                                            */
/**********************************************/

class NN_Softmax : public NN_Layer {
public:
	NN_Softmax(const std::string& name = "");

	void get_output_shape(const NN_List<NN_Shape>& input_shape, NN_List<NN_Shape>& output_shape);
	void build(const NN_List<NN_Shape>& input_shape, NN_List<GpuTensor<nn_type>>& weights);
	void run(NN_Stream& st, const NN_List<GpuTensor<nn_type>>& input, NN_List<GpuTensor<nn_type>>& output);
	NN_Backward* create_backward(std::vector<bool>& mask);
};


/**********************************************/
/*                                            */
/*                 NN_dSoftmax                */
/*                                            */
/**********************************************/

class NN_dSoftmax : public NN_Backward_t<NN_Softmax> {
public:
	NN_dSoftmax(NN_Softmax& layer);

	void run(
		NN_Stream& st,
		const NN_List<GpuTensor<nn_type>>& input,
		const NN_List<GpuTensor<nn_type>>& doutput,
		NN_List<GpuTensor<nn_type>>& dinput
	);
};


#endif