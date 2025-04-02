#pragma once
#include "nn_base.h"


/******************************************/
/*                                        */
/*                NN_Flatten              */
/*                                        */
/******************************************/

class NN_Flatten : public NN_Layer {
public:
	NN_Flatten(const std::string& name);

	void get_output_shape(const NN_List<NN_Shape>& input_shape, NN_List<NN_Shape>& output_shape);
	void run(const NN_List<NN_Tensor<nn_type>>& input, NN_List<NN_Tensor<nn_type>>& output);
	NN_Backward* create_backward(std::vector<bool>& mask);
};


/******************************************/
/*                                        */
/*               NN_dFlatten              */
/*                                        */
/******************************************/

class NN_dFlatten : public NN_Backward_t<NN_Flatten> {
public:
	NN_dFlatten(NN_Flatten& flatten);

	void run(
		const NN_List<NN_Tensor<nn_type>>& input,
		const NN_List<NN_Tensor<nn_type>>& doutput,
		NN_List<NN_Tensor<nn_type>>& dinput
	);
};