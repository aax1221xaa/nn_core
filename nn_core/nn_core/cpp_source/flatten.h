#pragma once
#include "nn_base.h"


/******************************************/
/*                                        */
/*                NN_Flatten              */
/*                                        */
/******************************************/

class NN_Flatten : public NN_Layer {
public:
	NN_Flatten(const char* name);

	void get_output_shape(const NN_List<NN_Shape>& input_shape, NN_List<NN_Shape>& output_shape);
	void build(const NN_List<NN_Shape>& input_shape, std::vector<GpuTensor<nn_type>>& weights);
	void run(NN_Stream& st, const NN_List<GpuTensor<nn_type>>& input, NN_List<GpuTensor<nn_type>>& output);
	NN_Backward* create_backward(NN_Optimizer& optimizer, std::vector<bool>& mask);
};


/******************************************/
/*                                        */
/*               NN_dFlatten              */
/*                                        */
/******************************************/

class NN_dFlatten : public NN_Backward {
public:
	NN_Flatten& _flatten;

	NN_dFlatten(NN_Flatten& flatten, NN_Optimizer& optimizer);

	void get_dinput_shape(const NN_List<NN_Shape>& dout_shape, NN_List<NN_Shape>& din_shape);
	void run(
		NN_Stream& st,
		const NN_List<GpuTensor<nn_type>>& input,
		const NN_List<GpuTensor<nn_type>>& doutput,
		NN_List<GpuTensor<nn_type>>& dinput
	);
};