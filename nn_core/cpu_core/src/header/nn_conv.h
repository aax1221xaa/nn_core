#pragma once
#include "nn_common.h"
#include "nn_base.h"



class NN_Conv2D : public NN_Layer {
	NN_Tensor<nn_type> _kernel;
	NN_Tensor<nn_type> _bias;

	const int _amounts;
	const NN_Shape _kernel_size;
	const NN_Shape _strides;
	const std::string _pad;

public:
	NN_Conv2D(int amounts, const NN_Shape& kernel_size, const NN_Shape& strides, const std::string& pad, const std::string& name);
	~NN_Conv2D();

	void get_output_shape(const NN_List<NN_Shape>& input_shape, NN_List<NN_Shape>& output_shape);
	void build(const NN_List<NN_Shape>& input_shape, NN_List<NN_Tensor<nn_type>>& weights);
	void run(const NN_List<NN_Tensor<nn_type>>& input, NN_List<NN_Tensor<nn_type>>& output);
	NN_Backward* create_backward(std::vector<bool>& mask);
	NN_List<NN_Tensor<nn_type>> get_weight();
};

class NN_dConv2D : public NN_Backward_t<NN_Conv2D> {
public:
	NN_dConv2D(NN_Conv2D& layer);

	void run(
		const NN_List<NN_Tensor<nn_type>>& input,
		const NN_List<NN_Tensor<nn_type>>& doutput,
		NN_List<NN_Tensor<nn_type>>& dinput
	);
	NN_Optimizer* create_optimizer(const NN_Optimizer& optimizer);
};