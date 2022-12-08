#pragma once
#include "nn_manager.h"


class NN_Model : public NN_Layer {
public:
	vector<NN_Link*> links;
	vector<NN_Link*> input_nodes;
	vector<NN_Link*> output_nodes;

	NN_Model(vector<NN_Link*> input, vector<NN_Link*> output);

	const Dim calculate_output_size(const vector<Dim>& input_size);
	void build(const vector<Dim>& input_size);
	void run_forward(const vector<NN_Tensor>& inputs, NN_Tensor& output);
	void run_backward(const vector<NN_Tensor>& d_outputs, NN_Tensor& d_input);

	void compile(vector<NN_Loss*> loss, vector<NN_Optimizer*> optimizer);

	const NN_Tensor& train_on_batch(const vector<NN_Tensor>& samples, const vector<NN_Tensor>& truth);
	const NN_Tensor& fit(
		const vector<NN_Tensor>& samples,
		const vector<NN_Tensor>& truth,
		uint batch,
		uint iter
	);
	const vector<NN_Tensor>& predict(const vector<NN_Tensor>& x);
};