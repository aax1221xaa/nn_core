#pragma once
#include "nn_link.h"
#include "nn_optimizer.h"
#include "nn_loss.h"


class NN_Model : public NN_Layer {
public:
	vector<NN_Ptr<NN_Link>> input_nodes;
	vector<NN_Ptr<NN_Link>> output_nodes;

	NN_Model(vector<NN_Ptr<NN_Link>>& input, vector<NN_Ptr<NN_Link>>& output);

	const Dim calculate_output_size(const vector<Dim>& input_size);
	void build(const vector<Dim>& input_size);
	void run_forward(const vector<NN_Ptr<NN_Tensor>>& inputs, NN_Ptr<NN_Tensor>& output);
	void run_backward(const vector<NN_Ptr<NN_Tensor>>& d_outputs, NN_Ptr<NN_Tensor>& d_input);

	void compile(vector<NN_Ptr<NN_Loss>>& loss, vector<NN_Ptr<NN_Optimizer>>& optimizer);

	NN_Ptr<NN_Tensor> train_on_batch(const vector<NN_Ptr<NN_Tensor>>& samples, const vector<NN_Ptr<NN_Tensor>>& truth);
	NN_Ptr<NN_Tensor> fit(
		const vector<NN_Ptr<NN_Tensor>>& samples,
		const vector<NN_Ptr<NN_Tensor>>& truth,
		uint batch,
		uint iter
	);
	vector<NN_Ptr<NN_Tensor>> predict(const vector<NN_Ptr<NN_Tensor>>& x);
};