#pragma once
#include "nn_manager.h"
#include "nn_optimizer.h"
#include "nn_loss.h"



class NN_Model : public NN_Layer, public NN_Link {
protected:
	static NN_Link* get_child_link(NN_Link* parent_link);
	static int get_unselect_prev(NN_Link* p_current);

public:
	vector<NN_Link*> input_nodes;
	vector<NN_Link*> output_nodes;

	NN_Model(NN_Vec<NN_Link*>& inputs, NN_Vec<NN_Link*>& outputs);
	~NN_Model();

	void calculate_output_size(vector<NN_Shape_t>& input_shape, NN_Shape_t& output_shape);
	void build(vector<NN_Shape_t>& input_shape);
	void run_forward(vector<NN_Tensor_t>& input, NN_Tensor_t& output);
	void run_backward(vector<NN_Tensor_t>& d_output, NN_Tensor_t& d_input);

	void compile(const vector<NN_Loss*>& loss, const vector<NN_Optimizer*>& optimizer);
	NN_Tensor_t train_on_batch(const vector<NN_Tensor_t>& samples, const vector<NN_Tensor_t>& truth);
	NN_Tensor_t fit(
		const vector<NN_Tensor_t>& samples,
		const vector<NN_Tensor_t>& truth,
		uint batch,
		uint iter
	);
	vector<NN_Tensor_t> predict(const vector<NN_Tensor_t>& x);

	NN_Vec<NN_Coupler<NN_Link>> operator()(const NN_Vec<NN_Coupler<NN_Link>> m_prev_link);
};

NN_Model& Model(NN_Vec<NN_Link*>& inputs, NN_Vec<NN_Link*>& outputs);