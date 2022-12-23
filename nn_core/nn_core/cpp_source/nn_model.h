#pragma once
#include "nn_manager.h"
#include "nn_optimizer.h"
#include "nn_loss.h"


/*
class NN_Link {
public:
	bool is_selected;
	bool trainable;

	vector<NN_Link*> prev_link;
	vector<NN_Link*> next_link;

	NN_Link* parent;
	vector<NN_Link*> child;

	NN_Tensor output;
	NN_Tensor d_input;
	Dim output_shape;

	NN_Layer* op_layer;

	NN_Link(NN_Layer* p_layer);
	NN_Link(NN_Link* parent_link);
	~NN_Link();

	NN_Vec<NN_Link*> operator()(const NN_Vec<NN_Link*> m_prev_link);
};
*/

class NN_Model : public NN_Layer, public NN_Link {
protected:
	static NN_Link* get_child_link(NN_Link* parent_link);

public:
	vector<NN_Link*> input_nodes;
	vector<NN_Link*> output_nodes;

	NN_Model(NN_Vec<NN_Link*>& inputs, NN_Vec<NN_Link*>& outputs);

	void calculate_output_size(NN_Vec<Dim*> input_shape, NN_Vec<Dim*> output_shape);
	void build(NN_Vec<Dim*> input_shape);
	void run_forward(NN_Vec<NN_Tensor*> input, NN_Vec<NN_Tensor*> output);
	void run_backward(NN_Vec<NN_Tensor*> d_output, NN_Vec<NN_Tensor*> d_input);

	void compile(const vector<NN_Loss*>& loss, const vector<NN_Optimizer*>& optimizer);
	NN_Tensor train_on_batch(const vector<NN_Tensor>& samples, const vector<NN_Tensor>& truth);
	NN_Tensor fit(
		const vector<NN_Tensor>& samples,
		const vector<NN_Tensor>& truth,
		uint batch,
		uint iter
	);
	vector<NN_Tensor> predict(const vector<NN_Tensor>& x);

	NN_Vec<NN_Coupler<NN_Link>> operator()(const NN_Vec<NN_Coupler<NN_Link>> m_prev_link);
};