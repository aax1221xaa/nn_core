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

	void calculate_output_size(NN_Vec<Dim*> input_shape, Dim& output_shape);
	void build(Dim& input_shape);
	void run_forward(NN_Vec<NN_Tensor*> input, NN_Tensor& output);
	void run_backward(NN_Vec<NN_Tensor*> d_output, NN_Tensor& d_input);

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

NN_Model& Model(NN_Vec<NN_Link*>& inputs, NN_Vec<NN_Link*>& outputs);