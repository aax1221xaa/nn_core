#pragma once
#include "nn_manager.h"
#include "nn_optimizer.h"
#include "nn_loss.h"
#include "nn_ptr.h"


/*
class NN_Link {
public:
	bool is_selected;
	bool trainable;

	vector<NN_Ptr<NN_Link>> prev_link;
	vector<NN_Ptr<NN_Link>> next_link;

	NN_Ptr<NN_Link> parent;
	vector<NN_Ptr<NN_Link>> child;

	NN_Ptr<NN_Tensor> output;
	NN_Ptr<NN_Tensor> d_input;
	Dim output_shape;

	NN_Ptr<NN_Layer> layer;

	NN_Link(NN_Ptr<NN_Layer> p_layer);
	NN_Link(NN_Ptr<NN_Link> parent_link);

	NN_Coupler<NN_Link> operator()(NN_Coupler<NN_Link>& m_prev_link);

	virtual NN_Ptr<NN_Link> get_link(NN_Ptr<NN_Link>& p_next);
};
*/

class NN_Model : public NN_Layer, public NN_Link {
protected:
	int get_link_index(vector<NN_Ptr<NN_Link>>& link_list, NN_Ptr<NN_Link>& target);

public:
	vector<NN_Ptr<NN_Link>> input_nodes;
	vector<NN_Ptr<NN_Link>> output_nodes;
	vector<NN_Ptr<NN_Link>> m_layers;

	NN_Model(NN_Coupler<NN_Link>& inputs, NN_Coupler<NN_Link>& outputs);

	void calculate_output_size(vector<Dim*>& input_shape, Dim& output_shape);
	void build(vector<Dim*>& input_shape);
	void run_forward(vector<NN_Ptr<NN_Tensor>>& input, NN_Ptr<NN_Tensor>& output);
	void run_backward(vector<NN_Ptr<NN_Tensor>>& d_output, NN_Ptr<NN_Tensor>& d_input);

	void compile(vector<NN_Loss*>& loss, vector<NN_Optimizer*>& optimizer);
	NN_Ptr<NN_Tensor> train_on_batch(const vector<NN_Ptr<NN_Tensor>>& samples, const vector<NN_Ptr<NN_Tensor>>& truth);
	NN_Ptr<NN_Tensor> fit(
		const vector<NN_Ptr<NN_Tensor>>& samples,
		const vector<NN_Ptr<NN_Tensor>>& truth,
		uint batch,
		uint iter
	);
	vector<NN_Ptr<NN_Tensor>> predict(const vector<NN_Ptr<NN_Tensor>>& x);
	
	NN_Ptr<NN_Tensor> get_prev_output(NN_Ptr<NN_Link>& p_current);
	NN_Ptr<NN_Tensor> get_next_dinput(NN_Ptr<NN_Link>& p_current);
	Dim get_next_output_shape(NN_Ptr<NN_Link>& p_current);
};


NN_Link& Model(vector<NN_Ptr<NN_Link>>& input, vector<NN_Ptr<NN_Link>>& output);