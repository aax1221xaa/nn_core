#pragma once
#include "nn_manager.h"
#include "nn_optimizer.h"
#include "nn_loss.h"



class NN_Model : public NN_Layer, public NN_Link {
protected:
	static NN_Link* get_child_link(NN_Link* parent_link);
	static int get_unselect_prev(NN_Link* p_current);

	static vector<NN_Link*> find_root(vector<NN_Link*>& in_nodes, vector<NN_Link*>& out_nodes);
	static vector<NN_Link*> gen_child(vector<NN_Link*>& selected_parents);
	static vector<NN_Link*> set_operate_list(vector<NN_Link*> in_layers);

public:
	vector<NN_Link*> input_nodes;
	vector<NN_Link*> output_nodes;
	vector<NN_Link*> operate_list;

	NN_Model(NN_Model* p_parent);
	NN_Model(const NN_Coupler<NN_Link>& inputs, const NN_Coupler<NN_Link>& outputs, const string& model_name);
	~NN_Model();

	NN_Coupler<NN_Link> operator()(NN_Coupler<NN_Link> m_prev_link);
	void inner_link(NN_Link* m_prev_link);
	
	NN_Link* get_prev_ptr(NN_Link* p_current);
	NN_Link* get_current_ptr(NN_Link* p_prev);

	NN_Link* create_child_link();

	void calculate_output_size(vector<NN_Shape>& input_shape, NN_Shape& output_shape);
	void build(vector<NN_Shape>& input_shape);
	void run_forward(vector<NN_Tensor>& input, NN_Tensor& output);
	void run_backward(vector<NN_Tensor>& input, NN_Tensor& output, NN_Tensor& d_output, vector<NN_Tensor>& d_input);

	void compile(const vector<NN_Loss*>& loss, const vector<NN_Optimizer*>& optimizer);
	NN_Tensor train_on_batch(const vector<NN_Tensor>& samples, const vector<NN_Tensor>& truth);
	NN_Tensor fit(
		const vector<NN_Tensor>& samples,
		const vector<NN_Tensor>& truth,
		uint batch,
		uint iter
	);
	vector<NN_Tensor> predict(const vector<NN_Tensor>& x);
	void summary();
};

NN_Model& Model(const NN_Coupler<NN_Link>& inputs, const NN_Coupler<NN_Link>& outputs, const string& model_name);