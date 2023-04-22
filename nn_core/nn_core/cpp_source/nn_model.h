#pragma once
#include "nn_manager.h"
#include "nn_optimizer.h"
#include "nn_loss.h"


#ifdef FIX_MODE

/**********************************************/
/*                                            */
/*                     Model                  */
/*                                            */
/**********************************************/

class Model : public NN_Layer, public NN_Link {
public:
	/*
	// NN_Layer
	const char* _layer_name;

	// NN_Link
	std::vector<NN_Link*> _prev;
	std::vector<NN_Link*> _next;

	NN_Link* _parent;
	std::vector<NN_Link*> _child;

	std::vector<NN_Tensor*> _input;
	NN_Tensor _output;

	std::vector<NN_Tensor*> _d_output;
	NN_Tensor _d_input;

	std::vector<nn_shape*> _ref_in_shape;
	nn_shape _ref_out_shape;

	std::vector<nn_shape*> _real_in_shape;
	nn_shape _real_out_shape;

	bool is_selected;
	bool trainable;

	NN_Layer* _forward;
	NN_Layer* _backward;
	*/

	std::vector<NN_Link*> _input_nodes;
	std::vector<NN_Link*> _output_nodes;
	std::vector<NN_Link*> _forward_list;
	std::vector<NN_Link*> _backward_list;

	std::vector<int> _output_indices;

	Model(const char* model_name);
	Model(const Layer_t& inputs, const Layer_t& outputs, const char* model_name);
	~Model();

	NN_Link* create_child();
	Layer_t operator()(const Layer_t& prev_node);

	int get_node_index(NN_Link* next_node);
	void set_next_node(NN_Link* next_node, int node_index);
	NN_Tensor<nn_type>& get_output(int node_index);
	std::vector<NN_Tensor<nn_type>*>& get_d_output(int node_index);
	nn_shape& get_out_shape(int node_index);
	void link_prev_child();

	nn_shape calculate_output_size(std::vector<nn_shape*>& input_shape);
	void build(std::vector<nn_shape*>& input_shape);
	NN_Tensor<nn_type> run_forward(cudaStream_t s, std::vector<NN_Tensor<nn_type>*>& input);
	NN_Tensor<nn_type> run_backward(cudaStream_t s, std::vector<NN_Tensor<nn_type>*>& d_output);

	void compile(const std::vector<NN_Loss>& loss, const std::vector<NN_Optimizer>& optimizer);

	template <typename sample_type, typename truth_type>
	std::vector<Tensor<nn_type>> train_on_batch(const std::vector<Tensor<sample_type>>& samples, const std::vector<Tensor<truth_type>>& truth);
	
	template <typename sample_type, typename truth_type>
	std::vector<Tensor<nn_type>> fit(
		const std::vector<Tensor<sample_type>>& samples,
		const std::vector<Tensor<truth_type>>& truth,
		uint batch,
		uint iter
	);

	template <typename d_type>
	std::vector<Tensor<nn_type>> predict(const std::initializer_list<Tensor<d_type>>& x);
	void summary();

	static void check_dimension(const nn_shape& ref_shape, const nn_shape& real_shape);
};

template <typename sample_type, typename truth_type>
std::vector<Tensor<nn_type>> Model::train_on_batch(const std::vector<Tensor<sample_type>>& samples, const std::vector<Tensor<truth_type>>& truth) {
	return Tensor<nn_type>();
}

template <typename sample_type, typename truth_type>
std::vector<Tensor<nn_type>> Model::fit(
	const std::vector<Tensor<sample_type>>& samples,
	const std::vector<Tensor<truth_type>>& truth,
	uint batch,
	uint iter
) {
	return Tensor<nn_type>();
}

template <typename d_type>
std::vector<Tensor<nn_type>> Model::predict(const std::initializer_list<Tensor<d_type>>& x) {

	if (x.size() != _input_nodes.size()) {
		ErrorExcept(
			"[Model::predict()] %d input layers are different %d samples.",
			_input_nodes.size(), x.size()
		);
	}

	int i = 0;
	for (const Tensor<d_type>& x_input : x) {
		NN_Tensor<d_type> p_input(x_input._shape);
		copy_to_nn_tensor(x_input, p_input);
		NN_Tensor<nn_type> *nn_input = new NN_Tensor<nn_type>(p_input.cast<nn_type>());

		_input_nodes[i]->_input.push_back(nn_input);
		++i;
	}
	for (NN_Link* node : _forward_list) {
		node->_output = node->_forward->run_forward(NN_Manager::_stream, node->_input);
	}
	for (NN_Link* p_input : _input_nodes) {
		delete p_input->_input[0];
		p_input->_input.clear();
	}

	std::vector<Tensor<nn_type>> output;

	for (NN_Link* node : _output_nodes) {
		Tensor<nn_type> p_output(node->_output._shape);
		copy_to_tensor(node->_output, p_output);
		output.push_back(p_output);
	}
	
	return output;
}

#endif

#ifndef FIX_MODE

/**********************************************/
/*                                            */
/*                     Model                  */
/*                                            */
/**********************************************/

class Model : public NN_Layer, public NN_Link {
public:
	/*
	// NN_Layer
	const char* _layer_name;

	// NN_Link
	vector<NN_Link*> _prev;
	vector<NN_Link*> _next;

	NN_Link* _parent;
	vector<NN_Link*> _child;

	vector<NN_Tensor*> _input;
	NN_Tensor _output;

	vector<NN_Tensor*> _d_output;
	NN_Tensor _d_input;

	bool is_selected;
	bool trainable;

	NN_Layer* _forward;
	NN_Layer* _backward;
	*/

	std::vector<NN_Link*> _input_nodes;
	std::vector<NN_Link*> _output_nodes;
	std::vector<NN_Link*> _forward_list;
	std::vector<NN_Link*> _backward_list;

	std::vector<int> _output_indices;

	Model(const char* model_name);
	Model(std::initializer_list<Layer_t> inputs, std::initializer_list<Layer_t> outputs, const char* model_name);
	~Model();

	NN_Link* create_child();
	Layer_t operator()(std::initializer_list<Layer_t> prev_node);
	Layer_t operator()(Layer_t& prev_node);

	int get_node_index(NN_Link* next_node);
	void set_next_node(NN_Link* next_node, int node_index);
	NN_Tensor& get_output(int node_index);
	std::vector<NN_Tensor*>& get_d_output(int node_index);
	nn_shape& get_out_shape(int node_index);
	void link_prev_child();

	nn_shape calculate_output_size(std::vector<nn_shape*>& input_shape);
	void build(std::vector<nn_shape*>& input_shape);
	NN_Tensor run_forward(cudaStream_t s, std::vector<NN_Tensor*>& input);
	NN_Tensor run_backward(cudaStream_t s, std::vector<NN_Tensor*>& d_output);

	void compile(const std::vector<NN_Loss>& loss, const std::vector<NN_Optimizer>& optimizer);
	NN_Tensor train_on_batch(const std::vector<NN_Tensor>& samples, const std::vector<NN_Tensor>& truth);
	NN_Tensor fit(
		const std::vector<NN_Tensor>& samples,
		const std::vector<NN_Tensor>& truth,
		uint batch,
		uint iter
	);
	std::vector<NN_Tensor> predict(std::vector<NN_Tensor>&& x);
	void summary();
};

#endif