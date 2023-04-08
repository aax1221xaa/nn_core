#pragma once
#include "nn_manager.h"
#include "nn_optimizer.h"
#include "nn_loss.h"


/**********************************************/
/*                                            */
/*                     Model                  */
/*                                            */
/**********************************************/

/*
class NN_Layer {
public:
	const char* _layer_name;

	NN_Layer(const char* layer_name);
	virtual ~NN_Layer();

	virtual shape_type calculate_output_size(shape_type& input_shape) = 0;
	virtual void build(shape_type& input_shape);
	virtual NN_Tensor run_forward(cudaStream_t s, vector<NN_Tensor*>& input) = 0;
	virtual NN_Tensor run_backward(cudaStream_t s, vector<NN_Tensor*>& d_output) = 0;
};

class NN_Link {
public:
	vector<NN_Link*> _prev;
	vector<NN_Link*> _next;

	bool is_selected;
	bool trainable;

	NN_Layer* _forward;
	NN_Layer* _backward;

	NN_Link();
	~NN_Link();

	virtual List<NN_Link*> operator()(List<NN_Link*> prev_node);
};

struct NN_Container {
	vector<NN_Tensor*> _input;
	NN_Tensor _output;

	vector<NN_Tensor*> _d_output;
	NN_Tensor _d_input;

	NN_Layer* _forward;
	NN_Layer* _backward;

	NN_Container() :
		_forward(NULL),
		_backward(NULL)
	{
	}
};
*/

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
	vector<NN_Link*> _input_nodes;
	vector<NN_Link*> _output_nodes;
	vector<NN_Link*> _forward_list;
	vector<NN_Link*> _backward_list;

	Model(const char* model_name);
	Model(initializer_list<vector<Layer_t<NN_Link>>> inputs, initializer_list<vector<Layer_t<NN_Link>>> outputs, const char* model_name);
	~Model();

	NN_Link* create_child();
	vector<Layer_t<NN_Link>> operator()(vector<Layer_t<NN_Link>>& prev_node);
	vector<Layer_t<NN_Link>> operator()(initializer_list<vector<Layer_t<NN_Link>>> prev_node);

	shape_type calculate_output_size(shape_type& input_shape);
	void build(shape_type& input_shape);
	NN_Tensor run_forward(cudaStream_t s, vector<NN_Tensor*>& input);
	NN_Tensor run_backward(cudaStream_t s, vector<NN_Tensor*>& d_output);

	void compile(const vector<NN_Loss>& loss, const vector<NN_Optimizer>& optimizer);
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