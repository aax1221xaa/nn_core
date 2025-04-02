#pragma once
#include "nn_common.h"
#include "nn_tensor.h"
#include "nn_optimizer.h"


/**********************************************/
/*                                            */
/*                 NN_Backward                */
/*                                            */
/**********************************************/

class NN_Backward {
public:
	NN_Backward();
	virtual ~NN_Backward();
	virtual void run( 
		const NN_List<NN_Tensor<nn_type>>& input,
		const NN_List<NN_Tensor<nn_type>>& doutput,
		NN_List<NN_Tensor<nn_type>>& dinput
	);
	virtual NN_Optimizer* create_optimizer(const NN_Optimizer& optimizer);
};

template <class _T>
class NN_Backward_t : public NN_Backward{
public:
	_T& _layer;

	NN_Backward_t(_T& layer);
};

template <class _T>
NN_Backward_t<_T>::NN_Backward_t(_T& layer) :
	_layer(layer)
{
}


/**********************************************/
/*                                            */
/*                  NN_Layer                  */
/*                                            */
/**********************************************/

class NN_Layer {
public:
	const std::string _layer_name;

	NN_Layer(const std::string& layer_name);
	virtual ~NN_Layer();

	virtual void get_output_shape(const NN_List<NN_Shape>& input_shape, NN_List<NN_Shape>& output_shape);
	virtual void build(const NN_List<NN_Shape>& input_shape, NN_List<NN_Tensor<nn_type>>& weights);
	virtual void run(const NN_List<NN_Tensor<nn_type>>& input, NN_List<NN_Tensor<nn_type>>& output);
	virtual NN_Backward* create_backward(std::vector<bool>& mask);
	virtual NN_List<NN_Tensor<nn_type>> get_weight();
	virtual void set_output(const NN_List<NN_Shape>& output_shape, NN_List<NN_Tensor<nn_type>>& input, NN_List<NN_Tensor<nn_type>>& output);
};


/**********************************************/
/*                                            */
/*                  NN_Input                  */
/*                                            */
/**********************************************/

class NN_Input : public NN_Layer {
public:
	NN_Shape _shape;

	NN_Input(const NN_Shape& input_size, int batch = -1, const std::string& layer_name = "Input");
	~NN_Input();

	void get_output_shape(const NN_List<NN_Shape>& input_shape, NN_List<NN_Shape>& output_shape);
	NN_Backward* create_backward(std::vector<bool>& mask);
	void set_output(const NN_List<NN_Shape>& output_shape, NN_List<NN_Tensor<nn_type>>& input, NN_List<NN_Tensor<nn_type>>& output);

	template <typename _sT, typename _dT>
	void trans_data(const NN_Tensor<_sT>& sample, NN_Tensor<_dT>& output) const;
};

template <typename _sT, typename _dT>
void NN_Input::trans_data(const NN_Tensor<_sT>& sample, NN_Tensor<_dT>& output) const {
	output = sample.cast<_dT>();
}


/**********************************************/
/*                                            */
/*                  NN_dInput                 */
/*                                            */
/**********************************************/

class NN_dInput : public NN_Backward_t<NN_Input> {
public:
	NN_dInput(NN_Input& input);
	void run(
		const NN_List<NN_Tensor<nn_type>>& input,
		const NN_List<NN_Tensor<nn_type>>& doutput,
		NN_List<NN_Tensor<nn_type>>& dinput
	);
};


/**********************************************/
/*                                            */
/*                   NN_Link                  */
/*                                            */
/**********************************************/

struct NN_Ptr;

typedef	NN_List<NN_Ptr> Layer_t;

class NN_Link {
protected:
	int _n_id;

	std::vector<NN_Link*> _prev;
	std::vector<NN_Link*> _next;
	std::vector<int> _out_indices;

	NN_Layer* _layer;
	NN_Backward* _backward;
	NN_Optimizer* _optimizer;

public:
	bool trainable;

	NN_Link();
	virtual ~NN_Link();

	const std::vector<NN_Link*>& get_prev_nodes() const;
	const std::vector<NN_Link*>& get_next_nodes() const;

	void set_prev_node(NN_Link* node);
	void set_next_node(NN_Link* node);

	NN_Layer& get_layer();
	void set_layer(NN_Layer* layer);
	NN_Backward* get_backward();
	void set_backward(NN_Backward* backward);
	NN_Optimizer* get_optimizer();
	void set_optimizer(NN_Optimizer* optimizer);

	const int& get_index() const;
	void set_index(int index);

	std::vector<int>& get_out_indices();

	Layer_t operator()(Layer_t prev_node);

	virtual NN_Link* create_child();
	void set_next_link(NN_Link* node, int index);

	int get_out_port(NN_Link* current);
	int get_out_port(NN_Link* current) const;
};

struct NN_Ptr {
	int _n_port;
	NN_Link* _node;
	NN_Shape _shape;
};


/**********************************************/
/*                                            */
/*                  NN_Manager                */
/*                                            */
/**********************************************/

class NN_Manager {
	std::vector<bool> _is_static;
	std::vector<NN_Input*> _input_layers;
	std::vector<NN_Link*> _nodes;
	std::vector<NN_Layer*> _layers;
	std::vector<NN_Backward*> _backward;
	std::vector<NN_Optimizer*> _optimizer;

	int _node_counter;

	NN_List<NN_Tensor<nn_type>> _weights;
	NN_List<NN_Shape> _out_shapes;
	NN_List<NN_Tensor<nn_type>> _outputs;
	NN_List<NN_Tensor<nn_type>> _doutputs;

public:
	NN_Manager();
	~NN_Manager();

	std::vector<NN_Link*>& get_nodes();
	std::vector<NN_Layer*>& get_layers();
	const std::vector<NN_Input*>& get_input_layers();
	std::vector<NN_Backward*>& get_backward();
	void set_optimizer(NN_Optimizer* optimizer);

	NN_List<NN_Tensor<nn_type>>& get_weights();
	void set_nodes(NN_Link* node);
	void set_layers(NN_Layer* layer);
	void set_static_node(NN_Link* const node);
	void set_backward(NN_Backward* backward);

	void set_reserved_weights();
	void set_reserved_shapes();
	void set_reserved_outputs();
	void set_reserved_doutputs();

	NN_List<NN_Shape>& get_node_shape();
	NN_List<NN_Tensor<nn_type>>& get_node_output();
	NN_List<NN_Tensor<nn_type>>& get_node_doutput();

	void clear_weights();
	void clear_shapes();
	void clear_outputs();
	void clear_dinputs();

	Layer_t input(const NN_Shape& input_size, int batch, const std::string& layer_name);

	template <class _T>
	NN_Link& operator()(const _T& layer);
};

template <class _T>
NN_Link& NN_Manager::operator()(const _T& layer) {
	NN_Link* p_node = new NN_Link;
	NN_Layer* p_layer = new _T(layer);

	p_node->set_layer(p_layer);

	set_nodes(p_node);
	_layers.push_back(p_layer);

	return *p_node;
}


/**********************************************/
/*                                            */
/*                    misc					  */
/*                                            */
/**********************************************/

void set_random_uniform(NN_Tensor<nn_type>& tensor, nn_type min, nn_type max);