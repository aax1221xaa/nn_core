#pragma once
#include "nn_tensor.h"
#include "nn_optimizer.h"



class NN_Link;

typedef	List<NN_Link::NN_Ptr> Layer_t;


/**********************************************/
/*                                            */
/*                  NN_Layer                  */
/*                                            */
/**********************************************/

class NN_Layer {
public:
	const char* _layer_name;

	NN_Layer(const char* layer_name);
	virtual ~NN_Layer();

	virtual void get_output_shape(const std::vector<NN_Shape>& input_shape, std::vector<NN_Shape>& output_shape);
	virtual void build(const std::vector<NN_Shape>& input_shape);
	virtual void run_forward(NN_Stream& st, const std::vector<GpuTensor<nn_type>>& input, std::vector<GpuTensor<nn_type>>& output);
	virtual void run_forward(const Tensor<nn_type>& src, GpuTensor<nn_type>& dst);
	virtual void run_backward(NN_Stream& st, const std::vector<GpuTensor<nn_type>>& d_output, std::vector<GpuTensor<nn_type>>& d_input);
};


/**********************************************/
/*                                            */
/*                  NN_Input                  */
/*                                            */
/**********************************************/

class NN_Input : public NN_Layer {
public:
	NN_Shape _shape;

	NN_Input(const NN_Shape& input_size, int batch = -1, const char* _layer_name = "Input");
	~NN_Input();

	void get_output_shape(const std::vector<NN_Shape>& input_shape, std::vector<NN_Shape>& output_shape);
	void build(const std::vector<NN_Shape>& input_shape);
	void run_forward(NN_Stream& st, const std::vector<GpuTensor<nn_type>>& input, std::vector<GpuTensor<nn_type>>& output);
	void run_forward(const Tensor<nn_type>& src, GpuTensor<nn_type>& dst);
};


/**********************************************/
/*                                            */
/*                   NN_Link                  */
/*                                            */
/**********************************************/

class NN_Link {
private:
	int _index;

	std::vector<NN_Link*> _prev;
	std::vector<NN_Link*> _next;

	NN_Layer* _layer;

public:
	struct NN_Ptr {
		int _index;
		NN_Link* _node;
	};

	bool trainable;

	NN_Link();
	virtual ~NN_Link();

	const std::vector<NN_Link*>& get_prev_nodes() const;
	const std::vector<NN_Link*>& get_next_nodes() const;
	
	void set_prev_node(NN_Link* node);
	void set_next_node(NN_Link* node);

	NN_Layer& get_layer();
	void set_layer(NN_Layer* layer);

	const int& get_index() const;
	void set_index(int index);

	Layer_t operator()(Layer_t prev_node);

	virtual NN_Link* create_child();
	virtual void set_next_link(NN_Link* node, int index);
};


/**********************************************/
/*                                            */
/*                  NN_Manager                */
/*                                            */
/**********************************************/

class NN_Manager {
	NN_Stream _stream;

	std::vector<bool> _is_static;
	std::vector<NN_Link*> _nodes;
	std::vector<NN_Layer*> _layers;

	int _node_counter;

	std::vector<std::vector<NN_Shape>> _out_shapes;
	std::vector<std::vector<GpuTensor<nn_type>>> _outputs;

public:
	NN_Manager();
	~NN_Manager();

	NN_Stream& get_streams();
	
	std::vector<NN_Link*>& get_nodes();
	std::vector<NN_Layer*>& get_layers();

	void set_nodes(NN_Link* node);
	void set_static_node(NN_Link* const node);

	void set_reserved_shapes();
	void set_reserved_outputs();

	std::vector<NN_Shape>& get_node_shape(int index);
	std::vector<GpuTensor<nn_type>>& get_node_output(int index);

	void clear_shapes();
	void clear_outputs();

	Layer_t input(const NN_Shape& input_size, int batch, const char* layer_name);

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

void set_random_uniform(GpuTensor<nn_type>& g_mat);