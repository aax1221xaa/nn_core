#pragma once
#include "nn_tensor.h"
#include "nn_optimizer.h"


#define Layer_t	List<NN_Link::NN_Ptr>

/**********************************************/
/*                                            */
/*                  NN_Layer                  */
/*                                            */
/**********************************************/

class NN_BackPropLayer {
public:
	virtual ~NN_BackPropLayer();
	virtual void set_dio(
		std::vector<nn_shape>& in_shape,
		std::vector<GpuTensor<nn_type>>& d_outputs,
		std::vector<GpuTensor<nn_type>>& d_inputs
	);
	virtual void run_backprop(
		std::vector<cudaStream_t>& s,
		std::vector<GpuTensor<nn_type>>& inputs,
		GpuTensor<nn_type>& outputs,
		GpuTensor<nn_type>& d_output,
		std::vector<GpuTensor<nn_type>>& d_input
	);
};

class NN_Layer {
public:
	const char* _layer_name;

	NN_Layer(const char* layer_name);
	virtual ~NN_Layer();
	virtual nn_shape calculate_output_size(nn_shape& input_shape) = 0;
	virtual void set_io(std::vector<GpuTensor<nn_type>>& input, nn_shape& out_shape, GpuTensor<nn_type>& output);
	virtual void run_forward(std::vector<cudaStream_t>& stream, std::vector<GpuTensor<nn_type>>& input, GpuTensor<nn_type>& output) = 0;
	virtual NN_BackPropLayer* create_backprop(NN_Optimizer& optimizer);

	virtual std::vector<int> digestion(const std::vector<int>& feed);
};


/**********************************************/
/*                                            */
/*                  NN_Input                  */
/*                                            */
/**********************************************/

class NN_Input : public NN_Layer {
public:
	nn_shape _shape;

	NN_Input(const nn_shape& input_size, int batch, const char* _layer_name);
	~NN_Input();

	nn_shape calculate_output_size(nn_shape& input_shape);
	void set_io(std::vector<GpuTensor<nn_type>>& input, nn_shape& out_shape, GpuTensor<nn_type>& output);
	void run_forward(std::vector<cudaStream_t>& stream, std::vector<GpuTensor<nn_type>>& input, GpuTensor<nn_type>& output);
	NN_BackPropLayer* create_backprop(NN_Optimizer& optimizer);
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
	NN_BackPropLayer* _backprop;

public:
	struct NN_Ptr {
		int _index;
		NN_Link* _node;
	};

	bool trainable;

	NN_Link();
	virtual ~NN_Link();

	std::vector<NN_Link*>& get_prev_nodes();
	std::vector<NN_Link*>& get_next_nodes();
	
	void set_prev_node(NN_Link* node);
	void set_next_node(NN_Link* node);

	NN_Layer& get_layer();
	NN_BackPropLayer& get_backprop();

	void set_layer(NN_Layer* layer);
	void set_backprop(NN_BackPropLayer* backprop);

	int get_index();
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
	cudaStream_t _streams[STREAMS];

	std::vector<bool> _is_static;
	std::vector<NN_Link*> _nodes;
	std::vector<NN_Layer*> _layers;

	int _node_counter;

public:
	NN_Manager();
	~NN_Manager();

	cudaStream_t* get_streams();
	
	std::vector<NN_Link*>& get_nodes();
	std::vector<NN_Layer*>& get_layers();

	void set_nodes(NN_Link* node);
	void set_static_node(NN_Link* const node);

	Layer_t input(const nn_shape& input_size, int batch, const char* layer_name);

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
