#pragma once
#include "nn_tensor_plus.h"
#include "../cuda_source/optimizer.cuh"
#include <map>


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
		NN_Stream& st, 
		const NN_List<GpuTensor<nn_type>>& input,
		const NN_List<GpuTensor<nn_type>>& doutput,
		NN_List<GpuTensor<nn_type>>& dinput
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
	static std::map<std::string, uint> _layer_cnt;
	static std::vector<std::string> _uniq_names;

	static std::string add_layer_cnt(
		const std::string& layer_name, 
		const std::string& uniq_name, 
		std::map<std::string, uint>& cnt,
		std::vector<std::string>& names
	);

public:
	std::string _layer_name;
	std::string _uniq_name;

	NN_Layer(const std::string& uniq_name, const std::string& layer_name);
	NN_Layer(const NN_Layer& p);
	NN_Layer(NN_Layer&& p);
	virtual ~NN_Layer();

	const NN_Layer& operator=(const NN_Layer& p);
	const NN_Layer& operator=(NN_Layer&& p);

	virtual void get_output_shape(const NN_List<NN_Shape>& input_shape, NN_List<NN_Shape>& output_shape);
	virtual void build(const NN_List<NN_Shape>& input_shape, NN_List<GpuTensor<nn_type>>& weights);
	virtual void run(NN_Stream& st, const NN_List<GpuTensor<nn_type>>& input, NN_List<GpuTensor<nn_type>>& output);
	virtual NN_Backward* create_backward(std::vector<bool>& mask);
	virtual NN_List<GpuTensor<nn_type>> get_weight();
	virtual void set_output(const NN_List<NN_Shape>& output_shape, NN_List<GpuTensor<nn_type>>& input, NN_List<GpuTensor<nn_type>>& output);
};


/**********************************************/
/*                                            */
/*                  NN_Input                  */
/*                                            */
/**********************************************/

class NN_Input : public NN_Layer {
public:
	NN_Shape _shape;

	NN_Input(const NN_Shape& input_size, int batch = -1, const std::string& name = "");
	~NN_Input();

	void get_output_shape(const NN_List<NN_Shape>& input_shape, NN_List<NN_Shape>& output_shape);
	void build(const NN_List<NN_Shape>& input_shape, NN_List<GpuTensor<nn_type>>& weights);
	NN_Backward* create_backward(std::vector<bool>& mask);
	void set_output(const NN_List<NN_Shape>& output_shape, NN_List<GpuTensor<nn_type>>& input, NN_List<GpuTensor<nn_type>>& output);

	void trans_data(const Tensor<uchar>& sample, GpuTensor<nn_type>& output) const;
};


/**********************************************/
/*                                            */
/*                  NN_dInput                 */
/*                                            */
/**********************************************/

class NN_dInput : public NN_Backward_t<NN_Input> {
public:
	NN_dInput(NN_Input& input);
	void run(
		NN_Stream& st,
		const NN_List<GpuTensor<nn_type>>& input,
		const NN_List<GpuTensor<nn_type>>& doutput,
		NN_List<GpuTensor<nn_type>>& dinput
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
	class Params {
	public:
		NN_Stream _stream;

		std::vector<bool> _is_static;
		std::vector<NN_Input*> _input_layers;
		std::vector<NN_Link*> _nodes;
		std::vector<NN_Layer*> _layers;
		std::vector<NN_Backward*> _backward;
		std::vector<NN_Optimizer*> _optimizer;

		int _node_counter;

		NN_List<GpuTensor<nn_type>> _weights;
		NN_List<NN_Shape> _out_shapes;
		NN_List<GpuTensor<nn_type>> _outputs;
		NN_List<GpuTensor<nn_type>> _doutputs;

		Params() : _node_counter(0) {}
	};

	std::shared_ptr<Params> _params;

	static void destroy_params(Params* p);

public:
	NN_Manager();
	NN_Manager(const NN_Manager& p);
	NN_Manager(NN_Manager&& p);
	~NN_Manager();

	const NN_Manager& operator=(const NN_Manager& p);
	const NN_Manager& operator=(NN_Manager&& p);

	NN_Stream& get_streams();

	std::vector<NN_Link*>& get_nodes();
	std::vector<NN_Layer*>& get_layers();
	const std::vector<NN_Input*>& get_input_layers();
	std::vector<NN_Backward*>& get_backward();
	NN_List<GpuTensor<nn_type>>& get_weights();

	const NN_Stream& get_streams() const;

	const std::vector<NN_Link*>& get_nodes() const;
	const std::vector<NN_Layer*>& get_layers() const;
	const std::vector<NN_Input*>& get_input_layers() const;
	const std::vector<NN_Backward*>& get_backward() const;
	const NN_List<GpuTensor<nn_type>>& get_weights() const;

	void set_optimizer(NN_Optimizer* optimizer);
	void set_nodes(NN_Link* node);
	void set_layers(NN_Layer* layer);
	void set_static_node(NN_Link* const node);
	void set_backward(NN_Backward* backward);

	void set_resize_weights();
	void set_resize_shapes();
	void set_resize_outputs();
	void set_resize_doutputs();

	NN_List<NN_Shape>& get_node_shape();
	NN_List<GpuTensor<nn_type>>& get_node_output();
	NN_List<GpuTensor<nn_type>>& get_node_doutput();

	const NN_List<NN_Shape>& get_node_shape() const;
	const NN_List<GpuTensor<nn_type>>& get_node_output() const;
	const NN_List<GpuTensor<nn_type>>& get_node_doutput() const;

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
	_params->_layers.push_back(p_layer);

	return *p_node;
}


/**********************************************/
/*                                            */
/*                    misc					  */
/*                                            */
/**********************************************/

void set_random_uniform(GpuTensor<nn_type>& tensor, nn_type a, nn_type b);