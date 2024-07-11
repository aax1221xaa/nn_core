#pragma once
#include "nn_list.h"
#include "nn_tensor.h"
#include "../cuda_source/optimizer.cuh"


struct NN_Ptr;

typedef	NN_List<NN_Ptr> Layer_t;

/**********************************************/
/*                                            */
/*                 NN_Backward                */
/*                                            */
/**********************************************/

class NN_Backward {
public:
	NN_Optimizer& _optimizer;

	NN_Backward(NN_Optimizer& optimizer);
	virtual ~NN_Backward();

	virtual void get_dinput_shape(const NN_List<NN_Shape>& dout_shape, NN_List<NN_Shape>& din_shape);
	virtual void run(
		NN_Stream& st, 
		const NN_List<GpuTensor<nn_type>>& input,
		const NN_List<GpuTensor<nn_type>>& doutput,
		NN_List<GpuTensor<nn_type>>& dinput
	);
};


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

	virtual void get_output_shape(const NN_List<NN_Shape>& input_shape, NN_List<NN_Shape>& output_shape);
	virtual void build(const NN_List<NN_Shape>& input_shape, std::vector<GpuTensor<nn_type>>& weights);
	virtual void run(NN_Stream& st, const NN_List<GpuTensor<nn_type>>& input, NN_List<GpuTensor<nn_type>>& output);
	virtual NN_Backward* create_backward(NN_Optimizer& optimizer, std::vector<bool>& mask);
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

	void(*p_convert)(const void* p_src, void* p_dst, const tbb::blocked_range<size_t>& q);

	NN_Input(const NN_Shape& input_size, int batch = -1, const char* layer_name = "Input", void(*convert_f)(const void*, void*, const tbb::blocked_range<size_t>&) = NULL);
	~NN_Input();

	void get_output_shape(const NN_List<NN_Shape>& input_shape, NN_List<NN_Shape>& output_shape);
	void build(const NN_List<NN_Shape>& input_shape, std::vector<GpuTensor<nn_type>>& weights);
	template <typename _sT, typename _dT>
	void trans_data(const Tensor<_sT>& sample, GpuTensor<_dT>& output) const;
	NN_Backward* create_backward(NN_Optimizer& optimizer, std::vector<bool>& mask);

	void set_output(const NN_List<NN_Shape>& output_shape, NN_List<GpuTensor<nn_type>>& input, NN_List<GpuTensor<nn_type>>& output);
};

template <typename _sT, typename _dT>
void NN_Input::trans_data(const Tensor<_sT>& sample, GpuTensor<_dT>& output) const {
	Tensor<_sT> src(sample.get_shape());
	Tensor<_dT> dst(output.get_shape());

	src = sample;

	if (p_convert) {
		tbb::parallel_for(
			tbb::blocked_range<size_t>(0, src.get_shape().total_size()),
			[&](const tbb::blocked_range<size_t>& q) {

			const _sT* p_src = src.get_ptr();
			_dT* p_dst = dst.get_ptr();

			(*p_convert)(p_src, p_dst, q);
		}
		);
	}
	else {
		tbb::parallel_for(
			tbb::blocked_range<size_t>(0, src.get_shape().total_size()),
			[&](const tbb::blocked_range<size_t>& q) {

			const _sT* p_src = src.get_ptr();
			_dT* p_dst = dst.get_ptr();

			for (size_t i = q.begin(); i < q.end(); ++i) {
				p_dst[i] = (_dT)p_src[i];
			}
		}
		);
	}

	output = dst;
}


/**********************************************/
/*                                            */
/*                 NN_dInput                  */
/*                                            */
/**********************************************/

class NN_dInput : public NN_Backward {
public:
	NN_Input& _input;

	NN_dInput(NN_Input& input, NN_Optimizer& optimizer);
	
	void get_dinput_shape(const NN_List<NN_Shape>& dout_shape, NN_List<NN_Shape>& din_shape);
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

class NN_Link {
protected:
	int _index;

	std::vector<NN_Link*> _prev;
	std::vector<NN_Link*> _next;

	NN_Layer* _layer;
	NN_Backward* _backward;

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
	NN_Backward& get_backward();
	void set_backward(NN_Backward* backward);

	const int& get_index() const;
	void set_index(int index);

	void set_weights(GpuTensor<nn_type>& weight);
	std::vector<GpuTensor<nn_type>>& get_weights();

	Layer_t operator()(Layer_t prev_node);

	virtual NN_Link* create_child();
	virtual void set_next_link(NN_Link* node, int index);
};

struct NN_Ptr {
	int _index;
	NN_Link* _node;
};


/**********************************************/
/*                                            */
/*                  NN_Manager                */
/*                                            */
/**********************************************/

class NN_Manager {
	NN_Stream _stream;

	std::vector<bool> _is_static;
	std::vector<NN_Input*> _input_layers;
	std::vector<NN_Link*> _nodes;
	std::vector<NN_Layer*> _layers;
	std::vector<NN_Backward*> _backward;

	int _node_counter;

	std::vector<GpuTensor<nn_type>> _weights;
	NN_List<NN_Shape> _out_shapes;
	NN_List<GpuTensor<nn_type>> _outputs;
	NN_List<GpuTensor<nn_type>> _dinputs;

public:
	NN_Manager();
	~NN_Manager();

	NN_Stream& get_streams();

	std::vector<NN_Link*>& get_nodes();
	std::vector<NN_Layer*>& get_layers();
	const std::vector<NN_Input*>& get_input_layers();
	std::vector<NN_Backward*>& get_backward();

	std::vector<GpuTensor<nn_type>>& get_weights();

	void set_nodes(NN_Link* node);
	void set_static_node(NN_Link* const node);
	void set_backward(NN_Backward* backward);

	void set_reserved_shapes();
	void set_reserved_outputs();
	void set_reserved_dinputs();

	NN_List<NN_Shape>& get_node_shape();
	NN_List<GpuTensor<nn_type>>& get_node_output();
	NN_List<GpuTensor<nn_type>>& get_node_dinput();

	void clear_weights();
	void clear_shapes();
	void clear_outputs();
	void clear_dinputs();

	Layer_t input(const NN_Shape& input_size, int batch, const char* layer_name, void(*convert_f)(const void*, void*, const tbb::blocked_range<size_t>&) = NULL);

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

void set_random_uniform(GpuTensor<nn_type>& tensor, nn_type a, nn_type b);