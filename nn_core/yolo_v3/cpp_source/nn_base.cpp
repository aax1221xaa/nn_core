#include "nn_base.h"
#include <opencv2/opencv.hpp>
#include <random>



/**********************************************/
/*                                            */
/*                 NN_Backward                */
/*                                            */
/**********************************************/

NN_Backward::NN_Backward() {

}

NN_Backward::~NN_Backward() {

}

void NN_Backward::run(
	NN_Stream& st, 
	const NN_List<GpuTensor<nn_type>>& input,
	const NN_List<GpuTensor<nn_type>>& doutput,
	NN_List<GpuTensor<nn_type>>& dinput
) {

}

NN_Optimizer* NN_Backward::create_optimizer(const NN_Optimizer& optimizer) {
	return NULL;
}


/**********************************************/
/*                                            */
/*                  NN_Layer                  */
/*                                            */
/**********************************************/

std::map<std::string, uint> NN_Layer::_layer_cnt;

std::vector<std::string> NN_Layer::_uniq_names;

std::string NN_Layer::add_layer_cnt(
	const std::string& layer_name,
	const std::string& uniq_name,
	std::map<std::string, uint>& cnt,
	std::vector<std::string>& names
) {
	std::string curr_name;

	if (cnt.find(layer_name) != cnt.end()) ++cnt[layer_name];
	else cnt[layer_name] = 0;

	if (uniq_name == "") {
		if (cnt[layer_name]) curr_name = layer_name + '_' + std::to_string(cnt[layer_name]);
		else curr_name = layer_name;
	}
	else curr_name = uniq_name;

	for (const std::string& str : names) {
		if (str == curr_name) {
			ErrorExcept(
				"[NN_Layer::add_layer_cnt] This layer Already added. %s",
				str.c_str()
			);
		}
	}

	names.push_back(curr_name);

	return curr_name;
}

NN_Layer::NN_Layer(const std::string& uniq_name, const std::string& layer_name) :
	_layer_name(layer_name),
	_uniq_name(add_layer_cnt(layer_name, uniq_name, _layer_cnt, _uniq_names))
{

}

NN_Layer::NN_Layer(const NN_Layer& p) :
	_layer_name(p._layer_name),
	_uniq_name(p._uniq_name)
{

}

NN_Layer::NN_Layer(NN_Layer&& p) :
	_layer_name(p._layer_name),
	_uniq_name(p._uniq_name)
{

}

NN_Layer::~NN_Layer() {

}

const NN_Layer& NN_Layer::operator=(const NN_Layer& p) {
	if (this == &p) return *this;

	_layer_name = p._layer_name;
	_uniq_name = p._uniq_name;

	return *this;
}

const NN_Layer& NN_Layer::operator=(NN_Layer&& p) {
	_layer_name = p._layer_name;
	_uniq_name = p._uniq_name;

	return *this;
}

void NN_Layer::get_output_shape(const NN_List<NN_Shape>& input_shape, NN_List<NN_Shape>& output_shape) {
	ErrorExcept(
		"[NN_Layer::get_output_shape] Make this function."
	);
}

void NN_Layer::build(const NN_List<NN_Shape>& input_shape, NN_List<GpuTensor<nn_type>>& weights) {

}

void NN_Layer::run(NN_Stream& st, const NN_List<GpuTensor<nn_type>>& input, NN_List<GpuTensor<nn_type>>& output) {
	ErrorExcept(
		"[NN_Layer::run] Make this function."
	);
}

NN_Backward* NN_Layer::create_backward(std::vector<bool>& mask) {
	return NULL;
}

NN_List<GpuTensor<nn_type>> NN_Layer::get_weight() {
	return NN_List<GpuTensor<nn_type>>();
}

void NN_Layer::set_output(const NN_List<NN_Shape>& output_shape, NN_List<GpuTensor<nn_type>>& input, NN_List<GpuTensor<nn_type>>& output) {
	for (const NN_List<NN_Shape>& shape : output_shape) output.append(GpuTensor<nn_type>(shape.val()));
}


/**********************************************/
/*                                            */
/*                  NN_Input                  */
/*                                            */
/**********************************************/

NN_Input::NN_Input(const NN_Shape& input_size, int batch, const std::string& name) :
	NN_Layer(name, "input"),
	_shape(input_size)
{
	_shape.push_front(batch);
}

NN_Input::~NN_Input() {

}

void NN_Input::get_output_shape(const NN_List<NN_Shape>& input_shape, NN_List<NN_Shape>& output_shape) {
	if (input_shape.size() > 1) {
		ErrorExcept(
			"[NN_Input::get_output_shape] Input node can't take %ld tensor shapes.",
			input_shape.size()
		);
	}
	else if (input_shape.size() == 0) {
		output_shape.append(_shape);
	}
	else {
		const NN_Shape& in_shape = input_shape[0].val();

		if (in_shape.ranks() != _shape.ranks()) {
			ErrorExcept(
				"[NN_Input::get_output_shape] Input excpected dimensions are %s. but take dimensions are %s.",
				shape_to_str(_shape),
				shape_to_str(in_shape)
			);
		}

		NN_Shape out_shape(_shape.ranks());

		int i = 0;
		for (const int& n : in_shape) {
			//if (n < 0) {
			//	ErrorExcept(
			//		"[NN_Input::get_output_shape] The dimensions of inputs must be greater than 0. %s",
			//		shape_to_str(in_shape)
			//	);
			//}

			if (_shape[i] < 0) out_shape[i] = n;
			else out_shape[i] = _shape[i];
			++i;
		}

		output_shape.append(out_shape);
	}
}

void NN_Input::build(const NN_List<NN_Shape>& input_shape, NN_List<GpuTensor<nn_type>>& weights) {

}

NN_Backward* NN_Input::create_backward(std::vector<bool>& mask) {
	return new NN_dInput(*this);
}

void NN_Input::set_output(const NN_List<NN_Shape>& output_shape, NN_List<GpuTensor<nn_type>>& input, NN_List<GpuTensor<nn_type>>& output) {
	if (input.size()) output.append(input[0].val());
	else output.append(GpuTensor<nn_type>(output_shape[0].val()));
}

void NN_Input::trans_data(const Tensor<uchar>& sample, GpuTensor<nn_type>& output) const {
	output = sample.cast<nn_type>();
}


/**********************************************/
/*                                            */
/*                 NN_dInput                  */
/*                                            */
/**********************************************/

NN_dInput::NN_dInput(NN_Input& input) :
	NN_Backward_t(input)
{
}

void NN_dInput::run(
	NN_Stream& st,
	const NN_List<GpuTensor<nn_type>>& input,
	const NN_List<GpuTensor<nn_type>>& doutput,
	NN_List<GpuTensor<nn_type>>& dinput
) {
	
}



/**********************************************/
/*                                            */
/*                   NN_Link                  */
/*                                            */
/**********************************************/

NN_Link::NN_Link() :
	_layer(NULL),
	_backward(NULL),
	_optimizer(NULL),
	_n_id(0),
	trainable(true)
{
}

NN_Link::~NN_Link() {

}

NN_Link* NN_Link::create_child() {
	NN_Link* child_node = new NN_Link;

	child_node->_layer = _layer;
	child_node->trainable = trainable;
	child_node->_out_indices = _out_indices;

	return child_node;
}

const std::vector<NN_Link*>& NN_Link::get_prev_nodes() const {
	return _prev;
}

const std::vector<NN_Link*>& NN_Link::get_next_nodes() const {
	return _next;
}

void NN_Link::set_prev_node(NN_Link* node) {
	_prev.push_back(node);
}

void NN_Link::set_next_node(NN_Link* node) {
	_next.push_back(node);
}

NN_Layer& NN_Link::get_layer() {
	return *_layer;
}

NN_Backward* NN_Link::get_backward() {
	return _backward;
}

void NN_Link::set_backward(NN_Backward* backward) {
	_backward = backward;
}

NN_Optimizer* NN_Link::get_optimizer() {
	return _optimizer;
}

void NN_Link::set_optimizer(NN_Optimizer* optimizer) {
	_optimizer = optimizer;
}

void NN_Link::set_layer(NN_Layer* layer) {
	_layer = layer;
}

const int& NN_Link::get_index() const {
	return _n_id;
}

void NN_Link::set_index(int index) {
	_n_id = index;
}

std::vector<int>& NN_Link::get_out_indices() {
	return _out_indices;
}

Layer_t NN_Link::operator()(Layer_t prev_node) {
	NN_List<NN_Shape> in_shapes;
	NN_List<NN_Shape> out_shapes;

	for (Layer_t& p_prev_node : prev_node) {
		NN_Ptr& prev_ptr = p_prev_node.is_scalar() ? p_prev_node.val() : p_prev_node[0].val();

		set_prev_node(prev_ptr._node);
		prev_ptr._node->set_next_link(this, prev_ptr._n_port);

		in_shapes.append(prev_ptr._shape);
	}

	_layer->get_output_shape(in_shapes, out_shapes);

	Layer_t out_ports;
	int i = 0;
	for (NN_List<NN_Shape>& out_shape : out_shapes) {
		out_ports.append({ i++, this, out_shape.val() });
	}

	return out_ports;
}

void NN_Link::set_next_link(NN_Link* node, int index) {
	_next.push_back(node);
	_out_indices.push_back(index);
}

int NN_Link::get_out_port(NN_Link* current) {
	int i = -1;
	for (NN_Link* p_next : _next) {
		++i;
		if (p_next == current) break;
	}

	return _out_indices[i];
}

int NN_Link::get_out_port(NN_Link* current) const {
	int i = -1;
	for (NN_Link* p_next : _next) {
		++i;
		if (p_next == current) break;
	}

	return _out_indices[i];
}


/**********************************************/
/*                                            */
/*                  NN_Manager                */
/*                                            */
/**********************************************/

void NN_Manager::destroy_params(Params* p) {
	try {
		for (size_t i = 0; i < p->_nodes.size(); ++i) {
			if (!p->_is_static[i]) delete p->_nodes[i];
		}
		for (NN_Layer* layer : p->_layers) delete layer;
		for (NN_Backward* backward : p->_backward) delete backward;
		for (NN_Optimizer* optimizer : p->_optimizer) delete optimizer;

		p->_is_static.clear();
		p->_input_layers.clear();
		p->_nodes.clear();
		p->_layers.clear();
		p->_backward.clear();
		p->_optimizer.clear();
	}
	catch (NN_Exception& e) {
		e.put();
	}

	delete p;
}

NN_Manager::NN_Manager() :
	_params(std::shared_ptr<Params>(new Params(), destroy_params))
{
}

NN_Manager::NN_Manager(const NN_Manager& p) :
	_params(p._params)
{

}

NN_Manager::NN_Manager(NN_Manager&& p) :
	_params(p._params)
{

}

NN_Manager::~NN_Manager() {

}

const NN_Manager& NN_Manager::operator=(const NN_Manager& p) {
	if (this == &p) return *this;

	_params = p._params;

	return *this;
}

const NN_Manager& NN_Manager::operator=(NN_Manager&& p) {
	_params = p._params;

	return *this;
}

NN_Stream& NN_Manager::get_streams() {
	return _params->_stream;
}

std::vector<NN_Link*>& NN_Manager::get_nodes() {
	return _params->_nodes;
}

std::vector<NN_Layer*>& NN_Manager::get_layers() {
	return _params->_layers;
}

const std::vector<NN_Input*>& NN_Manager::get_input_layers() {
	return _params->_input_layers;
}

std::vector<NN_Backward*>& NN_Manager::get_backward() {
	return _params->_backward;
}

NN_List<GpuTensor<nn_type>>& NN_Manager::get_weights() {
	return _params->_weights;
}

const NN_Stream& NN_Manager::get_streams() const {
	return _params->_stream;
}

const std::vector<NN_Link*>& NN_Manager::get_nodes() const {
	return _params->_nodes;
}

const std::vector<NN_Layer*>& NN_Manager::get_layers() const {
	return _params->_layers;
}

const std::vector<NN_Input*>& NN_Manager::get_input_layers() const {
	return _params->_input_layers;
}

const std::vector<NN_Backward*>& NN_Manager::get_backward() const {
	return _params->_backward;
}

const NN_List<GpuTensor<nn_type>>& NN_Manager::get_weights() const {
	return _params->_weights;
}

void NN_Manager::set_optimizer(NN_Optimizer* optimizer) {
	_params->_optimizer.push_back(optimizer);
}

void NN_Manager::set_nodes(NN_Link* node) {
	node->set_index(_params->_node_counter++);
	_params->_nodes.push_back(node);
	_params->_is_static.push_back(false);
}

void NN_Manager::set_layers(NN_Layer* layer) {
	_params->_layers.push_back(layer);
}

void NN_Manager::set_static_node(NN_Link* const node) {
	node->set_index(_params->_node_counter++);
	_params->_nodes.push_back(node);
	_params->_is_static.push_back(true);
}

void NN_Manager::set_backward(NN_Backward* backward) {
	_params->_backward.push_back(backward);
}

void NN_Manager::set_resize_weights() {
	_params->_weights.resize(_params->_nodes.size());
}

void NN_Manager::set_resize_shapes() {
	_params->_out_shapes.resize(_params->_nodes.size());
}

void NN_Manager::set_resize_outputs() {
	_params->_outputs.resize(_params->_nodes.size());
}

void NN_Manager::set_resize_doutputs() {
	_params->_doutputs.resize(_params->_nodes.size());
}

NN_List<NN_Shape>& NN_Manager::get_node_shape() {
	return _params->_out_shapes;
}

NN_List<GpuTensor<nn_type>>& NN_Manager::get_node_output() {
	return _params->_outputs;
}

NN_List<GpuTensor<nn_type>>& NN_Manager::get_node_doutput() {
	return _params->_doutputs;
}

const NN_List<NN_Shape>& NN_Manager::get_node_shape() const {
	return _params->_out_shapes;
}

const NN_List<GpuTensor<nn_type>>& NN_Manager::get_node_output() const {
	return _params->_outputs;
}

const NN_List<GpuTensor<nn_type>>& NN_Manager::get_node_doutput() const {
	return _params->_doutputs;
}

void NN_Manager::clear_weights() {
	_params->_weights.clear();
}

void NN_Manager::clear_shapes() {
	_params->_out_shapes.clear();
}

void NN_Manager::clear_outputs() {
	_params->_outputs.clear();
}

void NN_Manager::clear_dinputs() {
	_params->_doutputs.clear();
}

Layer_t NN_Manager::input(const NN_Shape& input_size, int batch, const std::string& layer_name) {
	NN_Input* layer = new NN_Input(input_size, batch, layer_name);
	NN_Link* node = new NN_Link;

	_params->_input_layers.push_back(layer);
	node->set_layer(layer);

	set_nodes(node);
	_params->_layers.push_back(layer);
	
	NN_List<NN_Shape> out_shape;

	layer->get_output_shape(NN_List<NN_Shape>(), out_shape);

	return NN_Ptr({ 0, node, out_shape[0].val() });
}


/**********************************************/
/*                                            */
/*                    misc					  */
/*                                            */
/**********************************************/

void set_random_uniform(GpuTensor<nn_type>& tensor, nn_type a, nn_type b) {
	std::random_device rd;
	cv::RNG rng(rd());

	cv::Mat tmp(tensor.get_shape().get_dims(), get_type(nn_type(0), 1));

	rng.fill(tmp, cv::RNG::UNIFORM, a, b, true);

	check_cuda(
		cudaMemcpy(
			tensor.get_ptr(),
			tmp.ptr<nn_type>(0),
			sizeof(nn_type) * tensor.get_shape().total_size(),
			cudaMemcpyHostToDevice
		)
	);
}