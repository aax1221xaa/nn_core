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

NN_Layer::NN_Layer(const std::string& layer_name) :
	_layer_name(layer_name)
{
}

NN_Layer::~NN_Layer() {

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

NN_Input::NN_Input(const NN_Shape& input_size, int batch, const std::string& layer_name) :
	NN_Layer(layer_name),
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
		NN_Ptr& prev_ptr = p_prev_node.val();

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

NN_Manager::NN_Manager() :
	_node_counter(0)
{
}

NN_Manager::~NN_Manager() {
	try {
		for (size_t i = 0; i < _nodes.size(); ++i) {
			if (!_is_static[i]) delete _nodes[i];
		}
		for (NN_Layer* layer : _layers) delete layer;
		for (NN_Backward* backward : _backward) delete backward;
		for (NN_Optimizer* optimizer : _optimizer) delete optimizer;

		_is_static.clear();
		_input_layers.clear();
		_nodes.clear();
		_layers.clear();
		_backward.clear();
		_optimizer.clear();
	}
	catch (NN_Exception& e) {
		e.put();
	}
}

NN_Stream& NN_Manager::get_streams() {
	return _stream;
}

std::vector<NN_Link*>& NN_Manager::get_nodes() {
	return _nodes;
}

std::vector<NN_Layer*>& NN_Manager::get_layers() {
	return _layers;
}

const std::vector<NN_Input*>& NN_Manager::get_input_layers() {
	return _input_layers;
}

std::vector<NN_Backward*>& NN_Manager::get_backward() {
	return _backward;
}

NN_List<GpuTensor<nn_type>>& NN_Manager::get_weights() {
	return _weights;
}

const NN_Stream& NN_Manager::get_streams() const {
	return _stream;
}

const std::vector<NN_Link*>& NN_Manager::get_nodes() const {
	return _nodes;
}

const std::vector<NN_Layer*>& NN_Manager::get_layers() const {
	return _layers;
}

const std::vector<NN_Input*>& NN_Manager::get_input_layers() const {
	return _input_layers;
}

const std::vector<NN_Backward*>& NN_Manager::get_backward() const {
	return _backward;
}

const NN_List<GpuTensor<nn_type>>& NN_Manager::get_weights() const {
	return _weights;
}

void NN_Manager::set_optimizer(NN_Optimizer* optimizer) {
	_optimizer.push_back(optimizer);
}

void NN_Manager::set_nodes(NN_Link* node) {
	node->set_index(_node_counter++);
	_nodes.push_back(node);
	_is_static.push_back(false);
}

void NN_Manager::set_layers(NN_Layer* layer) {
	_layers.push_back(layer);
}

void NN_Manager::set_static_node(NN_Link* const node) {
	node->set_index(_node_counter++);
	_nodes.push_back(node);
	_is_static.push_back(true);
}

void NN_Manager::set_backward(NN_Backward* backward) {
	_backward.push_back(backward);
}

void NN_Manager::set_reserved_weights() {
	_weights.reserve(_nodes.size());
}

void NN_Manager::set_reserved_shapes() {
	_out_shapes.reserve(_nodes.size());
}

void NN_Manager::set_reserved_outputs() {
	_outputs.reserve(_nodes.size());
}

void NN_Manager::set_reserved_doutputs() {
	_doutputs.reserve(_nodes.size());
}

NN_List<NN_Shape>& NN_Manager::get_node_shape() {
	return _out_shapes;
}

NN_List<GpuTensor<nn_type>>& NN_Manager::get_node_output() {
	return _outputs;
}

NN_List<GpuTensor<nn_type>>& NN_Manager::get_node_doutput() {
	return _doutputs;
}

const NN_List<NN_Shape>& NN_Manager::get_node_shape() const {
	return _out_shapes;
}

const NN_List<GpuTensor<nn_type>>& NN_Manager::get_node_output() const {
	return _outputs;
}

const NN_List<GpuTensor<nn_type>>& NN_Manager::get_node_doutput() const {
	return _doutputs;
}

void NN_Manager::clear_weights() {
	_weights.clear();
}

void NN_Manager::clear_shapes() {
	_out_shapes.clear();
}

void NN_Manager::clear_outputs() {
	_outputs.clear();
}

void NN_Manager::clear_dinputs() {
	_doutputs.clear();
}

Layer_t NN_Manager::input(const NN_Shape& input_size, int batch, const std::string& layer_name) {
	NN_Input* layer = new NN_Input(input_size, batch, layer_name);
	NN_Link* node = new NN_Link;

	_input_layers.push_back(layer);
	node->set_layer(layer);

	set_nodes(node);
	_layers.push_back(layer);
	
	NN_List<NN_Shape> out_shape;

	layer->get_output_shape(NN_List<NN_Shape>(), out_shape);

	return NN_Ptr({ 0, node, out_shape.val() });
}


/**********************************************/
/*                                            */
/*                    misc					  */
/*                                            */
/**********************************************/

void set_random_uniform(GpuTensor<nn_type>& tensor, nn_type a, nn_type b) {
	std::random_device rd;
	cv::RNG rng(rd());

	cv::Mat tmp(tensor.get_shape().get_dims(), CV_32FC1);

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