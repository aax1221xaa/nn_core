#include "nn_base.h"
#include <random>



/**********************************************/
/*                                            */
/*                  NN_Layer                  */
/*                                            */
/**********************************************/

NN_Layer::NN_Layer(const char* layer_name) :
	_layer_name(layer_name)
{
}

NN_Layer::~NN_Layer() {

}

void NN_Layer::get_output_shape(const std::vector<NN_Shape>& input_shape, std::vector<NN_Shape>& output_shape) {
	ErrorExcept(
		"[NN_Layer::get_output_shape] Make this function."
	);
}

void NN_Layer::build(const std::vector<NN_Shape>& input_shape) {
	ErrorExcept(
		"[NN_Layer::build] Make this function."
	);
}

void NN_Layer::run_forward(NN_Stream& st, const std::vector<GpuTensor<nn_type>>& input, std::vector<GpuTensor<nn_type>>& output) {
	ErrorExcept(
		"[NN_Layer::run_forward] Make this function."
	);
}

void NN_Layer::run_forward(const Tensor& src, GpuTensor& dst) {
	ErrorExcept(
		"[NN_Layer::run_forward] Make this function."
	);
}

void NN_Layer::run_backward(NN_Stream& st, const std::vector<GpuTensor>& d_output, std::vector<GpuTensor>& d_input) {
	ErrorExcept(
		"[NN_Layer::run_backward] Make this function."
	);
}


/**********************************************/
/*                                            */
/*                  NN_Input                  */
/*                                            */
/**********************************************/

NN_Input::NN_Input(const NN_Shape& input_size, int batch, const char* _layer_name) :
	NN_Layer(_layer_name),
	_shape(input_size)
{
	_shape.push_front(batch);
}

NN_Input::~NN_Input() {

}

void NN_Input::get_output_shape(const std::vector<NN_Shape>& input_shape, std::vector<NN_Shape>& output_shape) {
	if (input_shape.size() > 1) {
		ErrorExcept(
			"[NN_Input::get_output_shape] Input node can't take %ld tensor shapes.",
			input_shape.size()
		);
	}
	else if (input_shape.size() == 0) {
		output_shape.push_back(_shape);
	}
	else {
		const NN_Shape& in_shape = input_shape[0];

		if (in_shape.get_len() != _shape.get_len()) {
			ErrorExcept(
				"[NN_Input::get_output_shape] Input excpected dimensions are %s. but take dimensions are %s.",
				shape_to_str(_shape),
				shape_to_str(in_shape)
			);
		}

		nn_shape out_shape(_shape.size(), 0);
		
		if (!is_valid_shape(input_shape[0])) {
			ErrorExcept(
				"[NN_Input::get_output_shape] Input dimensions must be all grater than 0 but %s.",
				shape_to_str(input_shape[0])
			);
		}

		for (int i = 0; i < _shape.size(); ++i) {
			if (_shape[i] < 0) out_shape[i] = input_shape[0][i];
			else out_shape[i] = _shape[i];
		}

		output_shape.push_back(out_shape);
	}
}

void NN_Input::build(const std::vector<nn_shape>& input_shape) {
	std::cout << "build: " << _layer_name << std::endl;
}

void NN_Input::run_forward(NN_Stream& st, const std::vector<GpuTensor>& input, std::vector<GpuTensor>& output) {
	output = input;
}

void NN_Input::run_forward(const Tensor& src, GpuTensor& dst) {
	dst.upload(src);
	
}


/**********************************************/
/*                                            */
/*                   NN_Link                  */
/*                                            */
/**********************************************/

NN_Link::NN_Link() :
	_layer(NULL),
	_index(0),
	trainable(true)
{
}

NN_Link::~NN_Link() {

}

NN_Link* NN_Link::create_child() {
	NN_Link* child_node = new NN_Link;

	child_node->_layer = _layer;
	child_node->trainable = trainable;

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

void NN_Link::set_layer(NN_Layer* layer) {
	_layer = layer;
}

const int& NN_Link::get_index() const {
	return _index;
}

void NN_Link::set_index(int index) {
	_index = index;
}

Layer_t NN_Link::operator()(Layer_t prev_node) {
	for (Layer_t& p_prev_node : prev_node) {
		NN_Ptr& prev_ptr = p_prev_node.get_val();

		set_prev_node(prev_ptr._node);
		prev_ptr._node->set_next_link(this, prev_ptr._index);
	}

	return NN_Ptr({ 0, this });
}

void NN_Link::set_next_link(NN_Link* node, int index) {
	_next.push_back(node);
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

		_is_static.clear();
		_input_layers.clear();
		_nodes.clear();
		_layers.clear();
	}
	catch (Exception& e) {
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

void NN_Manager::set_nodes(NN_Link* node) {
	node->set_index(_node_counter++);
	_nodes.push_back(node);
	_is_static.push_back(false);
}

void NN_Manager::set_static_node(NN_Link* const node) {
	node->set_index(_node_counter++);
	_nodes.push_back(node);
	_is_static.push_back(true);
}

void NN_Manager::set_reserved_shapes() {
	_out_shapes.resize(_nodes.size());
}

void NN_Manager::set_reserved_outputs() {
	_outputs.resize(_nodes.size());
}

std::vector<NN_Shape>& NN_Manager::get_node_shape(int index) {
	return _out_shapes[index];
}

std::vector<GpuTensor<nn_type>>& NN_Manager::get_node_output(int index) {
	return _outputs[index];
}

void NN_Manager::clear_shapes() {
	_out_shapes.clear();
}

void NN_Manager::clear_outputs() {
	_outputs.clear();
}

Layer_t NN_Manager::input(const NN_Shape& input_size, int batch, const char* layer_name) {
	NN_Input* layer = new NN_Input(input_size, batch, layer_name);
	NN_Link* node = new NN_Link;

	_input_layers.push_back(layer);
	node->set_layer(layer);

	set_nodes(node);
	_layers.push_back(layer);

	return NN_Ptr({ 0, node });
}


/**********************************************/
/*                                            */
/*                    misc					  */
/*                                            */
/**********************************************/

void set_random_uniform(GpuTensor<nn_type>& tensor, nn_type a, nn_type b) {
	std::random_device rd;
	cv::RNG rng(rd());

	cv::Mat tmp(tensor.get_shape().get_vector(), CV_32FC1);

	rng.fill(tmp, cv::RNG::UNIFORM, a, b);

	check_cuda(
		cudaMemcpy(
			tensor.get_ptr(),
			tmp.ptr<nn_type>(0),
			sizeof(nn_type) * tensor.get_shape().total_size(),
			cudaMemcpyHostToDevice
		)
	);
}