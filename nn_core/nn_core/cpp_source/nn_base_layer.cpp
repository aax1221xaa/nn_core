#include "nn_base_layer.h"



/**********************************************/
/*                                            */
/*                  NN_Layer                  */
/*                                            */
/**********************************************/

NN_BackPropLayer::~NN_BackPropLayer() {

}

void NN_BackPropLayer::set_dio(
	std::vector<nn_shape>& in_shape,
	std::vector<GpuTensor<nn_type>>& d_outputs,
	std::vector<GpuTensor<nn_type>>& d_inputs
) {

}

void NN_BackPropLayer::run_backprop(
	std::vector<cudaStream_t>& s,
	std::vector<GpuTensor<nn_type>>& inputs,
	GpuTensor<nn_type>& outputs,
	GpuTensor<nn_type>& d_output,
	std::vector<GpuTensor<nn_type>>& d_input
) {

}

NN_Layer::NN_Layer(const char* layer_name) :
	_layer_name(layer_name)
{
}

NN_Layer::~NN_Layer() {

}

void NN_Layer::set_io(std::vector<GpuTensor<nn_type>>& input, nn_shape& out_shape, GpuTensor<nn_type>& output) {
	output.set(out_shape);
}

NN_BackPropLayer* NN_Layer::create_backprop(NN_Optimizer& optimizer) {
	return NULL;
}

std::vector<int> NN_Layer::digestion(const std::vector<int>& feed) {
	std::cout << _layer_name << " ";

	std::cout << '[';
	for (const int& val : feed)
		std::cout << val << ", ";
	std::cout << ']' << std::endl;

	int sum = 0; // = cv::Mat::zeros(cv::Size({ 1 }), CV_32SC1);

	for (const int& val : feed) sum += val;


	return { sum };
}


/**********************************************/
/*                                            */
/*                  NN_Input                  */
/*                                            */
/**********************************************/

NN_Input::NN_Input(const nn_shape& input_size, int batch, const char* _layer_name) :
	NN_Layer(_layer_name),
	_shape(input_size)
{
	try {
		_shape.push_back(batch);

		for (const List<int>& n : _shape) {
			if (*(n._ptr->data) < -1 || *(n._ptr->data) == 0) {
				ErrorExcept(
					"[NN_Input::NN_Input] can't create input layer by dimension(%s).",
					put_shape(input_size)
				);
			}
		}

	}
	catch (const Exception& e) {
		e.Put();
	}
}

NN_Input::~NN_Input() {

}

nn_shape NN_Input::calculate_output_size(nn_shape& input_shape) {
	return _shape ;
}

void NN_Input::set_io(std::vector<GpuTensor<nn_type>>& input, nn_shape& out_shape, GpuTensor<nn_type>& output) {

}

void NN_Input::run_forward(std::vector<cudaStream_t>& stream, std::vector<GpuTensor<nn_type>>& input, GpuTensor<nn_type>& output) {
	//check_cuda(cudaMemcpy(output._data, input[0]->_data, output._elem_size * output._len, cudaMemcpyDeviceToDevice));
}

NN_BackPropLayer* NN_Input::create_backprop(NN_Optimizer& optimizer) {
	return NULL;
}


/**********************************************/
/*                                            */
/*                   NN_Link                  */
/*                                            */
/**********************************************/

NN_Link::NN_Link() :
	_layer(NULL),
	_backprop(NULL),
	_index(0),
	trainable(true)
{
}

NN_Link::~NN_Link() {

}

NN_Link* NN_Link::create_child() {
	NN_Link* child_node = new NN_Link;

	child_node->_layer = _layer;
	child_node->_backprop = _backprop;
	child_node->trainable = trainable;

	return child_node;
}

std::vector<NN_Link*>& NN_Link::get_prev_nodes() {
	return _prev;
}

std::vector<NN_Link*>& NN_Link::get_next_nodes() {
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

NN_BackPropLayer& NN_Link::get_backprop() {
	return *_backprop;
}

void NN_Link::set_layer(NN_Layer* layer) {
	_layer = layer;
}

void NN_Link::set_backprop(NN_BackPropLayer* backprop) {
	_backprop = backprop;
}

int NN_Link::get_index() {
	return _index;
}

void NN_Link::set_index(int index) {
	_index = index;
}

Layer_t NN_Link::operator()(Layer_t prev_node) {
	for (Layer_t& p_prev_node : prev_node) {
		NN_Link::NN_Ptr& prev_ptr = p_prev_node.get_val();

		set_prev_node(prev_ptr._node);
		prev_ptr._node->set_next_link(this, prev_ptr._index);
	}

	return NN_Link::NN_Ptr({ 0, this });
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
	try {
		for (int i = 0; i < STREAMS; ++i) check_cuda(cudaStreamCreate(&_streams[i]));
	}
	catch (const Exception& e) {
		for (int i = 0; i < STREAMS; ++i) {
			cudaStreamDestroy(_streams[i]);
			_streams[i] = NULL;
		}
		e.Put();
	}
}

NN_Manager::~NN_Manager() {
	try {
		for (size_t i = 0; i < _nodes.size(); ++i) {
			if (!_is_static[i]) delete _nodes[i];
		}
		for (NN_Layer* layer : _layers) delete layer;
		for (int i = 0; i < STREAMS; ++i) check_cuda(cudaStreamDestroy(_streams[i]));

		_is_static.clear();
		_nodes.clear();
		_layers.clear();
	}
	catch (Exception& e) {
		e.Put();
	}
}

cudaStream_t* NN_Manager::get_streams() {
	return _streams;
}

std::vector<NN_Link*>& NN_Manager::get_nodes() {
	return _nodes;
}

std::vector<NN_Layer*>& NN_Manager::get_layers() {
	return _layers;
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

Layer_t NN_Manager::input(const nn_shape& input_size, int batch, const char* layer_name) {
	NN_Layer* layer = new NN_Input(input_size, batch, layer_name);
	NN_Link* node = new NN_Link;

	node->set_layer(layer);

	set_nodes(node);
	_layers.push_back(layer);

	return NN_Link::NN_Ptr({ 0, node });
}