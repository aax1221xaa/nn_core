#include "nn_model.h"


/**********************************************/
/*                                            */
/*                     Model                  */
/*                                            */
/**********************************************/

/***************************** static method start ***************************/

void Model::put_error(int n_error) {
	switch (n_error)
	{
	case 0:
		ErrorExcept(
			"[Model::put_error()] Please set layers."
		);
	case 1:
		ErrorExcept(
			"[Model::put_error()] No calculate output shapes."
		);
	case 2:
		ErrorExcept(
			"[Model::put_error()] No create weights of layers."
		);
	case 3:
		ErrorExcept(
			"[Model::put_error()] Please set outputs"
		);
	case 4:
		ErrorExcept(
			"[Model::put_error()] Please set backprops."
		);
	case 5:
		{
			std::string str;
			NN_List<NN_Shape>& in_shapes = _manager.get_node_shape();

			for (NN_Link* p_input : _input_nodes) {
				str += shape_to_str(in_shapes[p_input->get_index()].val());
			}

			ErrorExcept(
				"[Model::put_error()] This model can't accept input shapes. %s",
				str.c_str()
			);
		}
	default:
		break;
	}
}

int Model::get_n_input(const std::vector<NN_Link*>& input_node, const NN_Link* curr_node) {
	int n = -1;

	for (const NN_Link* p_input : input_node) {
		++n;

		if (p_input == curr_node) break;
	}

	return n;
}

std::vector<std::string> Model::get_layer_names(const H5::H5File& fp) {
	H5::Group main_group = fp.openGroup("/");
	H5::Attribute attr = main_group.openAttribute("layer_names");
	H5::DataType dtype = attr.getDataType();
	H5::DataSpace space = attr.getSpace();

	hsize_t layer_amounts = 0;

	space.getSimpleExtentDims(&layer_amounts);

	char** p_layer_names = new char*[layer_amounts];
	std::vector<std::string> layer_names(layer_amounts);

	attr.read(dtype, p_layer_names);

	for (hsize_t i = 0; i < layer_amounts; ++i) layer_names[i] = p_layer_names[i];

	delete[] p_layer_names;
	attr.close();

	return layer_names;
}

void Model::parsing_weight(const H5::Group& group, NN_List<GpuTensor<nn_type>>& g_tensor) {
	if (g_tensor.size() == 0) return;

	H5::Attribute attr = group.openAttribute("weight_names");
	H5::DataType dtype = attr.getDataType();
	H5::DataSpace space = attr.getSpace();

	hsize_t n_weights = 0;

	space.getSimpleExtentDims(&n_weights);

	char** p_weight_names = new char*[n_weights];
	std::vector<std::string> weight_names(n_weights);

	attr.read(dtype, p_weight_names);

	for (hsize_t i = 0; i < n_weights; ++i) weight_names[i] = p_weight_names[i];

	delete[] p_weight_names;
	attr.close();

	int i = 0;
	for (const std::string& name : weight_names) {
		H5::DataSet data_set = group.openDataSet(name);
		H5::DataSpace d_space = data_set.getSpace();
		H5::DataType data_type = data_set.getDataType();

		int n_dims = d_space.getSimpleExtentNdims();
		hsize_t* dims = new hsize_t[n_dims];

		d_space.getSimpleExtentDims(dims);

		Tensor<nn_type> tensor(dims, n_dims);

		data_set.read(tensor.get_ptr(), data_type);

		if (n_dims == 4) {
			g_tensor[i].val() = tensor.transpose({ 3, 2, 0, 1 });
		}
		else {
			g_tensor[i].val() = tensor;
		}

		++i;
		delete[] dims;
		data_set.close();
	}
}

NN_List<GpuTensor<nn_type>> Model::zero_clone(const NN_List<GpuTensor<nn_type>>& tensor) {
	NN_List<GpuTensor<nn_type>> tmp;

	if (tensor.is_scalar()) {
		const NN_Shape shape = tensor.val().get_shape();

		tmp.append(GpuTensor<nn_type>::zeros(shape));
	}
	else {
		for (const NN_List<GpuTensor<nn_type>>& m_tensor : tensor) {
			tmp.append(zero_clone(tensor));
		}
	}

	return tmp;
}

/***************************** static method end ***************************/

/***************************** private method start ***************************/

const NN_Input* Model::get_input_layer(NN_Link* link) {
	const NN_Input* input_layer = NULL;

	for (const NN_Input* p_input : _manager.get_input_layers()) {
		if (&link->get_layer() == p_input) {
			input_layer = p_input;
			break;
		}
	}

	return input_layer;
}

void Model::find_path(Layer_t& inputs, Layer_t& outputs, std::vector<int>& find_mask) {
	std::vector<NN_Link*> tmp;
	std::vector<NN_Link*> p_inputs;
	std::vector<NN_Link*> p_outputs;

	for (Layer_t& ptr : inputs) p_inputs.push_back(ptr.val()._node);
	for (Layer_t& ptr : outputs) p_outputs.push_back(ptr.val()._node);

	find_mask.resize(_manager.get_nodes().size(), 0);
	
	for (NN_Link* p_input : p_inputs) {
		for (NN_Link* p_output : p_outputs) {
			bool is_find = false;
			std::vector<int> mask(find_mask.size(), 0);

			tmp.push_back(p_output);

			while (!tmp.empty()) {
				NN_Link* node = tmp.front();

				mask[node->get_index()] = 1;
				
				for (NN_Link* p_prev : node->get_prev_nodes()) {
					if (p_prev != p_input) tmp.push_back(p_prev);
					else {
						mask[p_prev->get_index()] = 1;
						is_find = true;
					}
				}

				tmp.erase(tmp.begin());
			}

			if (is_find) {
				tmp.push_back(p_input);

				while (!tmp.empty()) {
					NN_Link* node = tmp.front();

					mask[node->get_index()] = -1;

					for (NN_Link* p_next : node->get_next_nodes()) {
						if (mask[p_next->get_index()] != 0) tmp.push_back(p_next);
					}

					tmp.erase(tmp.begin());
				}

				for (size_t i = 0; i < mask.size(); ++i) {
					if (find_mask[i] == 0 && mask[i] < 0) find_mask[i] = -1;
				}
			}

			tmp.clear();
		}
	}
}

void Model::count_branch(std::vector<int>& mask) {
	std::vector<NN_Link*>& nodes = _manager.get_nodes();
	std::vector<int> counter(nodes.size(), 0);

	for (const NN_Link* node : nodes) {
		if (mask[node->get_index()] < 0) {
			for (const NN_Link* prev_node : node->get_prev_nodes()) {
				if (mask[prev_node->get_index()] < 0) {
					++counter[node->get_index()];
				}
			}
		}
	}

	mask = counter;
}

void Model::set_childs(Layer_t& inputs, Layer_t& outputs, std::vector<int>& mask) {
	std::vector<NN_Link*>& nodes = _manager.get_nodes();
	std::vector<int> child_index(nodes.size(), -1);
	std::vector<int> m_mask = mask;
	std::vector<NN_Link*> parant_nodes;

	/*    create input to output layers    */
	for (Layer_t& input_node : inputs) {
		NN_Link* p_input = input_node.val()._node;
		std::vector<NN_Link*> tmp;

		tmp.push_back(p_input);

		while (!tmp.empty()) {
			NN_Link* p_current = tmp.front();
			NN_Link* p_child = p_current->create_child();

			_manager.set_nodes(p_child);
			_layers.push_back(p_child);
			parant_nodes.push_back(p_current);

			child_index[p_current->get_index()] = p_child->get_index();

			for (NN_Link* p_next : p_current->get_next_nodes()) {
				int& p_mask = mask[p_next->get_index()];

				if (p_mask > 0) {
					if (p_mask == 1) tmp.push_back(p_next);

					--p_mask;
				}
			}

			tmp.erase(tmp.begin());
		}
	}

	for (NN_Link* p_node : parant_nodes) {
		NN_Link* p_child = nodes[child_index[p_node->get_index()]];
		
		for (NN_Link* p_next : p_node->get_next_nodes()) {
			if (m_mask[p_next->get_index()] > 0) {
				NN_Link* p_next_child = nodes[child_index[p_next->get_index()]];

				p_child->set_next_node(p_next_child);
				p_next_child->set_prev_node(p_child);
			}
		}
	}

	for (Layer_t& p_input : inputs) {
		NN_Link* p_node = p_input.val()._node;

		_input_nodes.push_back(nodes[child_index[p_node->get_index()]]);
	}

	for (Layer_t& p_output : outputs) {
		NN_Link* p_node = p_output.val()._node;

		_output_nodes.push_back(nodes[child_index[p_node->get_index()]]);
	}
}

void Model::set_weights() {
	NN_List<NN_Shape> dummy_shape;
	NN_List<GpuTensor<nn_type>> dummy_tensor;

	build(dummy_shape, dummy_tensor);
}

void Model::set_shapes() {
	if (!(_status & 0x01)) put_error(_status);
	else if (!(_status & 0x02)) _manager.set_reserved_shapes();

	NN_List<NN_Shape>& nodes_shape = _manager.get_node_shape();

	for (NN_Link* node : _layers) {
		NN_List<NN_Shape>& m_output_shape = nodes_shape[node->get_index()];
		NN_List<NN_Shape> m_input_shape;

		if (node->get_prev_nodes().size() > 0) {
			for (NN_Link* p_prev_node : node->get_prev_nodes()) {
				int curr_n_out = p_prev_node->get_out_port(node);
				m_input_shape.append(nodes_shape[p_prev_node->get_index()][curr_n_out]);
			}
		}

		node->get_layer().get_output_shape(m_input_shape, m_output_shape);
	}

	_status |= 0x02;
}

/***************************** private method end ***************************/

/***************************** public create & destroy method start ***************************/

Model::Model(NN_Manager& manager, const std::string& model_name) :
	_status(0),
	NN_Layer(model_name),
	_manager(manager)
{
}

Model::Model(NN_Manager& manager, Layer_t inputs, Layer_t outputs, const char* model_name) :
	_status(0),
	NN_Layer(model_name),
	_manager(manager)
{
	try {
		std::vector<int> mask;

		find_path(inputs, outputs, mask);
		count_branch(mask);
		set_childs(inputs, outputs, mask);
		
		_status = 1;
		set_shapes();
		set_weights();
	}
	catch (const NN_Exception& e) {
		e.put();
	}
}

Model::~Model() {

}

/***************************** public create & destroy method end ***************************/

/***************************** Inheritanced NN_Link method start ***************************/

NN_Link* Model::create_child() {
	if (!(_status & 0x01)) put_error(0);

	std::vector<NN_Link*>& nodes = _manager.get_nodes();
	std::vector<int> child_index(nodes.size(), -1);

	Model* child_model = new Model(_manager, _layer_name);

	child_model->set_layer(child_model);
	child_model->trainable = trainable;
	child_model->_out_indices = _out_indices;

	for (NN_Link* p_node : _layers) {
		NN_Link* child_node = p_node->create_child();

		_manager.set_nodes(child_node);
		child_index[p_node->get_index()] = child_node->get_index();

		child_model->_layers.push_back(child_node);
	}

	for (NN_Link* p_node : _layers) {
		NN_Link* p_child = nodes[child_index[p_node->get_index()]];

		for (NN_Link* p_next_node : p_node->get_next_nodes()) {
			NN_Link* p_next_child = nodes[child_index[p_next_node->get_index()]];
			
			p_child->set_next_node(p_next_child);
			p_next_child->set_prev_node(p_child);
		}
	}

	for (NN_Link* p_input : _input_nodes) {
		child_model->_input_nodes.push_back(nodes[child_index[p_input->get_index()]]);
	}
	for (NN_Link* p_output : _output_nodes) {
		child_model->_output_nodes.push_back(nodes[child_index[p_output->get_index()]]);
	}

	child_model->_status |= 0x01;

	return child_model;
}

/***************************** Inheritanced NN_Link method end ***************************/

/***************************** Inheritanced NN_Layer method start ***************************/

void Model::get_output_shape(const NN_List<NN_Shape>& input_shape, NN_List<NN_Shape>& output_shape) {
	if (!(_status & 0x01)) put_error(0);

	NN_List<NN_Shape>& nodes_shape = _manager.get_node_shape();

	for (NN_Link* node : _layers) {
		NN_List<NN_Shape>& m_output_shape = nodes_shape[node->get_index()];
		NN_List<NN_Shape> m_input_shape;

		if (node->get_prev_nodes().size() > 0) {
			for (NN_Link* p_prev_node : node->get_prev_nodes()) {
				int curr_n_out = p_prev_node->get_out_port(node);
				m_input_shape.append(nodes_shape[p_prev_node->get_index()][curr_n_out]);
			}
		}
		else {
			size_t i = 0;
			for (NN_Link* p_input : _input_nodes) {
				if (p_input == node) break;
				else ++i;
			}
			m_input_shape.append(input_shape[i]);
		}

		node->get_layer().get_output_shape(m_input_shape, m_output_shape);
	}

	if (_out_indices.size() > 0) {
		for (size_t i = 0; i < _output_nodes.size(); ++i) {
			NN_List<NN_Shape>& out_shape = nodes_shape[_output_nodes[_out_indices[i]]->get_index()];
			output_shape.append(out_shape);
		}
	}
	else {
		for (size_t i = 0; i < _output_nodes.size(); ++i) {
			NN_List<NN_Shape>& out_shape = nodes_shape[_output_nodes[i]->get_index()];
			output_shape.append(out_shape);
		}
	}

	_status |= 0x02;
}

void Model::build(const NN_List<NN_Shape>& input_shape, NN_List<GpuTensor<nn_type>>& weights) {
	if (!(_status & 0x01)) put_error(_status);
	else if (!(_status & 0x02)) put_error(_status);
	else if (!(_status & 0x04)) {
		_manager.set_reserved_weights();

		NN_List<NN_Shape>& nodes_shape = _manager.get_node_shape();
		NN_List<GpuTensor<nn_type>>& weights = _manager.get_weights();

		for (NN_Link* node : _layers) {
			NN_List<NN_Shape> m_input_shape;

			if (node->get_prev_nodes().size() > 0) {
				for (NN_Link* p_prev_node : node->get_prev_nodes()) {
					int curr_n_out = p_prev_node->get_out_port(node);
					m_input_shape.append(nodes_shape[p_prev_node->get_index()][curr_n_out]);
				}
			}
			node->get_layer().build(m_input_shape, weights);
		}
		_status |= 0x04;
	}
}

void Model::run(NN_Stream& st, const NN_List<GpuTensor<nn_type>>& input, NN_List<GpuTensor<nn_type>>& output) {
	NN_List<GpuTensor<nn_type>>& nodes_output = _manager.get_node_output();

	for (NN_Link* p_node : _layers) {
		const int n_node = p_node->get_index();
		NN_List<GpuTensor<nn_type>>& m_output = nodes_output[n_node];
		NN_List<GpuTensor<nn_type>> m_input;

		if (p_node->get_prev_nodes().size() > 0) {
			for (NN_Link* p_prev_node : p_node->get_prev_nodes()) {
				const int n_prev_out = p_prev_node->get_out_port(p_node);
				const int prev_index = p_prev_node->get_index();

				m_input.append(nodes_output[prev_index][n_prev_out]);
			}
		}
		else {
			const int n_prev_out = get_n_input(_input_nodes, p_node);

			m_input.append(input[n_prev_out]);
		}

		p_node->get_layer().run(st, m_input, m_output);
	}
}

NN_Backward* Model::create_backward(std::vector<bool>& mask) {
	NN_List<GpuTensor<nn_type>>& weights = _manager.get_weights();

	for (NN_Link* node : _layers) {
		if (node->get_prev_nodes().size() > 0) {
			if (weights[node->get_index()].size() > 0) {
				mask[node->get_index()] = true;

				NN_Backward* backward = node->get_layer().create_backward(mask);
				NN_Optimizer* optimizer = node->get_backward()->create_optimizer(_optimizer);

				node->set_backward(backward);
				node->set_optimizer(optimizer);
				_manager.set_backward(backward);
				_manager.set_optimizer(optimizer);
			}
			else {
				for (NN_Link* prev_node : node->get_prev_nodes()) {
					if (mask[prev_node->get_index()]) {
						mask[node->get_index()] = true;

						NN_Backward* backward = node->get_layer().create_backward(mask);

						node->set_backward(backward);
						_manager.set_backward(backward);
					}
				}
			}
		}
		else {
			const int n_input = get_n_input(_input_nodes, node);

			if (mask[_prev[n_input]->get_index()]) {
				mask[node->get_index()] = true;

				NN_Backward* backward = node->get_layer().create_backward(mask);

				node->set_backward(backward);
				_manager.set_backward(backward);
			}
		}
	}

	return new dModel(*this);
}

NN_List<GpuTensor<nn_type>> Model::get_weight() {
	return GpuTensor<nn_type>();
}

void Model::set_output(const NN_List<NN_Shape>& output_shape, NN_List<GpuTensor<nn_type>>& input, NN_List<GpuTensor<nn_type>>& output) {
	if ((_status & 0x07) != 0x07) put_error(_status);

	NN_List<GpuTensor<nn_type>>& nodes_output = _manager.get_node_output();
	NN_List<NN_Shape>& node_shape = _manager.get_node_shape();

	for (NN_Link* node : _layers) {
		const int n_node = node->get_index();
		NN_List<GpuTensor<nn_type>>& m_output = nodes_output[n_node];
		NN_List<GpuTensor<nn_type>> m_input;

		if (node->get_prev_nodes().size() > 0) {
			for (NN_Link* p_prev_node : node->get_prev_nodes()) {
				const int n_prev_out = p_prev_node->get_out_port(node);
				const int prev_index = p_prev_node->get_index();

				m_input.append(nodes_output[prev_index][n_prev_out]);
			}
		}
		else {
			const int n_input = get_n_input(_input_nodes, node);

			m_input.append(input[n_input]);
		}

		node->get_layer().set_output(node_shape[n_node], m_input, m_output);
	}

	for (int& n : _out_indices) output.append(nodes_output[_output_nodes[n]->get_index()]);

	_status &= ~0x0100;
	_status |= 0x08;
}

/***************************** Inheritanced NN_Layer method end ***************************/

/***************************** public ex method start ***************************/

Layer_t Model::operator()(Layer_t prev_node) {
	if (!(_status & 0x01)) put_error(0);

	NN_List<NN_Shape> in_shapes;
	NN_List<NN_Shape> out_shapes;

	for (Layer_t& p_prev_node : prev_node) {
		NN_Ptr& prev_ptr = p_prev_node.val();

		set_prev_node(prev_ptr._node);
		prev_ptr._node->set_next_link(this, prev_ptr._n_port);
		in_shapes.append(prev_ptr._shape);
	}

	if (!(_status & 0x02)) _manager.set_reserved_shapes();

	get_output_shape(in_shapes, out_shapes);
	_manager.set_static_node(this);

	Layer_t output_nodes;
	int i = 0;
	for (NN_List<NN_Shape> out_shape : out_shapes) output_nodes.append(NN_Ptr({ i, this, out_shape.val() }));

	return output_nodes;
}

void Model::load_weights(const std::string& path, bool skip_mismatch) {
	if (!(_status & 0x01)) put_error(0);

	H5::H5File fp(path, H5F_ACC_RDONLY);
	std::vector<std::string> layer_names = get_layer_names(fp);

	for (const std::string& name : layer_names) {
		for (NN_Link* p_link : _layers) {
			if (name == p_link->get_layer()._layer_name) {
				std::cout << "Layer_name: " << name << ' ';
				NN_List<GpuTensor<nn_type>> weight = p_link->get_layer().get_weight();

				H5::Group group = fp.openGroup('/' + name);
				parsing_weight(group, weight);

				std::cout << "Done." << std::endl;
				group.close();
			}
		}
	}

	fp.close();
}

void Model::summary() {
	if (!(_status & 0x01)) put_error(_status);

	std::cout << '[' << _layer_name << ']' << std::endl;

	NN_List<NN_Shape>& nodes_shape = _manager.get_node_shape();
	
	int i = 0;
	for (NN_Link* p_node : _layers) {
		std::cout << ++i << " : layer_name = " << p_node->get_layer()._layer_name << ", output shape = ";
		std::cout << nodes_shape[p_node->get_index()];
	}
}

void Model::stand_by(const NN_Optimizer& optimizer, const std::initializer_list<NN_Loss>& loss) {
	for (const NN_Loss& p : loss) _losses.push_back(&p);

	NN_List<GpuTensor<nn_type>>& weights = _manager.get_weights();
	std::vector<bool> mask(_manager.get_nodes().size(), false);

	for (NN_Link* p_node : _layers) {
		if (weights[p_node->get_index()].size() > 0) {
			mask[p_node->get_index()] = true;

			NN_Backward* p_backward = p_node->get_layer().create_backward(mask);
			NN_Optimizer* p_optimizer = p_backward->create_optimizer(optimizer);
			
			_manager.set_backward(p_backward);
			_manager.set_optimizer(p_optimizer);
			p_node->set_backward(p_backward);
			p_node->set_optimizer(p_optimizer);
		}
		else {
			for (NN_Link* p_prev_node : p_node->get_prev_nodes()) {
				if (mask[p_prev_node->get_index()]) {
					mask[p_node->get_index()] = true;

					NN_Backward* p_backward = p_node->get_layer().create_backward(mask);

					_manager.set_backward(p_backward);
					p_node->set_backward(p_backward);

					break;
				}
			}
		}
	}

	_status |= 0x10;
}

/***************************** public ex method end ***************************/


/**********************************************/
/*                                            */
/*                    dModel                  */
/*                                            */
/**********************************************/

dModel::dModel(Model& model) :
	NN_Backward_t(model)
{

}

void dModel::run(
	NN_Stream& st,
	const NN_List<GpuTensor<nn_type>>& input,
	const NN_List<GpuTensor<nn_type>>& doutput,
	NN_List<GpuTensor<nn_type>>& dinput
) {

}