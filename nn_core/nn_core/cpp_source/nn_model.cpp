#include "nn_model.h"


/**********************************************/
/*                                            */
/*                     Model                  */
/*                                            */
/**********************************************/

int Model::_stack = 0;

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

	for (Layer_t& ptr : inputs) p_inputs.push_back(ptr.get_val()._node);
	for (Layer_t& ptr : outputs) p_outputs.push_back(ptr.get_val()._node);

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
				if (mask[node->get_index()] < 0) {
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
		NN_Link* p_input = input_node.get_val()._node;
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
		NN_Link* p_node = p_input.get_val()._node;

		_input_nodes.push_back(nodes[child_index[p_node->get_index()]]);
	}

	for (Layer_t& p_output : outputs) {
		NN_Link* p_node = p_output.get_val()._node;

		_output_nodes.push_back(nodes[child_index[p_node->get_index()]]);
	}
}

int Model::get_n_node_prev_for_next(const NN_Link* prev_node, const NN_Link* curr_node) {
	int n = -1;

	for (const NN_Link* p_next : prev_node->get_next_nodes()) {
		++n;

		if (p_next == curr_node) break;
	}

	return n;
}

int Model::get_n_input(const std::vector<NN_Link*>& input_node, const NN_Link* curr_node) {
	int n = -1;

	for (const NN_Link* p_input : input_node) {
		++n;

		if (p_input == curr_node) break;
	}

	return n;
}

const std::vector<int>& Model::get_output_indice() const {
	return _output_indice;
}

void Model::set_output_indice(const std::vector<int>& indice) {
	_output_indice = indice;
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

void Model::set_weight(const H5::Group& group, std::vector<GpuTensor<nn_type>>& g_tensor) {
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
			g_tensor[i] = tensor.transpose({ 3, 2, 0, 1 });
		}
		else {
			g_tensor[i] = tensor;
		}

		++i;
		delete[] dims;
		data_set.close();
	}
}

Model::Model(NN_Manager& manager, const char* model_name) :
	NN_Layer(model_name),
	_manager(manager)
{
}

Model::Model(NN_Manager& manager, Layer_t inputs, Layer_t outputs, const char* model_name) :
	NN_Layer(model_name),
	_manager(manager)
{
	try {
		std::vector<int> mask;

		find_path(inputs, outputs, mask);
		count_branch(mask);
		set_childs(inputs, outputs, mask);
	}
	catch (const NN_Exception& e) {
		e.put();
	}
}

Model::~Model() {

}

Layer_t Model::operator()(Layer_t prev_node) {
	int i = 0;

	for (Layer_t& p_prev_node : prev_node) {
		NN_Ptr& prev_ptr = p_prev_node.get_val();

		set_prev_node(prev_ptr._node);
		prev_ptr._node->set_next_link(this, i++);
	}

	_manager.set_static_node(this);

	Layer_t output_nodes;

	for (i = 0; i < _output_nodes.size(); ++i) output_nodes.push_back(NN_Ptr({ i, this }));

	return output_nodes;
}

NN_Link* Model::create_child() {
	std::vector<NN_Link*>& nodes = _manager.get_nodes();
	std::vector<int> child_index(nodes.size(), -1);

	Model* child_model = new Model(_manager, _layer_name);

	child_model->set_layer(child_model);
	child_model->trainable = trainable;
	child_model->set_output_indice(_output_indice);

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

	return child_model;
}

void Model::set_next_link(NN_Link* node, int index) {
	set_next_node(node);
	_output_indice.push_back(index);
}

void Model::get_output_shape(const std::vector<NN_Shape>& input_shape, std::vector<NN_Shape>& output_shape) {
	std::vector<std::vector<NN_Shape>>& nodes_shape = _manager.get_node_shape();

	for (NN_Link* node : _layers) {
		std::vector<NN_Shape>& m_output_shape = nodes_shape[node->get_index()];
		std::vector<NN_Shape> m_input_shape;

		if (node->get_prev_nodes().size() > 0) {
			for (NN_Link* p_prev_node : node->get_prev_nodes()) {
				int curr_n_out = get_n_node_prev_for_next(p_prev_node, node);
				m_input_shape.push_back(nodes_shape[p_prev_node->get_index()][curr_n_out]);
			}
		}
		else {
			size_t i = 0;
			for (NN_Link* p_input : _input_nodes) {
				if (p_input == node) break;
				else ++i;
			}
			m_input_shape.push_back(input_shape[i]);
		}

		node->get_layer().get_output_shape(m_input_shape, m_output_shape);
	}

	for (size_t i = 0; i < _output_nodes.size(); ++i) {
		std::vector<NN_Shape>& out_shape = nodes_shape[_output_nodes[_output_indice[i]]->get_index()];
		output_shape.insert(output_shape.end(), out_shape.begin(), out_shape.end());
	}
}

void Model::build(const std::vector<NN_Shape>& input_shape) {
	const std::vector<std::vector<NN_Shape>>& shapes = _manager.get_node_shape();

	for (NN_Link* node: _layers) {
		std::vector<NN_Shape> m_input_shape;

		if (node->get_prev_nodes().size() > 0) {
			for (const NN_Link* p_prev : node->get_prev_nodes()) {
				int n_prev_out = get_n_node_prev_for_next(p_prev, node);
				int n_prev = p_prev->get_index();

				m_input_shape.push_back(shapes[n_prev][n_prev_out]);
			}
		}
		else {
			int n_input = get_n_input(_input_nodes, node);

			m_input_shape.push_back(input_shape[n_input]);
		}

		node->get_layer().build(m_input_shape);
	}
}

void Model::run_forward(NN_Stream& st, const std::vector<GpuTensor<nn_type>>& input, std::vector<GpuTensor<nn_type>>& output) {

}

void Model::run_backward(NN_Stream& st, const std::vector<GpuTensor<nn_type>>& d_output, std::vector<GpuTensor<nn_type>>& d_input) {

}

void Model::summary() {
	int i = 0;
	std::vector<std::vector<NN_Shape>>& nodes_shape = _manager.get_node_shape();

	std::cout << '[' << _layer_name << ']' << std::endl;
	
	_manager.set_reserved_shapes();
	
	for (NN_Link* node : _layers) {
		std::vector<NN_Shape>& m_output_shape = nodes_shape[node->get_index()];
		std::vector<NN_Shape> m_input_shape;

		if (node->get_prev_nodes().size() > 0) {
			for (NN_Link* p_prev_node : node->get_prev_nodes()) {
				int curr_n_out = get_n_node_prev_for_next(p_prev_node, node);
				m_input_shape.push_back(nodes_shape[p_prev_node->get_index()][curr_n_out]);
			}
		}

		node->get_layer().get_output_shape(m_input_shape, m_output_shape);
	}
	
	for (NN_Link* p_node : _layers) {
		std::cout << ++i << " : layer_name = " << p_node->get_layer()._layer_name << ", output shape = ";
		for (const NN_Shape& shape : nodes_shape[p_node->get_index()]) {
			std::cout << shape_to_str(shape) << ", ";
		}
		std::cout << std::endl;
	}

	_manager.clear_shapes();
}

void Model::load_weights(const std::string& path, bool skip_mismatch) {
	std::vector<std::vector<NN_Shape>>& nodes_shape = _manager.get_node_shape();

	_manager.set_reserved_shapes();

	for (NN_Link* node : _layers) {
		std::vector<NN_Shape>& m_output_shape = nodes_shape[node->get_index()];
		std::vector<NN_Shape> m_input_shape;

		if (node->get_prev_nodes().size() > 0) {
			for (NN_Link* p_prev_node : node->get_prev_nodes()) {
				int curr_n_out = get_n_node_prev_for_next(p_prev_node, node);
				m_input_shape.push_back(nodes_shape[p_prev_node->get_index()][curr_n_out]);
			}
		}

		node->get_layer().get_output_shape(m_input_shape, m_output_shape);
		node->get_layer().build(m_input_shape);
	}

	_manager.clear_shapes();

	H5::H5File fp(path, H5F_ACC_RDONLY);
	std::vector<std::string> layer_names = get_layer_names(fp);

	for (const std::string& name : layer_names) {
		for (NN_Link* p_link : _layers) {
			if (name == p_link->get_layer()._layer_name) {
				std::cout << "Layer_name: " << name << ' ';
				std::vector<GpuTensor<nn_type>> weight = p_link->get_layer().get_weight();

				H5::Group group = fp.openGroup('/' + name);
				set_weight(group, weight);

				std::cout << "Done." << std::endl;
				group.close();
			}
		}
	}

	fp.close();
}