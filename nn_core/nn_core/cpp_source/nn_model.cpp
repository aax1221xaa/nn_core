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
	catch (const Exception& e) {
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