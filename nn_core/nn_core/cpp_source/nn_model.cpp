#include "nn_model.h"


/**********************************************/
/*                                            */
/*                     Model                  */
/*                                            */
/**********************************************/

int Model::_stack = 0;

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
		/*    marking output to input    */
		std::vector<NN_Link*>& node_list = manager.get_nodes();
		std::vector<int> mask;

		find_path(inputs, outputs, mask);

		std::vector<int> m_mask(mask.size(), 0);

		for (NN_Link* node : node_list) {
			int i = 0;

			for (NN_Link* p_prev : node->get_prev_nodes()) {
				if (mask[p_prev->get_index()] == -1) ++i;
			}

			m_mask[node->get_index()] = i;
		}

		set_childs(inputs, outputs, m_mask);
	}
	catch (const Exception& e) {
		e.Put();
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
	child_model->set_backprop(NULL);
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

const std::vector<int>& Model::get_output_indice() {
	return _output_indice;
}

void Model::set_output_indice(const std::vector<int>& indice) {
	_output_indice = indice;
}

void Model::test(const std::vector<Tensor<nn_type>>& in_val, std::vector<Tensor<nn_type>>& out_val) {
	++_stack;

	std::vector<std::vector<Tensor<nn_type>>> outputs(_manager.get_nodes().size());

	for (NN_Link* p_node : _layers) {
		std::vector<Tensor<nn_type>> input_storage;
		std::vector<Tensor<nn_type>>& output_storage = outputs[p_node->get_index()];

		if (p_node->get_prev_nodes().size() > 0) {
			for (NN_Link* p_prev_node : p_node->get_prev_nodes()) {
				size_t i = 0;
				for (NN_Link* p_next_node : p_prev_node->get_next_nodes()) {
					if (p_node == p_next_node) break;
					else ++i;
				}
				input_storage.push_back(outputs[p_prev_node->get_index()][i]);
			}
		}
		else {
			size_t i = 0;
			for (NN_Link* p_input : _input_nodes) {
				if (p_input == p_node) break;
				else ++i;
			}
			input_storage.push_back(in_val[i]);
		}

		p_node->get_layer().test(input_storage, output_storage);
	}
	if (_output_indice.size() > 0) {
		for (size_t i = 0; i < _output_nodes.size(); ++i) {
			std::vector<Tensor<nn_type>>& p_output = outputs[_output_nodes[_output_indice[i]]->get_index()];

			out_val.insert(out_val.end(), p_output.begin(), p_output.end());
		}
	}
	else {
		for (size_t i = 0; i < _output_nodes.size(); ++i) {
			std::vector<Tensor<nn_type>>& p_output = outputs[_output_nodes[i]->get_index()];

			out_val.insert(out_val.end(), p_output.begin(), p_output.end());
		}
	}
	--_stack;
}

void Model::summary() {
	int i = 0;

	std::cout << '[' << _layer_name << ']' << std::endl;

	for (NN_Link* p_node : _layers) {
		std::cout << ++i << " : layer_name = " << p_node->get_layer()._layer_name << std::endl;
			//<< " output size: " << put_shape(p_node->) << std::endl;
	}
}