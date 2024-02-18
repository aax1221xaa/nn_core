#include "nn_model.h"


/**********************************************/
/*                                            */
/*                     Model                  */
/*                                            */
/**********************************************/

Model::Model(const char* model_name) :
	NN_Layer(model_name)
{
}

Model::Model(Layer_t inputs, Layer_t outputs, const char* model_name) :
	NN_Layer(model_name)
{
	try {
		if (!NN_Manager::condition) {
			ErrorExcept(
				"[Model::Model] can't create model."
			);
		}

		NN_Manager::clear_mark();

		/*    marking output to input    */

		for (Layer_t& output_node : outputs) {
			NN_Link* p_output = output_node._val._p_node;

			for (Layer_t& input_node : inputs) {
				NN_Link* p_input = input_node._val._p_node;

				std::vector<NN_Link*> node_list;
				node_list.push_back(p_output);

				while (!node_list.empty()) {
					NN_Link* p_current = node_list.front();

					p_current->_mark = (uint)(p_current->_prev.size());

					for (NN_Link* p_prev : p_current->_prev) {
						if (p_prev != p_input) {
							if (!p_prev->_mark) node_list.push_back(p_prev);
						}
					}
					node_list.erase(node_list.begin());
				}
			}
		}

		/*    create input to output layers    */
		for (Layer_t& input_node : inputs) {
			NN_Link* p_input = input_node._val._p_node;
			std::vector<NN_Link*> node_list;

			node_list.push_back(p_input);

			while (!node_list.empty()) {
				NN_Link* p_current = node_list.front();
				NN_Link* p_child = p_current->create_child();

				_layers.push_back(p_child);
				NN_Manager::add_node(p_child);

				for (NN_Link* p_next : p_current->_next) {
					if (p_next->_mark > 0) {
						if (p_next->_mark == 1) node_list.push_back(p_next);
						
						--p_next->_mark;
					}
				}

				node_list.erase(node_list.begin());
			}
		}

		for (Layer_t& input_node : inputs) {
			NN_Link* p_current = input_node._val._p_node;

			if (p_current->_p_link == NULL) {
				ErrorExcept(
					"[Model::Model()] can't create model."
				);
			}
			_input_nodes.push_back(p_current->_p_link);
		}
		for (Layer_t& output_node : outputs) {
			NN_Link* p_current = output_node._val._p_node;

			if (p_current->_p_link == NULL) {
				ErrorExcept(
					"[Model::Model()] can't create model."
				);
			}
			_output_nodes.push_back(p_current->_p_link);
		}

		/*   link child node   */
		for (NN_Link* p_child : _layers) {
			NN_Link* p_parant = p_child->_p_link;

			for (NN_Link* p_next_paranet : p_parant->_next) {
				NN_Link* p_next_child = p_next_paranet->_p_link;
				
				p_child->_next.push_back(p_next_child);
				p_next_child->_prev.push_back(p_child);
			}
		}
	}
	catch (const Exception& e) {
		NN_Manager::condition = false;
		e.Put();
	}
}

Model::~Model() {

}

NN_Link* Model::create_child() {
	/*
	NN_Link* child_node = new NN_Link;

	child_node->_parent = this;
	child_node->is_selected = true;
	child_node->_forward = _forward;

	_child.push_back(child_node);

	return child_node;
	*/

	Model* child_model = new Model(_layer_name);

	_p_link = child_model;

	child_model->_p_link = this;
	child_model->_forward = child_model;
	child_model->_backward = NULL;
	child_model->trainable = trainable;
	child_model->_out_indice = _out_indice;

	for (NN_Link* p_node : _layers) {
		NN_Link* child_node = p_node->create_child();
		child_model->_layers.push_back(child_node);

		NN_Manager::add_node(child_node);
	}
	for (NN_Link* p_input : _input_nodes) {
		child_model->_input_nodes.push_back(p_input->_p_link);
	}
	for (NN_Link* p_output : _output_nodes) {
		child_model->_output_nodes.push_back(p_output->_p_link);
	}
	for (NN_Link* p_child : child_model->_layers) {
		NN_Link* p_parant = p_child->_p_link;

		for (NN_Link* p_next_paranet : p_parant->_next) {
			NN_Link* p_next_child = p_next_paranet->_p_link;

			p_child->_next.push_back(p_next_child);
			p_next_child->_prev.push_back(p_child);
		}
	}

	return child_model;
}

Layer_t Model::operator()(Layer_t prev_node) {
	for (Layer_t& p_prev_node : prev_node) {
		NN_LinkPtr& m_prev = p_prev_node._val;

		m_prev._p_node->set_link(this, m_prev._n_node);
		_prev.push_back(m_prev._p_node);
		//_input_nodes[i]->_input.push_back(&m_prev_node->get_output(n_prev_node));
		//_input_nodes[i]->_in_shape.push_back(&m_prev_node->get_out_shape(n_prev_node));
		//m_prev_node->get_d_output(n_prev_node).push_back(&_input_nodes[i]->_d_input);
	}

	Layer_t output_nodes;

	for (int i = 0; i < _output_nodes.size(); ++i) output_nodes.push_back(NN_LinkPtr({ i, this }));

	return output_nodes;
}

void Model::set_link(NN_Link* node, int index) {
	_next.push_back(node);
	_out_indice.push_back(index);
}

void Model::calculate_output_size(std::vector<nn_shape>& input_shape, nn_shape& out_shape) {

}

void Model::build(std::vector<nn_shape>& input_shape) {

}

void Model::set_io(std::vector<GpuTensor<nn_type>>& input, nn_shape& out_shape, GpuTensor<nn_type>& output) {

}

void Model::run_forward(std::vector<cudaStream_t>& stream, std::vector<GpuTensor<nn_type>>& input, GpuTensor<nn_type>& output) {

}

NN_BackPropLayer* Model::create_backprop(NN_Optimizer& optimizer) {
	return NULL;
}

void Model::summary() {
	int i = 0;

	std::cout << '[' << _layer_name << ']' << std::endl;

	for (NN_Link* p_node : _layers) {
		std::cout << ++i << " : layer_name = " << p_node->_forward->_layer_name << std::endl;
			//<< " output size: " << put_shape(p_node->) << std::endl;
	}
}