#include "nn_model.h"


#ifdef FIX_MODE

/**********************************************/
/*                                            */
/*                     Model                  */
/*                                            */
/**********************************************/

Model::Model(const char* model_name) :
	NN_Layer(model_name)
{
}

Model::Model(const Layer_t& inputs, const Layer_t& outputs, const char* model_name) :
	NN_Layer(model_name)
{
	try {
		if (!NN_Manager::condition) {
			ErrorExcept(
				"[Model::Model()] can't create model."
			);
		}

		/*    marking output to input    */
		NN_Manager::clear_select_mask();

		for (Layer_t output_node : outputs) {
			NN_Link* p_output = output_node[0].get()._link;

			for (Layer_t input_node : inputs) {
				NN_Link* p_input = input_node[0].get()._link;

				std::vector<NN_Link*> node_list;
				node_list.push_back(p_output);

				while (!node_list.empty()) {
					NN_Link* p_current = node_list.front();

					p_current->is_selected = true;

					for (NN_Link* p_prev : p_current->_prev) {
						if (p_prev != p_input) {
							if (!p_prev->is_selected) node_list.push_back(p_prev);
						}
						else {
							p_input->is_selected = true;
						}
					}
					node_list.erase(node_list.begin());
				}
			}
		}

		/*    create input to output layers    */
		for (Layer_t input_node : inputs) {
			NN_Link* p_input = input_node[0].get()._link;
			std::vector<NN_Link*> node_list;

			node_list.push_back(p_input);

			while (!node_list.empty()) {
				NN_Link* p_current = node_list.front();
				int prev_selects = 0;

				if (p_current->is_selected) {
					for (NN_Link* p_prev : p_current->_prev) {
						if (p_prev->is_selected) ++prev_selects;
					}

					if (prev_selects == 0) {
						p_current->is_selected = false;
						NN_Link* p_current_child = p_current->create_child();
						_forward_list.push_back(p_current_child);
						NN_Manager::add_node(p_current_child);

						for (NN_Link* p_next : p_current->_next) {
							if (p_next->is_selected) node_list.push_back(p_next);
						}
					}
				}
				node_list.erase(node_list.begin());
			}
		}

		for (Layer_t input_node : inputs) {
			NN_Link* p_child_input = NN_Link::get_child(input_node[0].get()._link);

			if (p_child_input == NULL) {
				ErrorExcept(
					"[Model::Model()] can't create model."
				);
			}
			_input_nodes.push_back(p_child_input);
		}
		for (Layer_t output_node : outputs) {
			NN_Link* p_child_output = NN_Link::get_child(output_node[0].get()._link);

			if (p_child_output == NULL) {
				ErrorExcept(
					"[Model::Model()] can't create model."
				);
			}
			_output_nodes.push_back(p_child_output);
		}

		/*   link child node   */
		for (NN_Link* p_child : _forward_list) {
			p_child->link_prev_child();
			//p_child->is_selected = false;
		}
		NN_Manager::clear_select_mask();

		/*   first calculate output size   */
		for (NN_Link* p_child : _forward_list) {
			 p_child->_forward->calculate_output_size(p_child->_in_shape, p_child->_out_shape);

			for (const int& n : p_child->_out_shape) {
				if (n < -1 || n == 0) {
					ErrorExcept(
						"[Model::Model()] can't create model. invalid %s layer's dimension %s.",
						p_child->_forward->_layer_name, dimension_to_str(p_child->_out_shape)
					);
				}
			}
		}

		/*    build    */
		for (NN_Link* p_child : _forward_list) {
			p_child->_forward->build(p_child->_in_shape);
		}
	}
	catch (const Exception& e) {
		NN_Manager::condition = false;
		e.Put();
	}
}

Model::~Model() {
	for (NN_Tensor<nn_type>* p : _d_outputs) delete p;
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

	child_model->_parent = this;
	child_model->is_selected = true;
	child_model->_forward = child_model;
	child_model->_output_indices = _output_indices;

	_child.push_back(child_model);

	for (NN_Link* p_node : _forward_list) {
		NN_Link* child_node = p_node->create_child();
		child_model->_forward_list.push_back(child_node);

		NN_Manager::add_node(child_node);
	}
	for (NN_Link* p_input : _input_nodes) {
		child_model->_input_nodes.push_back(NN_Link::get_child(p_input));
	}
	for (NN_Link* p_output : _output_nodes) {
		child_model->_output_nodes.push_back(NN_Link::get_child(p_output));
	}
	for (NN_Link* p_child : child_model->_forward_list) p_child->link_prev_child();

	return child_model;
}

Layer_t Model::operator()(const Layer_t& prev_node) {
	int i = 0;

	for (Layer_t p_prev_node : prev_node) {
		NN_Link* m_prev_node = p_prev_node[0].get()._link;
		int n_prev_node = p_prev_node[0].get()._output_index;

		m_prev_node->set_next_node(this, n_prev_node);
		_prev.push_back(m_prev_node);
		//_input_nodes[i]->_input.push_back(&m_prev_node->get_output(n_prev_node));
		//_input_nodes[i]->_in_shape.push_back(&m_prev_node->get_out_shape(n_prev_node));
		//m_prev_node->get_d_output(n_prev_node).push_back(&_input_nodes[i]->_d_input);

		++i;
	}

	Layer_t output_nodes;
	i = 0;

	for (NN_Link* p_output : _output_nodes) {
		output_nodes.push_back(Layer_Ptr<NN_Link> { this, i++});
	}

	return output_nodes;
}

int Model::get_node_index(NN_Link* next_node) {
	int n = 0;

	for (int i = 0; i < _next.size(); ++i) {
		if (next_node == _next[i]) {
			n = _output_indices[i];
			break;
		}
	}

	return n;
}

void Model::set_next_node(NN_Link* next_node, int node_index) {
	_next.push_back(next_node);
	_output_indices.push_back(node_index);
}

NN_Tensor<nn_type>& Model::get_output(int node_index) {
	return _output_nodes[node_index]->_output;
}

std::vector<NN_Tensor<nn_type>*>& Model::get_d_output(int node_index) {
	return _output_nodes[node_index]->_d_outputs;
}

nn_shape& Model::get_out_shape(int node_index) {
	return _output_nodes[node_index]->_out_shape;
}

void Model::link_prev_child() {
	int i = 0;
	for (NN_Link* p_prev : _parent->_prev) {
		NN_Link* p_prev_child = NN_Link::get_child(p_prev);

		if (p_prev_child) {
			int n_prev_child = p_prev->get_node_index(this->_parent);

			p_prev_child->set_next_node(this, n_prev_child);
			_prev.push_back(p_prev_child);
			_input_nodes[i]->_input.push_back(&p_prev_child->get_output(n_prev_child));
			_input_nodes[i]->_in_shape.push_back(&p_prev_child->get_out_shape(n_prev_child));

			NN_Tensor<nn_type>* pd_input = new NN_Tensor<nn_type>();

			_input_nodes[i]->_d_inputs.push_back(pd_input);
			p_prev_child->get_d_output(n_prev_child).push_back(pd_input);

			++i;
		}
	}
}

void Model::calculate_output_size(std::vector<nn_shape*>& input_shape, nn_shape& out_shape) {
	for (NN_Link* p_link : _forward_list) {
		p_link->_forward->calculate_output_size(p_link->_in_shape, p_link->_out_shape);
	}
}

void Model::build(std::vector<nn_shape*>& input_shape) {
	for (NN_Link* p_link : _forward_list) {
		p_link->_forward->build(p_link->_in_shape);
	}
}

void Model::run_forward(cudaStream_t s, std::vector<NN_Tensor<nn_type>*>& input, NN_Tensor<nn_type>& output) {
	for (NN_Link* p_node : _forward_list) {
		p_node->_forward->run_forward(s, p_node->_input, p_node->_output);
	}
}

void Model::run_backward(cudaStream_t s, NN_Tensor<nn_type>& d_output, std::vector<NN_Tensor<nn_type>*>& d_input) {
	
}

void Model::compile(const std::vector<NN_Loss>& loss, const std::vector<NN_Optimizer>& optimizer) {

}

void Model::summary() {
	int i = 0;

	std::cout << '[' << _layer_name << ']' << std::endl;

	for (NN_Link* p_node : _forward_list) {
		std::cout << ++i << " : layer_name = " << p_node->_forward->_layer_name
			<< " output size: " << dimension_to_str(p_node->_out_shape) << std::endl;
	}
}

void Model::check_dimension(const nn_shape& ref_shape, const nn_shape& real_shape) {
	if (ref_shape.size() != real_shape.size()) {
		ErrorExcept(
			"[Model::check_dimension()] mismatched shape. ref_shape: %s, real_shape: %s.",
			dimension_to_str(ref_shape), dimension_to_str(real_shape)
		);
	}

	for (int i = 0; i < ref_shape.size(); ++i) {
		if ((ref_shape[i] > 0 && ref_shape[i] != real_shape[i]) || ref_shape[i] == 0 || real_shape[i] == 0) {
			ErrorExcept(
				"[Model::check_dimension()] mismatched shape. ref_shape: %s, real_shape: %s.",
				dimension_to_str(ref_shape), dimension_to_str(real_shape)
			);
		}
	}
}

#endif

#ifndef FIX_MODE

/**********************************************/
/*                                            */
/*                     Model                  */
/*                                            */
/**********************************************/

Model::Model(const char* model_name) :
	NN_Layer(model_name)
{
}

Model::Model(std::initializer_list<Layer_t> inputs, std::initializer_list<Layer_t> outputs, const char* model_name) :
	NN_Layer(model_name)
{
	/*    marking output to input    */
	NN_Manager::clear_select_mask();

	for (Layer_t output_node : outputs) {
		NN_Link* p_output = output_node[0].get()._link;

		for (Layer_t input_node : inputs) {
			NN_Link* p_input = input_node[0].get()._link;

			std::vector<NN_Link*> node_list;
			node_list.push_back(p_output);

			while (!node_list.empty()) {
				NN_Link* p_current = node_list.front();

				p_current->is_selected = true;

				for (NN_Link* p_prev : p_current->_prev) {
					if (p_prev != p_input) {
						if (!p_prev->is_selected) node_list.push_back(p_prev);
					}
					else {
						p_input->is_selected = true;
					}
				}
				node_list.erase(node_list.begin());
			}
		}
	}

	/*    create input to output layers    */
	for (Layer_t input_node : inputs) {
		NN_Link* p_input = input_node[0].get()._link;
		std::vector<NN_Link*> node_list;

		node_list.push_back(p_input);

		while (!node_list.empty()) {
			NN_Link* p_current = node_list.front();
			int prev_selects = 0;

			if (p_current->is_selected) {
				for (NN_Link* p_prev : p_current->_prev) {
					if (p_prev->is_selected) ++prev_selects;
				}

				if (prev_selects == 0) {
					p_current->is_selected = false;
					NN_Link* p_current_child = p_current->create_child();
					_forward_list.push_back(p_current_child);
					NN_Manager::add_node(p_current_child);

					for (NN_Link* p_next : p_current->_next) {
						if (p_next->is_selected) node_list.push_back(p_next);
					}
				}
			}
			node_list.erase(node_list.begin());
		}
	}

	for (Layer_t input_node : inputs) _input_nodes.push_back(NN_Link::get_child(input_node[0].get()._link));
	for (Layer_t output_node : outputs) _output_nodes.push_back(NN_Link::get_child(output_node[0].get()._link));


	/*   link child node   */
	for (NN_Link* p_child : _forward_list) {
		p_child->link_prev_child();
		//p_child->is_selected = false;
	}
	NN_Manager::clear_select_mask();
}

Model::~Model() {
	//for (NN_Link* p : _forward_list) delete p;
	//for (NN_Link* p : _backward_list) delete p;
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

	child_model->_parent = this;
	child_model->is_selected = true;
	child_model->_forward = child_model;
	child_model->_output_indices = _output_indices;

	_child.push_back(child_model);

	for (NN_Link* p_node : _forward_list) {
		NN_Link* child_node = p_node->create_child();
		child_model->_forward_list.push_back(child_node);

		NN_Manager::add_node(child_node);
	}
	for (NN_Link* p_input : _input_nodes) {
		child_model->_input_nodes.push_back(NN_Link::get_child(p_input));
	}
	for (NN_Link* p_output : _output_nodes) {
		child_model->_output_nodes.push_back(NN_Link::get_child(p_output));
	}
	for (NN_Link* p_child : child_model->_forward_list) p_child->link_prev_child();

	return child_model;
}

Layer_t Model::operator()(Layer_t& prev_node) {
	NN_Link* m_prev_node = prev_node[0].get()._link;
	int n_prev_node = prev_node[0].get()._output_index;

	m_prev_node->set_next_node(this, n_prev_node);
	_prev.push_back(m_prev_node);
	_input_nodes[0]->_input.push_back(&m_prev_node->get_output(n_prev_node));
	_input_nodes[0]->_in_shape.push_back(&m_prev_node->get_out_shape(n_prev_node));
	m_prev_node->get_d_output(n_prev_node).push_back(&_input_nodes[0]->_d_input);

	int i = 0;
	Layer_t output_nodes;

	for (NN_Link* p_output : _output_nodes) {
		output_nodes.push_back(Layer_Ptr<NN_Link> { this, i++ });
	}

	return output_nodes;
}

Layer_t Model::operator()(std::initializer_list<Layer_t> prev_node) {
	int i = 0;

	for (Layer_t p_prev_node : prev_node) {
		NN_Link* m_prev_node = p_prev_node[0].get()._link;
		int n_prev_node = p_prev_node[0].get()._output_index;

		m_prev_node->set_next_node(this, n_prev_node);
		_prev.push_back(m_prev_node);
		_input_nodes[i]->_input.push_back(&m_prev_node->get_output(n_prev_node));
		_input_nodes[i]->_in_shape.push_back(&m_prev_node->get_out_shape(n_prev_node));
		m_prev_node->get_d_output(n_prev_node).push_back(&_input_nodes[i]->_d_input);

		++i;
	}

	Layer_t output_nodes;
	i = 0;

	for (NN_Link* p_output : _output_nodes) {
		output_nodes.push_back(Layer_Ptr<NN_Link> { this, i++});
	}

	return output_nodes;
}

int Model::get_node_index(NN_Link* next_node) {
	int n = 0;

	for (int i = 0; i < _next.size(); ++i) {
		if (next_node == _next[i]) {
			n = _output_indices[i];
			break;
		}
	}

	return n;
}

void Model::set_next_node(NN_Link* next_node, int node_index) {
	_next.push_back(next_node);
	_output_indices.push_back(node_index);
}

NN_Tensor& Model::get_output(int node_index) {
	return _output_nodes[node_index]->_output;
}

std::vector<NN_Tensor*>& Model::get_d_output(int node_index) {
	return _output_nodes[node_index]->_d_output;
}

nn_shape& Model::get_out_shape(int node_index) {
	return _output_nodes[node_index]->_out_shape;
}

void Model::link_prev_child() {
	int i = 0;
	for (NN_Link* p_prev : _parent->_prev) {
		NN_Link* p_prev_child = NN_Link::get_child(p_prev);

		if (p_prev_child) {
			int n_prev_child = p_prev->get_node_index(this->_parent);

			p_prev_child->set_next_node(this, n_prev_child);
			_prev.push_back(p_prev_child);
			_input_nodes[i]->_input.push_back(&p_prev_child->get_output(n_prev_child));
			_input_nodes[i]->_in_shape.push_back(&p_prev_child->get_out_shape(n_prev_child));
			p_prev_child->get_d_output(n_prev_child).push_back(&_input_nodes[i]->_d_input);

			++i;
		}
	}
}

nn_shape Model::calculate_output_size(std::vector<nn_shape*>& input_shape) {
	return *input_shape[0];
}

void Model::build(std::vector<nn_shape*>& input_shape) {

}

NN_Tensor Model::run_forward(cudaStream_t s, std::vector<NN_Tensor*>& input) {
	for (int i = 0; i < input.size(); ++i) _input_nodes[i]->_input[0] = input[i];
	for (NN_Link* p_node : _forward_list) {
		std::vector<NN_Tensor*>& input = p_node->_input;
		NN_Tensor& output = p_node->_output;

		output = p_node->_forward->run_forward(s, input);
	}

	return NN_Tensor();
}

NN_Tensor Model::run_backward(cudaStream_t s, std::vector<NN_Tensor*>& d_output) {
	return NN_Tensor();
}

void Model::compile(const std::vector<NN_Loss>& loss, const std::vector<NN_Optimizer>& optimizer) {

}

NN_Tensor Model::train_on_batch(const std::vector<NN_Tensor>& samples, const std::vector<NN_Tensor>& truth) {
	return NN_Tensor();
}

NN_Tensor Model::fit(
	const std::vector<NN_Tensor>& samples,
	const std::vector<NN_Tensor>& truth,
	uint batch,
	uint iter
) {
	return NN_Tensor();
}

std::vector<NN_Tensor> Model::predict(std::vector<NN_Tensor>&& x) {
	std::vector<NN_Tensor> result;

	for (int i = 0; i < x.size(); ++i) _input_nodes[i]->_input.push_back(&x[i]);
	int i = 0;
	for (NN_Link* p_node : _forward_list) {
		std::vector<NN_Tensor*>& input = p_node->_input;
		NN_Tensor& output = p_node->_output;

		output = p_node->_forward->run_forward(NN_Manager::_stream, input);
		++i;
	}
	for (NN_Link* p_output : _output_nodes) result.push_back(p_output->_output);

	return result;
}

void Model::summary() {
	int i = 0;

	std::cout << '[' << _layer_name << ']' << std::endl;

	for (NN_Link* p_node : _forward_list) {
		std::cout << ++i << " : layer_name = " << p_node->_forward->_layer_name << std::endl;
	}
}

#endif