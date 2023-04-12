#include "nn_model.h"


#if !(FIX_MODE)

/**********************************************/
/*                                            */
/*                     Model                  */
/*                                            */
/**********************************************/

Model::Model(const char* model_name) :
	NN_Layer(model_name)
{
}

Model::Model(initializer_list<Layer_t> inputs, initializer_list<Layer_t> outputs, const char* model_name) :
	NN_Layer(model_name)
{
	/*    marking output to input    */
	NN_Manager::clear_select_mask();

	for (Layer_t output_node : outputs) {
		NN_Link* p_output = output_node[0]._link;
		
		for (Layer_t input_node : inputs) {
			NN_Link* p_input = input_node[0]._link;

			vector<NN_Link*> node_list;
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
		NN_Link* p_input = input_node[0]._link;
		vector<NN_Link*> node_list;

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
		NN_Link* p_input = input_node[0]._link;
		for (NN_Link* input_child : p_input->_child) {
			if (input_child->is_selected) {
				input_child->_input.push_back(NULL);
				_input_nodes.push_back(input_child);
			}
		}
	}
	for (Layer_t output_node : outputs) {
		NN_Link* p_output = output_node[0]._link;
		for (NN_Link* output_child : p_output->_child) {
			if (output_child->is_selected) _output_nodes.push_back(output_child);
		}
	}

	/*   link child node   */
	for (NN_Link* p_child : _forward_list) {
		p_child->link_prev_node();
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

	_child.push_back(child_model);

	for (NN_Link* p_node : _forward_list) {
		NN_Link* child_node = p_node->create_child();
		child_model->_forward_list.push_back(child_node);
		
		NN_Manager::add_node(child_node);
	}
	for (NN_Link* p_input : _input_nodes) {
		for (NN_Link* p_input_child : p_input->_child) {
			if (p_input_child->is_selected) {
				p_input_child->_input.push_back(NULL);
				child_model->_input_nodes.push_back(p_input_child);
			}
		}
	}
	for (NN_Link* p_output : _output_nodes) {
		for (NN_Link* p_output_child : p_output->_child) {
			if (p_output_child->is_selected) {
				child_model->_output_nodes.push_back(p_output_child);
			}
		}
	}
	for (NN_Link* p_child : child_model->_forward_list) p_child->link_prev_node();

	return child_model;
}

Layer_t Model::operator()(Layer_t& prev_node) {
	Layer_t output_nodes;

	prev_node[0]._link->_next.push_back(this);
	_prev.push_back(prev_node[0]._link);
	_input_nodes[0]->_input.push_back(prev_node[0]._output);
	prev_node[0]._d_output->push_back(&_d_input);

	for (NN_Link* p_output : _output_nodes) {
		output_nodes.push_back(Layer_Ptr<NN_Link> {this, &p_output->_output, &p_output->_d_output});
	}

	return output_nodes;
}

Layer_t Model::operator()(initializer_list<Layer_t> prev_node) {
	int i = 0;
	Layer_t output_nodes;

	for (Layer_t p_prev_node : prev_node) {
		p_prev_node[0]._link->_next.push_back(this);
		_prev.push_back(p_prev_node[0]._link);
		_input_nodes[i]->_input.push_back(p_prev_node[0]._output);
		p_prev_node[0]._d_output->push_back(&_d_input);

		++i;
	}

	for (NN_Link* p_output : _output_nodes) {
		output_nodes.push_back(Layer_Ptr<NN_Link> {this, &p_output->_output, &p_output->_d_output});
	}

	return output_nodes;
}

NN_Link* Model::get_node(NN_Link* current_node) {
	NN_Link* p_node = NULL;

	for (int i = 0; i < _next.size(); ++i) {
		if (_next[i] == current_node) p_node = _output_nodes[i];
	}

	return p_node;
}

void Model::link_prev_node() {

}

shape_type Model::calculate_output_size(shape_type& input_shape) {
	return input_shape;
}

void Model::build(shape_type& input_shape) {

}

NN_Tensor Model::run_forward(cudaStream_t s, vector<NN_Tensor*>& input) {
	for (int i = 0; i < input.size(); ++i) _input_nodes[i]->_input[0] = input[i];
	for (NN_Link* p_node : _forward_list) {
		vector<NN_Tensor*>& input = p_node->_input;
		NN_Tensor& output = p_node->_output;

		output = p_node->_forward->run_forward(s, input);
	}

	return NN_Tensor();
}

NN_Tensor Model::run_backward(cudaStream_t s, vector<NN_Tensor*>& d_output) {
	return NN_Tensor();
}

void Model::compile(const vector<NN_Loss>& loss, const vector<NN_Optimizer>& optimizer) {

}

NN_Tensor Model::train_on_batch(const vector<NN_Tensor>& samples, const vector<NN_Tensor>& truth) {
	return NN_Tensor();
}

NN_Tensor Model::fit(
	const vector<NN_Tensor>& samples,
	const vector<NN_Tensor>& truth,
	uint batch,
	uint iter
) {
	return NN_Tensor();
}

vector<NN_Tensor> Model::predict(vector<NN_Tensor>&& x) {
	vector<NN_Tensor> result;

	for (int i = 0; i < x.size(); ++i) _input_nodes[i]->_input[0] = &x[i];
	for (NN_Link* p_node : _forward_list) {
		vector<NN_Tensor*>& input = p_node->_input;
		NN_Tensor& output = p_node->_output;

		output = p_node->_forward->run_forward(NN_Manager::_stream, input);
	}
	for (NN_Link* p_output : _output_nodes) result.push_back(p_output->_output);

	return result;
}

void Model::summary() {
	int i = 0;

	cout <<'[' << _layer_name << ']' << endl;

	for (NN_Link* p_node : _forward_list) {
		cout << ++i << " : layer_name = " << p_node->_forward->_layer_name << endl;
	}
}

#else

/**********************************************/
/*                                            */
/*                     Model                  */
/*                                            */
/**********************************************/

Model::Model(const char* model_name) :
	NN_Layer(model_name)
{
}

Model::Model(initializer_list<Layer_t> inputs, initializer_list<Layer_t> outputs, const char* model_name) :
	NN_Layer(model_name)
{
	/*    marking output to input    */
	NN_Manager::clear_select_mask();

	for (Layer_t output_node : outputs) {
		NN_Link* p_output = output_node[0]._link;

		for (Layer_t input_node : inputs) {
			NN_Link* p_input = input_node[0]._link;

			vector<NN_Link*> node_list;
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
		NN_Link* p_input = input_node[0]._link;
		vector<NN_Link*> node_list;

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

	for (Layer_t input_node : inputs) _input_nodes.push_back(NN_Link::get_child(input_node[0]._link));
	for (Layer_t output_node : outputs) _output_nodes.push_back(NN_Link::get_child(output_node[0]._link));


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
	NN_Link* m_prev_node = prev_node[0]._link;
	int n_prev_node = prev_node[0]._output_index;

	m_prev_node->set_next_node(this, n_prev_node);
	_prev.push_back(m_prev_node);
	_input_nodes[0]->_input.push_back(&m_prev_node->get_output(n_prev_node));
	m_prev_node->get_d_output(n_prev_node).push_back(&_input_nodes[0]->_d_input);

	int i = 0;
	Layer_t output_nodes;

	for (NN_Link* p_output : _output_nodes) {
		output_nodes.push_back(Layer_Ptr<NN_Link> { this, i++ });
	}

	return output_nodes;
}

Layer_t Model::operator()(initializer_list<Layer_t> prev_node) {
	int i = 0;

	for (Layer_t p_prev_node : prev_node) {
		NN_Link* m_prev_node = p_prev_node[0]._link;
		int n_prev_node = p_prev_node[0]._output_index;

		m_prev_node->set_next_node(this, n_prev_node);
		_prev.push_back(m_prev_node);
		_input_nodes[i]->_input.push_back(&m_prev_node->get_output(n_prev_node));
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

vector<NN_Tensor*>& Model::get_d_output(int node_index) {
	return _output_nodes[node_index]->_d_output;
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
			p_prev_child->get_d_output(n_prev_child).push_back(&_input_nodes[i]->_d_input);

			++i;
		}
	}
}

shape_type Model::calculate_output_size(shape_type& input_shape) {
	return input_shape;
}

void Model::build(shape_type& input_shape) {

}

NN_Tensor Model::run_forward(cudaStream_t s, vector<NN_Tensor*>& input) {
	for (int i = 0; i < input.size(); ++i) _input_nodes[i]->_input[0] = input[i];
	for (NN_Link* p_node : _forward_list) {
		vector<NN_Tensor*>& input = p_node->_input;
		NN_Tensor& output = p_node->_output;

		output = p_node->_forward->run_forward(s, input);
	}

	return NN_Tensor();
}

NN_Tensor Model::run_backward(cudaStream_t s, vector<NN_Tensor*>& d_output) {
	return NN_Tensor();
}

void Model::compile(const vector<NN_Loss>& loss, const vector<NN_Optimizer>& optimizer) {

}

NN_Tensor Model::train_on_batch(const vector<NN_Tensor>& samples, const vector<NN_Tensor>& truth) {
	return NN_Tensor();
}

NN_Tensor Model::fit(
	const vector<NN_Tensor>& samples,
	const vector<NN_Tensor>& truth,
	uint batch,
	uint iter
) {
	return NN_Tensor();
}

vector<NN_Tensor> Model::predict(vector<NN_Tensor>&& x) {
	vector<NN_Tensor> result;

	for (int i = 0; i < x.size(); ++i) _input_nodes[i]->_input.push_back(&x[i]);
	int i = 0;
	for (NN_Link* p_node : _forward_list) {
		vector<NN_Tensor*>& input = p_node->_input;
		NN_Tensor& output = p_node->_output;

		output = p_node->_forward->run_forward(NN_Manager::_stream, input);
		++i;
	}
	for (NN_Link* p_output : _output_nodes) result.push_back(p_output->_output);

	return result;
}

void Model::summary() {
	int i = 0;

	cout << '[' << _layer_name << ']' << endl;

	for (NN_Link* p_node : _forward_list) {
		cout << ++i << " : layer_name = " << p_node->_forward->_layer_name << endl;
	}
}

#endif