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

Model::Model(initializer_list<vector<Layer_t<NN_Link>>> inputs, initializer_list<vector<Layer_t<NN_Link>>> outputs, const char* model_name) :
	NN_Layer(model_name)
{
	/*    marking output to input    */
	NN_Manager::clear_select_mask();

	for (vector<Layer_t<NN_Link>> output_node : outputs) {
		NN_Link* p_output = output_node[0]._link;
		
		for (vector<Layer_t<NN_Link>> input_node : inputs) {
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
	for (vector<Layer_t<NN_Link>> input_node : inputs) {
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

	for (vector<Layer_t<NN_Link>> input_node : inputs) {
		NN_Link* p_input = input_node[0]._link;
		for (NN_Link* input_child : p_input->_child) {
			if (input_child->is_selected) {
				_input_nodes.push_back(input_child);
			}
		}
	}
	for (vector<Layer_t<NN_Link>> output_node : outputs) {
		NN_Link* p_output = output_node[0]._link;
		for (NN_Link* output_child : p_output->_child) {
			if (output_child->is_selected) _output_nodes.push_back(output_child);
		}
	}

	/*   link child node   */
	for (NN_Link* p_child : _forward_list) {
		NN_Link::set_child_link(p_child);
		p_child->is_selected = false;
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
	for (NN_Link* p_child : child_model->_forward_list) {
		NN_Link::set_child_link(p_child);
		p_child->is_selected = false;
	}

	return child_model;
}

vector<Layer_t<NN_Link>> Model::operator()(vector<Layer_t<NN_Link>>& prev_node) {
	vector<Layer_t<NN_Link>> output_nodes;

	prev_node[0]._link->_next.push_back(this);
	_prev.push_back(prev_node[0]._link);
	_input_nodes[0]->_input.push_back(prev_node[0]._output);
	prev_node[0]._d_output->push_back(&_d_input);

	for (NN_Link* p_output : _output_nodes) {
		output_nodes.push_back(Layer_t<NN_Link> {this, &p_output->_output, &p_output->_d_output});
	}

	return output_nodes;
}

vector<Layer_t<NN_Link>> Model::operator()(initializer_list<vector<Layer_t<NN_Link>>> prev_node) {
	int i = 0;
	vector<Layer_t<NN_Link>> output_nodes;

	for (vector<Layer_t<NN_Link>> p_prev_node : prev_node) {
		p_prev_node[0]._link->_next.push_back(this);
		_prev.push_back(p_prev_node[0]._link);
		_input_nodes[i]->_input.push_back(p_prev_node[0]._output);
		p_prev_node[0]._d_output->push_back(&_d_input);

		++i;
	}

	for (NN_Link* p_output : _output_nodes) {
		output_nodes.push_back(Layer_t<NN_Link> {this, &p_output->_output, &p_output->_d_output});
	}

	return output_nodes;
}

shape_type Model::calculate_output_size(shape_type& input_shape) {
	return input_shape;
}

void Model::build(shape_type& input_shape) {

}

NN_Tensor Model::run_forward(cudaStream_t s, vector<NN_Tensor*>& input) {
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

vector<NN_Tensor> Model::predict(const vector<NN_Tensor>& x) {
	return x;
}

void Model::summary() {
	int i = 0;

	cout <<'[' << _layer_name << ']' << endl;

	for (NN_Link* p_node : _forward_list) {
		cout << ++i << " : layer_name = " << p_node->_forward->_layer_name << endl;
	}
}