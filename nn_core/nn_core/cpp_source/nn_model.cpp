#include "nn_model.h"
#include <queue>


NN_Link* NN_Model::get_child_link(NN_Link* parent_link) {
	NN_Link* p_child = NULL;

	for (NN_Link* current_child : parent_link->child) {
		if (current_child->is_selected) p_child = current_child;
	}

	return p_child;
}

int NN_Model::get_unselect_prev(NN_Link* p_current) {
	int unselectd_link = 0;

	for (NN_Link* p_link : p_current->prev_link) {
		if (!p_link->is_selected) ++unselectd_link;
	}

	return unselectd_link;
}

vector<NN_Link*> NN_Model::find_root(vector<NN_Link*>& in_nodes, vector<NN_Link*>& out_nodes) {
	vector<NN_Link*> tmp_links;
	queue<NN_Link*> order_links;
	int least_io = in_nodes.size();

	NN_Manager::clear_select_flag();

	for (NN_Link* p_out_node : out_nodes) order_links.push(p_out_node);

	while (!order_links.empty() && least_io > 0) {
		NN_Link* p_current = order_links.front();
		bool is_touched = false;

		p_current->is_selected = true;

		for (vector<NN_Link*>::iterator i = in_nodes.begin(); i != in_nodes.end(); ++i) {
			if (*i == p_current) {
				is_touched = true;
				--least_io;
				
				break;
			}
		}

		if (!is_touched) {
			for (NN_Link* p_prev_link : p_current->prev_link) {
				if (!p_prev_link->is_selected) {
					order_links.push(p_prev_link);
				}
			}
		}

		order_links.pop();
	}

	order_links = queue<NN_Link*>();
	
	for (NN_Link* p_in_node : in_nodes) order_links.push(p_in_node);

	least_io = out_nodes.size();

	while (!order_links.empty() && least_io > 0) {
		NN_Link* p_current = order_links.front();
		bool is_touched = false;

		p_current->is_selected = false;
		tmp_links.push_back(p_current);

		for (vector<NN_Link*>::iterator i = out_nodes.begin(); i != out_nodes.end(); ++i) {
			if (*i == p_current) {
				is_touched = true;
				--least_io;

				break;
			}
		}

		if (!is_touched) {
			for (NN_Link* p_next_link : p_current->next_link) {
				if (p_next_link->is_selected) {
					order_links.push(p_next_link);
				}
			}
		}
		order_links.pop();
	}
	NN_Manager::clear_select_flag();

	return tmp_links;
}

vector<NN_Link*> NN_Model::gen_child(vector<NN_Link*>& selected_parents) {
	vector<NN_Link*> tmp_links;

	for (NN_Link* p_link : selected_parents) {
		if (!p_link->is_selected) {
			NN_Link* p_child_link = p_link->create_child_link();

			NN_Manager::add_link(p_child_link);
			tmp_links.push_back(p_child_link);

			p_child_link->is_selected = true;
			p_link->is_selected = true;
		}
	}
	for (NN_Link* p_parent : selected_parents) {
		NN_Link* p_child = get_child_link(p_parent);

		if (p_child) {
			for (NN_Link* p_next_parent : p_parent->next_link) {
				if (p_next_parent->is_selected) {
					NN_Link* p_next_child = get_child_link(p_next_parent);

					if (p_next_child) (*p_next_child)(p_child);
				}
			}
		}
	}

	return tmp_links;
}

vector<NN_Link*> NN_Model::set_operate_list(vector<NN_Link*> in_layers) {
	vector<NN_Link*> tmp_list;
	queue<NN_Link*> _in_layers;

	for (NN_Link* p_in_layer : in_layers) _in_layers.push(p_in_layer);

	while (!_in_layers.empty()) {
		NN_Link* p_link = _in_layers.front();

		_in_layers.pop();
		if (!p_link->is_selected && get_unselect_prev(p_link) == 0) {
			for (NN_Link* p_next_link : p_link->next_link) {
				if (!p_next_link->is_selected) {
					_in_layers.push(p_next_link);
				}
			}
			tmp_list.push_back(p_link);
			p_link->is_selected = true;
		}
	}

	return tmp_list;
}

NN_Model::NN_Model(NN_Model* p_parent) :
	NN_Layer("model_child")
{
	parent = p_parent;
	p_parent->child.push_back(this);

//	NN_Manager::clear_select_flag();
	gen_child(p_parent->operate_list);
	
	for (NN_Link* p_input_parent : p_parent->input_nodes) {
		input_nodes.push_back(get_child_link(p_input_parent));
	}
	for (NN_Link* p_output_parent : p_parent->output_nodes) {
		output_nodes.push_back(get_child_link(p_output_parent));
	}

//	NN_Manager::clear_select_flag();

	operate_list = set_operate_list(input_nodes);
	op_layer = this;
}

NN_Model::NN_Model(const NN_Coupler<NN_Link>& inputs, const NN_Coupler<NN_Link>& outputs, const string& model_name) :
	NN_Layer(model_name)
{
	op_layer = this;

	for (Link_Param<NN_Link>& p_input : inputs) input_nodes.push_back(p_input.link);
	for (Link_Param<NN_Link>& p_output : outputs) output_nodes.push_back(p_output.link);

	vector<NN_Link*> tmp_links = find_root(input_nodes, output_nodes);
	tmp_links = gen_child(tmp_links);

	for (NN_Link*& p_input_parent : input_nodes) {
		p_input_parent = get_child_link(p_input_parent);
	}
	for (NN_Link*& p_output_parent : output_nodes) {
		p_output_parent = get_child_link(p_output_parent);
	}

	NN_Manager::clear_select_flag();

	operate_list = set_operate_list(input_nodes);
	NN_Manager::clear_select_flag();
}

NN_Model::~NN_Model() {

}

NN_Coupler<NN_Link> NN_Model::operator()(NN_Coupler<NN_Link> m_prev_link) {
	for (int i = 0; i < m_prev_link.len; ++i) {
		prev_link.push_back(m_prev_link[i].link);
		m_prev_link[i].link->next_link.push_back(this);

		input_nodes[i]->input.push_back(&m_prev_link[i].sub_link->output);

		NN_Tensor_t p_dio = new NN_Tensor;
		m_prev_link[i].sub_link->d_output.push_back(p_dio);
		input_nodes[i]->d_input.push_back(p_dio);

		input_nodes[i]->in_shape.push_back(&m_prev_link[i].sub_link->out_shape);
	}

	vector<Link_Param<NN_Link>> args;
	for (NN_Link* p_link : output_nodes) {
		Link_Param<NN_Link> arg;

		arg.link = this;
		arg.sub_link = p_link;

		args.push_back(arg);
	}

	return args;
}

NN_Link* NN_Model::get_prev_ptr(NN_Link* p_current) {
	int index = 0;

	for (NN_Link* p_next : next_link) {
		if (p_next == p_current) break;
		else ++index;
	}

	return output_nodes[index];
}

NN_Link* NN_Model::get_current_ptr(NN_Link* p_prev) {
	int index = 0;

	for (NN_Link* p_current : prev_link) {
		if (p_current == p_prev) break;
		else ++index;
	}

	return input_nodes[index];
}

NN_Link* NN_Model::create_child_link() {
	NN_Model* p_child = new NN_Model(this);

	return p_child;
}

void NN_Model::calculate_output_size(vector<NN_Shape>& input_shape, NN_Shape& output_shape) {

}

void NN_Model::build(vector<NN_Shape>& input_shape) {

}

void NN_Model::run_forward(vector<NN_Tensor>& input, NN_Tensor& output) {
	vector<NN_Link*>::iterator p_op = input_nodes.begin();

	for (NN_Link* p_prev_link : prev_link) {
		NN_Link* p_prev = p_prev_link->get_prev_ptr(p_prev_link);
		
		(*p_op++)->op_layer->run_forward({ p_prev->output }, output);

	}
}

void NN_Model::run_backward(vector<NN_Tensor>& input, NN_Tensor& output, NN_Tensor& d_output, vector<NN_Tensor>& d_input) {

}

void NN_Model::compile(const vector<NN_Loss*>& loss, const vector<NN_Optimizer*>& optimizer) {

}

NN_Tensor NN_Model::train_on_batch(const vector<NN_Tensor>& samples, const vector<NN_Tensor>& truth) {
	return NN_Tensor();
}

NN_Tensor NN_Model::fit(
	const vector<NN_Tensor>& samples,
	const vector<NN_Tensor>& truth,
	uint batch,
	uint iter
) {
	return NN_Tensor();
}

vector<NN_Tensor> NN_Model::predict(const vector<NN_Tensor>& x) {
	return x;
}

void NN_Model::summary() {
	for (NN_Link* p : operate_list) printf("%s\n", p->op_layer->layer_name.c_str());
}

NN_Model& Model(const NN_Coupler<NN_Link>& inputs, const NN_Coupler<NN_Link>& outputs, const string& model_name) {
	NN_Model* model = new NN_Model(inputs, outputs, model_name);
	NN_Manager::add_link(model);
	
	return *model;
}