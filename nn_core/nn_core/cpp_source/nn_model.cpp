#include "nn_model.h"


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

NN_Vec<NN_Coupler<NN_Link>> NN_Model::operator()(const NN_Vec<NN_Coupler<NN_Link>> m_prev_link) {
	for (int i = 0; i < input_nodes.size(); ++i) {
		prev_link.push_back(m_prev_link[i].link);
		input_nodes[i]->input.push_back(m_prev_link[i].output);
		input_nodes[i]->d_output.push_back(m_prev_link[i].d_input);
		input_nodes[i]->in_shape.push_back(m_prev_link[i].out_size);
		m_prev_link[i].link->next_link.push_back(this);
	}

	vector<NN_Coupler<NN_Link>> p_currents;
	for (int i = 0; i < output_nodes.size(); ++i) {
		NN_Coupler<NN_Link> args;

		args.link = this;
		args.output = &output_nodes[i]->output;
		args.d_input = &output_nodes[i]->d_input;
		args.out_size = &output_nodes[i]->out_shape;

		p_currents.push_back(args);
	}

	return p_currents;
}

vector<NN_Link*> NN_Model::find_root(vector<NN_Link*>& in_nodes, vector<NN_Link*>& out_nodes) {
	vector<NN_Link*> tmp_links;
	vector<NN_Link*> order_links = out_nodes;
	int least_io = in_nodes.size();

	NN_Manager::clear_select_flag();

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
					order_links.push_back(p_prev_link);
				}
			}
		}

		order_links.erase(order_links.begin());
	}

	order_links = in_nodes;
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
					order_links.push_back(p_next_link);
				}
			}
		}
		order_links.erase(order_links.begin());
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

vector<NN_Link*> NN_Model::set_operate_list(vector<NN_Link*> in_layers, vector<NN_Link*> out_layers) {
	vector<NN_Link*> tmp_list;

	for (NN_Link* p_link : in_layers) {
		if (!p_link->is_selected && get_unselect_prev(p_link) == 0) {
			for (NN_Link* p_next_link : p_link->next_link) {
				if (!p_next_link->is_selected) {
					in_layers.push_back(p_next_link);
				}
			}
			tmp_list.push_back(p_link);
			p_link->is_selected = true;
		}
	}

	return tmp_list;
}

NN_Model::NN_Model(NN_Model* parent) :
	NN_Layer("model_child")
{
	NN_Manager::clear_select_flag();
	gen_child(parent->operate_list);
	
	for (NN_Link* p_input_parent : parent->input_nodes) {
		input_nodes.push_back(get_child_link(p_input_parent));
	}
	for (NN_Link* p_output_parent : parent->output_nodes) {
		output_nodes.push_back(get_child_link(p_output_parent));
	}

	NN_Manager::clear_select_flag();

	operate_list = set_operate_list(input_nodes, output_nodes);
	op_layer = this;
}

NN_Model::NN_Model(NN& inputs, NN& outputs, const string& model_name) :
	NN_Layer(model_name)
{
	op_layer = this;

	for (NN_Coupler<NN_Link>& p_input : inputs.arr) input_nodes.push_back(p_input.link);
	for (NN_Coupler<NN_Link>& p_output : outputs.arr) output_nodes.push_back(p_output.link);

	vector<NN_Link*> tmp_links = find_root(input_nodes, output_nodes);
	tmp_links = gen_child(tmp_links);

	for (NN_Link*& p_input_parent : input_nodes) {
		p_input_parent = get_child_link(p_input_parent);
	}
	for (NN_Link*& p_output_parent : output_nodes) {
		p_output_parent = get_child_link(p_output_parent);
	}

	NN_Manager::clear_select_flag();

	operate_list = set_operate_list(input_nodes, output_nodes);
}

NN_Model::~NN_Model() {

}

NN_Link* NN_Model::create_child_link() {
	NN_Model* p_child = new NN_Model(this);

	return p_child;
}

void NN_Model::calculate_output_size(vector<NN_Shape*>& input_shape, NN_Shape& output_shape) {
	for (NN_Link* p : operate_list) {
		vector<NN_Shape*>& in_shape = p->in_shape;
		NN_Shape& out_shape = p->out_shape;

		p->op_layer->calculate_output_size(in_shape, out_shape);
	}
}

void NN_Model::build(vector<NN_Shape*>& input_shape) {
	for (NN_Link* p : operate_list) {
		vector<NN_Shape*>& in_shape = p->in_shape;


	}
}

void NN_Model::run_forward(vector<NN_Tensor*>& input, NN_Tensor& output) {
	for (NN_Link* p : operate_list) {
		vector<NN_Tensor*>& input = p->input;
		NN_Tensor& output = p->output;

		p->op_layer->run_forward(input, output);
	}
}

void NN_Model::run_backward(vector<NN_Tensor*>& d_output, NN_Tensor& d_input) {

}

void NN_Model::compile(const vector<NN_Loss*>& loss, const vector<NN_Optimizer*>& optimizer) {

}

NN_Tensor_t NN_Model::train_on_batch(const vector<NN_Tensor_t>& samples, const vector<NN_Tensor_t>& truth) {
	NN_Tensor_t loss(new NN_Tensor());

	return loss;
}

NN_Tensor_t NN_Model::fit(
	const vector<NN_Tensor_t>& samples,
	const vector<NN_Tensor_t>& truth,
	uint batch,
	uint iter
) {
	NN_Tensor_t loss(new NN_Tensor());

	return loss;
}

vector<NN_Tensor_t> NN_Model::predict(const vector<NN_Tensor_t>& x) {
	return x;
}

NN_Model& Model(NN& inputs, NN& outputs, const string& model_name) {
	NN_Model* model = new NN_Model(inputs, outputs, model_name);
	NN_Manager::add_link(model);
	
	return *model;
}