#include "nn_model.h"
#include <queue>


NN_Link* NN_Model::get_child_link(NN_Link* parent_link) {
	NN_Link* p_child = NULL;

	for (NN_Link* current_child : parent_link->child) {
		if (current_child->is_selected) p_child = current_child;
	}

	return p_child;
}

NN_Vec<NN_Coupler<NN_Link>> NN_Model::operator()(const NN_Vec<NN_Coupler<NN_Link>> m_prev_link) {
	for (int i = 0; i < input_nodes.size(); ++i) {
		prev_link.push_back(m_prev_link[i].link);
		input_nodes[i]->input.push_back(m_prev_link[i].output);
		input_nodes[i]->d_output.push_back(m_prev_link[i].d_input);
		input_nodes[i]->input_shape.push_back(m_prev_link[i].out_size);
		m_prev_link[i].link->next_link.push_back(this);
	}

	vector<NN_Coupler<NN_Link>> p_currents;
	for (int i = 0; i < output_nodes.size(); ++i) {
		NN_Coupler<NN_Link> args;

		args.link = this;
		args.output = &output_nodes[i]->output;
		args.d_input = &output_nodes[i]->d_input;
		args.out_size = &output_nodes[i]->output_shape;
	}

	return p_currents;
}

NN_Model::NN_Model(NN_Vec<NN_Link*>& inputs, NN_Vec<NN_Link*>& outputs) :
	NN_Link((NN_Layer*)this)
{
	vector<NN_Link*> tmp_links;
	vector<NN_Link*> order_links;
	vector<NN_Link*> io_links;

	for (NN_Link* p_link : outputs.arr) order_links.push_back(p_link);
	for (NN_Link* p_link : inputs.arr) io_links.push_back(p_link);

	NN_Manager::clear_select_flag();

	while (!order_links.empty() && !io_links.empty()) {
		NN_Link* p_current = order_links.front();
		bool is_touched = false;

		p_current->is_selected = true;

		for (vector<NN_Link*>::iterator i = io_links.begin(); i != io_links.end(); ++i) {
			if (*i == p_current) {
				is_touched = true;
				io_links.erase(i);
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

	order_links.clear();
	io_links.clear();
	
	for (NN_Link* p_link : inputs.arr) order_links.push_back(p_link);
	for (NN_Link* p_link : outputs.arr) io_links.push_back(p_link);

	while (!order_links.empty() && !io_links.empty()) {
		NN_Link* p_current = order_links.front();
		bool is_touched = false;

		p_current->is_selected = false;
		tmp_links.push_back(p_current);

		for (vector<NN_Link*>::iterator i = io_links.begin(); i != io_links.end(); ++i) {
			if (*i == p_current) {
				is_touched = true;
				io_links.erase(i);
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

	for (NN_Link* p_link : tmp_links) {
		if (!p_link->is_selected) {
			NN_Link* p_child_link = new NN_Link(p_link);
			
			p_child_link->is_selected = true;
			p_link->child.push_back(p_child_link);
			p_link->is_selected = true;
		}
	}
	for (NN_Link* p_parent : tmp_links) {
		NN_Link* p_child = get_child_link(p_parent);

		if (p_child) {
			for (NN_Link* p_next_parent : p_parent->next_link) {
				if (p_next_parent->is_selected) {
					NN_Link* p_next_child = get_child_link(p_next_parent);

					if (p_next_child) (*p_child)(p_next_parent);
				}
			}
		}
	}
	NN_Manager::clear_select_flag();
}

void NN_Model::calculate_output_size(NN_Vec<Dim*> input_shape, NN_Vec<Dim*> output_shape) {

}

void NN_Model::build(NN_Vec<Dim*> input_shape) {

}

void NN_Model::run_forward(NN_Vec<NN_Tensor*> input, NN_Vec<NN_Tensor*> output) {
	if (input_nodes.size() != input.arr.size()) {
		ErrorExcept(
			"[NN_Model::run_forward] invalid input and input_nodes size."
		);
	}

	vector<NN_Link*> order_links;
	
	NN_Manager::clear_select_flag();

	for (int i = 0; i < input_nodes.size(); ++i) {
		input_nodes[i]->op_layer->run_forward(input[i], &input_nodes[i]->output);
		input_nodes[i]->is_selected = true;
		order_links.push_back(input_nodes[i]);
	}
	while (!order_links.empty()) {
		NN_Link* p_current_link = order_links.front();
		
		if (!p_current_link->is_selected) {
			p_current_link->op_layer->run_forward(
				p_current_link->input,
				p_current_link->output
			);
		}
		p_current_link->is_selected = true;
		for (NN_Ptr<NN_Link>& p_next_link : p_current_link->next_link) {
			order_links.push(p_next_link);
		}
		order_links.pop();
	}
}

void NN_Model::run_backward(NN_Vec<NN_Tensor*> d_output, NN_Vec<NN_Tensor*> d_input) {

}

void NN_Model::compile(const vector<NN_Loss*>& loss, const vector<NN_Optimizer*>& optimizer) {

}

NN_Tensor NN_Model::train_on_batch(const vector<NN_Tensor>& samples, const vector<NN_Tensor>& truth) {

}

NN_Tensor NN_Model::fit(
	const vector<NN_Tensor>& samples,
	const vector<NN_Tensor>& truth,
	uint batch,
	uint iter
) {

}

vector<NN_Tensor> NN_Model::predict(const vector<NN_Tensor>& x) {

}