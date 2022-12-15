#include "nn_model.h"


int NN_Model::get_link_index(vector<NN_Ptr<NN_Link>>& link_list, NN_Ptr<NN_Link>& target) {
	int index = -1;

	for (int i = 0; i < link_list.size(); ++i) {
		if (target != link_list[i]) {
			index = i;
			break;
		}
	}

	return index;
}

NN_Model::NN_Model(NN_Coupler<NN_Link>& inputs, NN_Coupler<NN_Link>& outputs) :
	NN_Link((NN_Layer*)this)
{
	vector<NN_Ptr<NN_Link>> m_links;

	for (NN_Ptr<NN_Link>& p_output : outputs.get()) {
		for (NN_Ptr<NN_Link>& p_input : inputs.get()) {
			vector<NN_Ptr<NN_Link>> selected_link;

			selected_link.push_back(p_output);

			for (NN_Ptr<NN_Link>& current_link : selected_link) {
				current_link->is_selected = true;

				if (current_link != p_input) {
					for (NN_Ptr<NN_Link>& p_prev : current_link->prev_link) {
						if (!p_prev->is_selected) selected_link.push_back(p_prev);
					}
				}
			}

			selected_link.clear();
			selected_link.push_back(p_input);

			for (NN_Ptr<NN_Link>& current_link : selected_link) {
				m_links.push_back(current_link);
				current_link->is_selected = false;

				for (NN_Ptr<NN_Link>& p_next : current_link->next_link) {
					if (p_next->is_selected) {
						selected_link.push_back(p_next);
					}
				}
			}
		}
	}
	NN_Manager::clear_select_flags();

	for (NN_Ptr<NN_Link>& p_link : m_links) {
		p_link->is_selected = true;
	}
	for (NN_Ptr<NN_Link>& p_input : inputs.get()) {
		vector<NN_Ptr<NN_Link>> sel_child_links;
		NN_Ptr<NN_Link> p_input_child = new NN_Link(p_input);

		sel_child_links.push_back(p_input_child);
		input_nodes.push_back(p_input_child);

		for (NN_Ptr<NN_Link>& curr_child_link : sel_child_links) {
			curr_child_link->is_selected = false;
			m_layers.push_back(curr_child_link);

			for (NN_Ptr<NN_Link>& p_next_parent : curr_child_link->parent->next_link) {
				if (p_next_parent->is_selected) {
					NN_Ptr<NN_Link> p_next_child = new NN_Link(p_next_parent);
					
					(*p_next_child)(curr_child_link);
					sel_child_links.push_back(p_next_child);

					for (NN_Ptr<NN_Link>& p_output_parent : outputs.get()) {
						if (p_output_parent == p_next_parent) {
							output_nodes.push_back(p_next_child);
						}
					}
				}
			}
		}
	}
	NN_Manager::clear_select_flags();
}

void NN_Model::calculate_output_size(vector<Dim*>& input_shape, Dim& output_shape) {

}

void NN_Model::build(vector<Dim*>& input_shape) {

}

void NN_Model::run_forward(vector<NN_Ptr<NN_Tensor>>& input, NN_Ptr<NN_Tensor>& output) {
	if (input_nodes.size() != input.size()) {
		ErrorExcept(
			"[NN_Model::run_forward] invalid input and input_nodes size."
		);
	}

	for (size_t i = 0; i < input.size(); ++i) {
		vector<NN_Ptr<NN_Tensor>> p_input;
		NN_Ptr<NN_Tensor>& p_output = input_nodes[i]->output;

		p_input.push_back(input[i]);

		input_nodes[i]->layer->run_forward(p_input, p_output);
	}
	for (NN_Ptr<NN_Link>& p_link : m_layers) {
		vector<NN_Ptr<NN_Tensor>>& p_input = p_link->input;
		NN_Ptr<NN_Tensor>& p_output = p_link->output;

		p_link->layer->run_forward(p_input, p_output);
	}
}

void NN_Model::run_backward(vector<NN_Ptr<NN_Tensor>>& d_output, NN_Ptr<NN_Tensor>& d_input) {

}

void NN_Model::compile(vector<NN_Loss*>& loss, vector<NN_Optimizer*>& optimizer) {

}

NN_Ptr<NN_Tensor> NN_Model::train_on_batch(const vector<NN_Ptr<NN_Tensor>>& samples, const vector<NN_Ptr<NN_Tensor>>& truth) {

}

NN_Ptr<NN_Tensor> NN_Model::fit(
	const vector<NN_Ptr<NN_Tensor>>& samples,
	const vector<NN_Ptr<NN_Tensor>>& truth,
	uint batch,
	uint iter
) {

}

vector<NN_Ptr<NN_Tensor>> NN_Model::predict(const vector<NN_Ptr<NN_Tensor>>& x) {

}

NN_Ptr<NN_Tensor> NN_Model::get_prev_output(NN_Ptr<NN_Link>& p_current) {
	int index = get_link_index(prev_link, p_current);

	return output_nodes[index]->output;
}

NN_Ptr<NN_Tensor> NN_Model::get_next_dinput(NN_Ptr<NN_Link>& p_current) {
	int index = get_link_index(next_link, p_current);

	return input_nodes[index]->d_input;
}

Dim NN_Model::get_next_output_shape(NN_Ptr<NN_Link>& p_current) {
	int index = get_link_index(prev_link, p_current);

	return output_nodes[index]->output_shape;
}



NN_Link& Model(vector<NN_Ptr<NN_Link>>& input, vector<NN_Ptr<NN_Link>>& output) {
	NN_Ptr<NN_Link> link = new NN_Model(input, output);

	NN_Manager::add_link(link);

	return *link;
}