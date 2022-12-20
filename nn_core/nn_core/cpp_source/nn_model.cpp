#include "nn_model.h"
#include <queue>


int NN_Model::where_selected_link(vector<NN_Ptr<NN_Link>>& link_list) {
	int index = -1;

	for (int i = 0; i < link_list.size(); ++i) {
		if (link_list[i]->is_selected) {
			index = i;
			break;
		}
	}

	return index;
}

int NN_Model::get_link_index(vector<NN_Ptr<NN_Link>>& link_list, NN_Ptr<NN_Link>& target) {
	int index = -1;

	for (int i = 0; i < link_list.size(); ++i) {
		if (link_list[i] == target) {
			index = i;
			break;
		}
	}

	return index;
}

bool NN_Model::set_terminal_node(vector<NN_Ptr<NN_Link>>& storage, vector<NN_Ptr<NN_Link>>& node) {
	bool success_flag = true;

	for (NN_Ptr<NN_Link>& p_node : node) {
		int index = where_selected_link(p_node->child);
		if (index > -1) {
			storage.push_back(p_node->child[index]);
		}
		else {
			success_flag = false;
			break;
		}
	}

	return success_flag;
}

NN_Model::NN_Model(NN_Coupler<NN_Link>& inputs, NN_Coupler<NN_Link>& outputs) :
	NN_Link((NN_Layer*)this)
{
	vector<NN_Ptr<NN_Link>> tmp_links;

	NN_Manager::set_linked_count();
	for (NN_Ptr<NN_Link>& p_output : outputs.get()) {
		for (NN_Ptr<NN_Link>& p_input : inputs.get()) {
			vector<NN_Ptr<NN_Link>> selected_links;

			selected_links.push_back(p_output);

			for (NN_Ptr<NN_Link>& current_link : selected_links) {
				if (current_link != p_input) {
					for (NN_Ptr<NN_Link>& p_prev : current_link->prev_link) {
						if (!p_prev->is_selected) {
							p_prev->is_selected = true;
							selected_links.push_back(p_prev);
						}
					}
				}
			}

			selected_links.clear();
			selected_links.push_back(p_input);

			for (NN_Ptr<NN_Link>& current_link : selected_links) {
				tmp_links.push_back(current_link);
				if (current_link != p_output) {
					for (NN_Ptr<NN_Link>& p_next : current_link->next_link) {
						if (p_next->is_selected) {
							p_next->is_selected = false;
							selected_links.push_back(p_next);
						}
					}
				}
			}
			NN_Manager::clear_select_flag();
		}
	}

	for (NN_Ptr<NN_Link>& p_link : tmp_links) {
		if (!p_link->is_selected) {
			p_link->child.push_back(new NN_Link(p_link));
			p_link->is_selected = true;
		}
	}

	set_terminal_node(input_nodes, inputs.get());
	set_terminal_node(output_nodes, outputs.get());

	for (NN_Ptr<NN_Link>& p_current_parent : tmp_links) {
		if (p_current_parent->is_selected) {
			int current_child_index = where_selected_link(p_current_parent->child);
			if (current_child_index > -1) {
				NN_Ptr<NN_Link>& p_current_child = p_current_parent->child[current_child_index];
				for (NN_Ptr<NN_Link>& p_next_parent : p_current_parent->next_link) {
					int next_child_index = where_selected_link(p_next_parent->child);
					if (next_child_index > -1) {
						NN_Ptr<NN_Link>& p_next_child = p_next_parent->child[next_child_index];
						(*p_next_child)(p_current_child);
					}
				}
			}
			p_current_parent->is_selected = false;
		}
	}
	
	NN_Manager::clear_select_flag();
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

	queue<NN_Ptr<NN_Link>> order_links;
	
	for (int i = 0; i < input_nodes.size(); ++i) {
		input_nodes[i]->input.clear();
		input_nodes[i]->input.push_back(input[i]);
		order_links.push(input_nodes[i]);
	}

	while (!order_links.empty()) {
		NN_Ptr<NN_Link> p_current_link = order_links.front();
		
		if (!p_current_link->is_selected) {
			p_current_link->layer->run_forward(
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