#include "nn_model.h"


NN_Model::NN_Model(vector<NN_Link*> input, vector<NN_Link*> output) {
	vector<NN_Link*> m_links;

	for (NN_Link* p_input : input) {
		for (NN_Link* p_output : output) {
			vector<NN_Link*> selected_link;

			selected_link.push_back(p_input);

			for (NN_Link* current_link : selected_link) {
				current_link->is_selected = true;

				if (current_link != p_output) {
					for (NN_Link* next_link : current_link->m_next) {
						if (!next_link->is_selected) selected_link.push_back(next_link);
					}
				}
			}

			selected_link.clear();
			selected_link.push_back(p_output);

			for (NN_Link* current_link : selected_link) {
				m_links.push_back(current_link);
				current_link->is_selected = false;

				for (NN_Link* p_prev_link : current_link->m_prev) {
					if (p_prev_link->is_selected) {
						selected_link.push_back(p_prev_link);
					}
				}
			}
		}
	}
	NN_Manager::clear_select();

	for (NN_Link* p_link : m_links) {
		p_link->is_selected = true;
	}
	for (NN_Link* p_input : input) {
		vector<NN_Link*> sel_child_links;
		NN_Link* p_input_child = new NN_Link(p_input);

		sel_child_links.push_back(p_input_child);
		input_nodes.push_back(p_input_child);

		for (NN_Link* curr_child_link : sel_child_links) {
			NN_Manager::add_link(curr_child_link);

			for (NN_Link* p_next_parent : curr_child_link->parent->m_next) {
				if (p_next_parent->is_selected) {
					NN_Link* p_next_child = new NN_Link(p_next_parent);
					
					(*p_next_child)(curr_child_link);
					sel_child_links.push_back(p_next_child);

					for (NN_Link* p_output_parent : output) {
						if (p_output_parent == p_next_parent) {
							output_nodes.push_back(p_next_child);
						}
					}
				}
			}
		}
	}
	NN_Manager::clear_select();
}

const Dim NN_Model::calculate_output_size(const vector<Dim>& input_size) {

}

void NN_Model::build(const vector<Dim>& input_size) {

}

void NN_Model::run_forward(const vector<NN_Tensor>& inputs, NN_Tensor& output) {

}

void NN_Model::run_backward(const vector<NN_Tensor>& d_outputs, NN_Tensor& d_input) {

}

void NN_Model::compile(vector<NN_Loss*> loss, vector<NN_Optimizer*> optimizer) {

}

const NN_Tensor& NN_Model::train_on_batch(const vector<NN_Tensor>& samples, const vector<NN_Tensor>& truth) {

}

const NN_Tensor& NN_Model::fit(
	const vector<NN_Tensor>& samples,
	const vector<NN_Tensor>& truth,
	uint batch,
	uint iter
) {

}

const vector<NN_Tensor>& NN_Model::predict(const vector<NN_Tensor>& x) {

}