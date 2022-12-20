#include "nn_manager.h"


bool NN_Manager::init_flag = false;
vector<NN_Ptr<NN_Link>> NN_Manager::links;

void NN_Manager::add_link(NN_Ptr<NN_Link>& link) {
	links.push_back(link);
}

void NN_Manager::clear_links() {
	links.clear();
}

void NN_Manager::clear_select_flag() {
	for (NN_Ptr<NN_Link>& p : links) {
		p->is_selected = false;
	}
}

vector<NN_Ptr<NN_Link>>& NN_Manager::get_links() {
	return links;
}

void NN_Manager::set_linked_count() {
	for (NN_Ptr<NN_Link>& p : links) {
		NN_Link::set_linked_count(p);
	}
}

NN_Manager::NN_Manager() {
	init_flag = true;
}

NN_Manager::~NN_Manager() {
	clear_links();
}