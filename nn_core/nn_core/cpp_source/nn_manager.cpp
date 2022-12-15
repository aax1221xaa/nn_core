#include "nn_manager.h"


bool NN_Manager::init_flag = false;
vector<NN_Ptr<NN_Link>> NN_Manager::links;

void NN_Manager::add_link(NN_Ptr<NN_Link>& link) {
	links.push_back(link);
}

void NN_Manager::clear_links() {
	links.clear();
}

void NN_Manager::clear_select_flags() {
	for (NN_Ptr<NN_Link>& p : links) {
		p->is_selected = false;
	}
}

NN_Manager::NN_Manager() {
	init_flag = true;
}

NN_Manager::~NN_Manager() {
	clear_links();
}