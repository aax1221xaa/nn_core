#include "nn_manager.h"


bool NN_Manager::init_flag = false;
vector<NN_Link*> NN_Manager::reg_links;

void NN_Manager::add_link(NN_Link* link) {
	reg_links.push_back(link);
}

void NN_Manager::clear_links() {
	for (NN_Link* p : reg_links) delete p;
	reg_links.clear();
}

void NN_Manager::clear_select_flag() {
	for (NN_Link* p : reg_links) {
		p->is_selected = false;
	}
}

vector<NN_Link*> NN_Manager::get_links() {
	return reg_links;
}



NN_Manager::NN_Manager() {
	init_flag = true;
}

NN_Manager::~NN_Manager() {
	clear_links();
}