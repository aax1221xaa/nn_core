#include "nn_manager.h"


Object_Linker NN_Manager::linker;

void NN_Manager::create() {
	id = linker.Create();
}

void NN_Manager::destroy() {

}

void NN_Manager::clear_param() {
	id = NULL;
}

NN_Manager::NN_Manager() {
	clear_param();
}

NN_Manager::NN_Manager(const NN_Manager& p) {
	copy_id(p);
}

NN_Manager::~NN_Manager() {
	clear();
}

void NN_Manager::clear() {
	if (id) {
		if (id->nCpy > 1) --id->nCpy;
		else {
			destroy();
			linker.Erase(id);
			clear_param();
		}
	}
}

void NN_Manager::copy_id(const NN_Manager& p) {
	id = p.id;

	if (id) {
		++id->nCpy;
	}
}