#include <iostream>

#define PUT_LIST

#include "../nn_core/cpp_source/nn_list.h"

#ifdef _DEBUG
#include "vld.h"
#endif



int main() {
	try {
		NN_List<int> test;

		test.resize(1);

		for (NN_List<int>& n : test) std::cout << n;
	}
	catch (const NN_Exception& e) {
		e.put();

		return -1;
	}

	return 0;
}
