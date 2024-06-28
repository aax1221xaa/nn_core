#include <iostream>
#include "../nn_core/cpp_source/nn_list.h"

#ifdef _DEBUG
#include "vld.h"
#endif



int main() {
	try {
		NN_List<int> test(NN_List<int>({ 1, 2, 3 }));

		std::cout << test;
	}
	catch (const NN_Exception& e) {
		e.put();

		return -1;
	}

	return 0;
}
