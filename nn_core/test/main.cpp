#include <iostream>

#define PUT_LIST

#include "../nn_core/cpp_source/nn_list.h"
#include "../nn_core/cpp_source/nn_tensor.h"

#ifdef _DEBUG
#include "vld.h"
#endif



int main() {
	try {
		NN_List<int> a = { 1, 2, 3 };

		std::cout << a;

		a = { 4, 5, 6 };

		std::cout << a;
	}
	catch (const NN_Exception& e) {
		e.put();

		return -1;
	}

	return 0;
}
