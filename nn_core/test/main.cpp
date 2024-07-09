#include <iostream>

#define PUT_LIST

#include "../nn_core/cpp_source/nn_list.h"
#include "../nn_core/cpp_source/nn_tensor.h"

#ifdef _DEBUG
#include "vld.h"
#endif



int main() {
	try {
		Tensor<int> a({ 3, 3 });

		a = 1;
		std::cout << a;

		a.expand_dims(1);

		std::cout << a;
	}
	catch (const NN_Exception& e) {
		e.put();

		return -1;
	}

	return 0;
}
