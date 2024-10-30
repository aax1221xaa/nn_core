#include <iostream>

#define PUT_LIST

#include "../nn_core/cpp_source/nn_list.h"
#include "../nn_core/cpp_source/nn_tensor.h"

#ifdef _DEBUG
#include "vld.h"
#endif



int main() {
	try {
		Tensor<int> a = Tensor<int>::zeros({ 3, 3, 3 });
		Tensor<int> b = Tensor<int>::zeros({ 2, 2, 2 });

		b(0, 2)(0, 2)(0, 2) = 1;
		a(0, 2)(0, 2)(0, 2) = b;

		std::cout << a << std::endl;
	}
	catch (const NN_Exception& e) {
		e.put();

		return -1;
	}

	return 0;
}
