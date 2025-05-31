#include <iostream>
#include "./CppSource/Exception.h"
#include "./CppSource/nn_tensor_plus.h"
#include "./CudaSource/nn_concat.cuh"


#ifdef _DEBUG
#include "vld.h"
#endif


int main() {
	try {
		NN_List<int> test;

		test.resize(3);
		std::cout << test << std::endl;

		test[1] = 5;
		std::cout << test << std::endl;
	}
	catch (const NN_Exception& e) {
		e.put();
	}

	return 0;
}
