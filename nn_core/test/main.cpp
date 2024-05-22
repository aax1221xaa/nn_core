#include "../nn_core/cpp_source/nn_tensor.h"


#ifdef _DEBUG
#include "vld.h"
#endif



int main() {
	try {
		Tensor<int> test(NN_Shape({ 3, 5, 5 }));
		GpuTensor<int> test2(NN_Shape({ 3, 5, 5 }));
		Tensor<int> test3(NN_Shape({ 3, 5, 5 }));

		test = 3;
		test2 = test;
		test3 = test2;

		std::cout << test3;
	}
	catch (const Exception& e) {
		e.put();
	}

	return 0;
}

