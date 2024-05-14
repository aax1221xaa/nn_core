#include "../nn_core/cpp_source/nn_tensor.h"


#ifdef _DEBUG
#include "vld.h"
#endif



int main() {
	try {
		Tensor<int> test(NN_Shape({ 5, 5, 5 }));

		test = 0;
		test(1, 4)(1, 4)(1, 4) = 1;

		tensor_t<int> data = test.get_data();

		for (int c = 0; c < 5; ++c) {
			for (int h = 0; h < 5; ++h) {
				for (int w = 0; w < 5; ++w) {
					std::cout << data[c * 25 + h * 5 + w] << ' ';
				}
				std::cout << std::endl;
			}
			std::cout << std::endl;
		}
	}
	catch (const Exception& e) {
		e.put();
	}

	return 0;
}

