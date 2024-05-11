#include "../nn_core/cpp_source/nn_tensor.h"


#ifdef _DEBUG
#include "vld.h"
#endif



int main() {
	try {
		Tensor<int> test(NN_Shape({ 5, 5, 5 }));

		std::cout << test.get_shape();

		std::shared_ptr<int[]>& n = test.get_data();

		for (int i = 0; i < 125; ++i) n[i] = 0;

		test(1, 4)(1, 4)(1, 4) = 1;

		for (int k = 0; k < 5; ++k) {
			for (int i = 0; i < 5; ++i) {
				for (int j = 0; j < 5; ++j) {
					std::cout << n[k * 25 + i * 5 + j] << ' ';
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

