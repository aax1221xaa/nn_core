#include "../nn_core/cpp_source/nn_tensor.h"


#ifdef _DEBUG
#include "vld.h"
#endif



int main() {
	try {
		Tensor<int> test(NN_Shape({ 3, 3, 3 }));

		for (int c = 0; c < 3; ++c) {
			for (int h = 0; h < 3; ++h) {
				for (int w = 0; w < 3; ++w) {
					test[c][h][w] = 1;
				}
			}
		}

		tensor_t<int> data = test.get_data();

		for (int c = 0; c < 3; ++c) {
			for (int h = 0; h < 3; ++h) {
				for (int w = 0; w < 3; ++w) {
					std::cout << data[c * 9 + h * 3 + w] << ' ';
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

