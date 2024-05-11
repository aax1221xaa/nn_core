#include "../nn_core/cpp_source/nn_tensor.h"


#ifdef _DEBUG
#include "vld.h"
#endif



int main() {
	try {
		Tensor<int> test({ 5, 5, 5 });
		Tensor<int> test2({ 3, 3 });

		tensor_t<int>& val1 = test.get_data();
		tensor_t<int>& val2 = test2.get_data();

		printf("%s\n", shape_to_str(test2.get_shape()));

		for (int i = 0; i < 125; ++i) val1[i] = 0;
		for (int i = 0; i < 9; ++i) val2[i] = i / 3;

		test(1, 4)(1, 4)(1, 4) = test2;

		for (int k = 0; k < 5; ++k) {
			for (int i = 0; i < 5; ++i) {
				for (int j = 0; j < 5; ++j) {
					std::cout << val1[k * 25 + i * 5 + j] << ' ';
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

