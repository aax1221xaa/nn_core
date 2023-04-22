#define PUT_LIST

#include "../nn_core/cpp_source/nn_tensor.h"
#include "vld.h"



int main() {
	try {
		Tensor<int> tensor = zeros<int>({ 3, 5 });
		nn_shape indices(2, 0);

		for (; indices[0] < tensor._shape[0]; ++indices[0]) {
			for (; indices[1] < tensor._shape[1]; ++indices[1]) {
				tensor.get(indices) = 1;
			}
			indices[1] = 0;
		}

		NN_Tensor<int> nn_tensor = NN_Tensor<int>::zeros(tensor._shape);
		copy_to_nn_tensor(tensor, nn_tensor);

		Tensor<int> tensor_2 = zeros<int>(tensor._shape);
		copy_to_tensor(nn_tensor, tensor_2);

		std::cout << tensor_2;
	}
	catch (Exception& p) {
		p.Put();
	}

	return 0;
}