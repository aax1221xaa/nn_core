#include "../nn_core/cpp_source/nn_tensor.h"
#include <H5Cpp.h>

#include <Windows.h>

#ifdef _DEBUG
#include "vld.h"
#endif



int main() {
	SetConsoleOutputCP(65001);
	std::cout.precision(3);

	Tensor<int> tensor(NN_Shape({ 2, 3, 4, 5 }));
	int* p = tensor.get_ptr();

	for (int i = 0; i < 120; ++i) p[i] = i;

	std::cout << tensor;

	Tensor<int> tensor2 = tensor.transpose({ 1, 0, 2, 3 });

	std::cout << tensor2;

	return 0;
}
