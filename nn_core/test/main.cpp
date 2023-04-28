#define PUT_LIST

#include "../nn_core/cpp_source/nn_tensor.h"
#include "vld.h"



int main() {
	try {
		CPU_Tensor test = CPU_Tensor::zeros({ 5, 5 }, DType::int32);
		CPU_Tensor test2 = test.slice({ 1, 1 }, { 4, 4 });
		for (int i = 0; i < test._len; ++i) ((int*)test._data)[i] = i;

		std::cout << test;

		std::cout << "shape ";
		std::cout << dimension_to_str(test2._shape) << std::endl;

		std::cout << "steps ";
		std::cout << '[';
		for (uint n : test2._steps) {
			std::cout << n << ", ";
		}
		std::cout << ']' << std::endl;

		std::cout << "start ";
		std::cout << '[';
		for (uint n : test2._start) {
			std::cout << n << ", ";
		}
		std::cout << ']' << std::endl;
		
		std::cout << "offset ";
		std::cout << test2._offset << std::endl;
	}
	catch (Exception& p) {
		p.Put();
	}

	return 0;
}