#include "../nn_core/cpp_source/cuda_common.h"
#include "vld.h"


int main() {
	try {
		std::vector<int> a = { 1, 2, 3, 4 ,5 };
		std::vector<int> b;

		b = a;

		a.push_back(6);
		b.push_back(7);

		for (const int& val : a) std::cout << val << std::endl;
		for (const int& val : b) std::cout << val << std::endl;

	}
	catch (const Exception& e) {
		e.Put();
	}

	return 0;
}