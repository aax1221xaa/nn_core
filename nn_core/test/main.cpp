#include "../nn_core/cpp_source/cuda_common.h"
#include "vld.h"



int main() {
	List<int> test_1(1);
	List<int> test_2({ 1, 2, 3 });
	List<int> test_4({ 7, 8, 9 });
	List<int> test_3({ test_1, test_2, {4, 5, 6, test_4} });

	std::cout << test_1;
	std::cout << test_2;
	std::cout << test_3;

	std::cout << "========================================" << std::endl;

	try {
		for (const List<int>& p : test_1) std::cout << p;
	}
	catch (Exception& p) {
		p.Put();
	}

	return 0;
}