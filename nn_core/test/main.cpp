#include "cuda_kernel.cuh"
#include "../nn_core/cpp_source/cuda_common.h"
#include "vld.h"



int main() {
	List<int> test_1(1);
	cout << "==============================" << endl;
	List<int> test_2({ 1, 2, 3 });
	cout << "==============================" << endl;
	List<int> test_4({ 7, 8, 9 });
	cout << "==============================" << endl;
	List<int> test_3({ test_1, test_2, {4, 5, 6, test_4} });
	cout << "==============================" << endl;
	cout << test_1;
	cout << test_2;
	cout << test_3;

	return 0;
}