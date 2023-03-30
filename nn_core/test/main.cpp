#include "cuda_kernel.cuh"
#include "../nn_core/cpp_source/cuda_common.h"
#include "vld.h"


int main() {
	List<int> list({ 0, {1, 2}, {3, 4}, 5, {7} });

	cout << list;

	return 0;
}