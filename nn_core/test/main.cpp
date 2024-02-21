#include "../nn_core/cpp_source/cuda_common.h"
#include "vld.h"


int main() {
	List<int> a;

	a.resize(10);

	for (int i = 0; i < 10; ++i) a[i] = i;

	for (const List<int>& b : a) std::cout << b._val << std::endl;

	return 0;
}