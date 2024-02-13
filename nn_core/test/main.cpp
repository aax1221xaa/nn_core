#include "../nn_core/cpp_source/cuda_common.h"
#include "vld.h"


int main() {
	List<uint> a({ 1, 2, 3, 4, 5 });
	
	for (const List<uint>& val : a) {
		std::cout << val;
	}

	return 0;
}