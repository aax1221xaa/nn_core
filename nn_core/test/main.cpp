#define PUT_LIST

#include "../nn_core/cpp_source/cuda_common.h"
#include "vld.h"



int main() {
	try {
		List<int> test;

		test.push_back(1);
		test.push_back(2);
		test.push_back(3);

		for (const List<int>& p : test) std::cout << p;
	}
	catch (Exception& p) {
		p.Put();
	}

	return 0;
}