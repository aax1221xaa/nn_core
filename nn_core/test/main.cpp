#include <iostream>

#define PUT_LIST

#include "../nn_core/cpp_source/nn_list.h"
#include "../nn_core/cpp_source/nn_tensor.h"

#ifdef _DEBUG
#include "vld.h"
#endif



int main() {
	try {
		NN_List<int> a;

		a.reserve(3);
		a[0].append(1);
		a[0].append(2);
		a[1].append({ 1, 2 });
		a[1].append({ 4, 5 });
		a[2].append({ 1, 2 });
		a[2].append(3);

		std::cout << a << std::endl;
	}
	catch (const NN_Exception& e) {
		e.put();

		return -1;
	}

	return 0;
}
