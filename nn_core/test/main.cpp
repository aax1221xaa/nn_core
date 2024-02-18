#include "../nn_core/cpp_source/nn_tensor.h"
#include "vld.h"


int main() {
	List<uint> a({ 1, 2, 3 });
	List<uint> b({ 4, 5, 6 });
	List<uint> c({ 7, 8, 9 });

	List<uint> d({ a, b, c });

	for (const List<uint>& val : d) {
		if (val._list.size() > 0) {
			for (const List<uint>& v : val._list) std::cout << v._val << std::endl;
		}
		else std::cout << val._val << std::endl;
	}

	return 0;
}