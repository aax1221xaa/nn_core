#include "cuda_kernel.cuh"
#include "../nn_core/cpp_source/Dimension.h"
#include "vld.h"


int main() {
	NN_Shape shape({ 1, 2, 3, 4, 5 });
	printf("%s\n", shape.get_str());

	return 0;
}