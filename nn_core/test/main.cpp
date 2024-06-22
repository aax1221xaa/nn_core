#include "../nn_core/cuda_source/cuda_misc.cuh"

#include <Windows.h>

#ifdef _DEBUG
#include "vld.h"
#endif



int main() {
	SetConsoleOutputCP(65001);
	std::cout.precision(3);

	Tensor<nn_type> A({ 2, 3, 4 });
	GpuTensor<nn_type> dA({ 2, 3, 4 });

	Tensor<nn_type> B({ 4, 3, 2 });
	GpuTensor<nn_type> dB({ 4, 3, 2 });

	nn_type* pA = A.get_ptr();
	
	for (int i = 0; i < 24; ++i) pA[i] = (float)i;

	dA = A;

	transpose(dA, dB, { 2, 1, 0 });

	B = dB;

	std::cout << A;
	std::cout << std::endl;
	std::cout << B;

	return 0;
}
