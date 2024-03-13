#include "../nn_core/cpp_source/nn_tensor.h"
#include "vld.h"


int main() {
	try {
		Tensor<float> a({ 5, 5 });
		Tensor<int> b = zeros_like<int>(a);

		GpuTensor<int> c({ 5, 5 });
		GpuTensor<float> d = gpu_zeros_like<float>(c);
	}
	catch (const Exception& e) {
		e.Put();
	}

	return 0;
}