#include <iostream>

#define PUT_LIST

#include "../nn_core/cpp_source/gpu_tensor_misc.h"
#include "../nn_core/cuda_source/nn_operators.cuh"

#ifdef _DEBUG
#include "vld.h"
#endif



int main() {
	try {
		Tensor<nn_type> _a({ 5, 5 });
		Tensor<nn_type> _b({ 5, 5 });
		Tensor<nn_type> _c({ 5, 5 });

		_a.all() = 1.f;
		_b.all() = 2.f;
		_c.all() = 0.f;

		GpuTensorManager nn;
		GpuTensor<nn_type> a, b, c;

		nn.do_record(true);

		a = _a;
		b = _b;
		c = nn(Add(), a, b);
		_c = c;

		nn.do_record(false);

		std::cout << _c;

	}
	catch (const NN_Exception& e) {
		e.put();

		return -1;
	}

	return 0;
}
