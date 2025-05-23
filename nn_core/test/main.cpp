#include <iostream>
#include "./CppSource/nn_tensor_plus.h"
#include "./CudaSource/convolution.cuh"



#ifdef _DEBUG
#include "vld.h"
#endif


int main() {
	try {
		NN_Stream st;

		Tensor<nn_type> h_src({1, 5, 5, 3 });
		Tensor<nn_type> h_dst({1, 5, 5, 3 });
		Tensor<nn_type> h_filter({ 3, 3, 3, 3 });

		h_src = 1;
		h_filter = 1;

		GpuTensor<nn_type> g_src = h_src;
		GpuTensor<nn_type> g_dst(h_dst.get_shape());
		GpuTensor<nn_type> g_filter = h_filter;

		conv2d(
			st,
			g_src,
			g_filter,
			g_dst,
			"same",
			{ 1, 1 }
		);

		h_dst = g_dst;

		std::cout << h_dst[0].transpose({ 2, 0, 1 });
	}
	catch (const NN_Exception& e) {
		e.put();
	}

	return 0;
}
