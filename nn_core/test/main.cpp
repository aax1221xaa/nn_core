#include <iostream>
#include <opencv2/opencv.hpp>
#include "../nn_core/cpp_source/Exception.h"
#include "../nn_core/cuda_source/convolution.cuh"

#ifdef _DEBUG
#include "vld.h"
#endif



int main() {
	try {
		Tensor<nn_type> in_data({ 4, 1, 10, 10 });
		Tensor<nn_type> k_data({ 3, 3, 1, 3 });
		Tensor<nn_type> out_data({ 4, 8, 8, 3 });
		
		for (int n = 0; n < 4; ++n) {
			for (int c = 0; c < 1; ++c) {
				for (int h = 0; h < 10; ++h) {
					for (int w = 0; w < 10; ++w) {
						in_data[n][c][h][w] = (c + h) % 2 == 0 ? (nn_type)(w % 2) : (nn_type)((w + 1) % 2);
					}
				}
			}
		}
		
		k_data = 1.f;

		GpuTensor<nn_type> din_data = in_data.transpose({ 0, 2, 3, 1 });
		GpuTensor<nn_type> dk_data = k_data;
		GpuTensor<nn_type> dout_data(NN_Shape({ 4, 8, 8, 3 }));

		NN_Stream streams;

		conv_test(streams, din_data, dk_data, NN_Shape({ 1, 1 }), "valid", dout_data);
		out_data = dout_data;

		std::cout << in_data << std::endl;
		std::cout << out_data.transpose({ 0, 3, 1, 2 });
	}
	catch (const NN_Exception& e) {
		e.put();

		return -1;
	}

	return 0;
}
