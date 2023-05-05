#include "../nn_core/cpp_source/nn_tensor.h"
#include "../nn_core/cuda_source/convolution.cuh"
#include <time.h>
#include "tbb/tbb.h"
#include "vld.h"

using namespace tbb;

int main() {
	try {

		Tensor<float> input({ 1, 2, 28, 28 });
		Tensor<float> output({ 1, 2, 26, 26 });

		
		//std::random_device rd;
		//std::mt19937 gen(rd());
		//std::uniform_real_distribution<float> dis(-0.1f, 0.1f);

		tbb::parallel_for(tbb::blocked_range<unsigned int>(0, input._len),
			[&](const tbb::blocked_range<unsigned int>& p) {

			for (unsigned int i = p.begin(); i < p.end(); ++i) input._data[i] = 1.f;
		});
		

		for (size_t i = 0; i < input._len; ++i) input._data[i] = 1.f;

		NN_Tensor<float> d_input({ 1, 2, 28, 28 });
		NN_Tensor<float> weight({ 2, 2, 3, 3 });
		NN_Tensor<float> d_output({ 1, 2, 26, 26 });

		set_uniform(weight);
		check_cuda(cudaMemcpy(d_input._data, input._data, input._elem_size * input._len, cudaMemcpyHostToDevice));

		uint* indice = new uint[3 * 3];
		for (uint y = 0; y < 3; ++y) {
			for (uint x = 0; x < 3; ++x) {
				indice[y * 3 + x] = y * 28 + x;
			}
		}
		copy_to_indice(indice, sizeof(uint) * 3 * 3, 0);
		delete[] indice;

		CudaTensor din(
			d_input._data,
			d_input._shape[0],
			d_input._shape[1],
			d_input._shape[2],
			d_input._shape[3]
		);
		CudaTensor dk(
			weight._data,
			weight._shape[0],
			weight._shape[1],
			weight._shape[2],
			weight._shape[3]
		);
		CudaTensor dout(
			d_output._data,
			d_output._shape[0],
			d_output._shape[1],
			d_output._shape[2],
			d_output._shape[3]
		);

		clock_t start = clock();
		conv_2d(
			NULL,
			din,
			dk,
			dout,
			1, 1,
			0
		);
		check_cuda(cudaMemcpy(output._data, d_output._data, d_output._elem_size * d_output._len, cudaMemcpyDeviceToHost));

		std::cout << output;
		
	}
	catch (Exception& p) {
		p.Put();
	}

	return 0;
}