#include "../nn_core/cpp_source/nn_tensor.h"
#include "../nn_core/cuda_source/convolution.cuh"
#include <time.h>
#include "tbb/tbb.h"
#include "vld.h"

using namespace tbb;

int main() {
	try {

		Tensor<float> input({ 128, 3, 28, 28 });
		Tensor<float> output({ 128, 32, 26, 26 });

		
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_real_distribution<float> dis(-0.1f, 0.1f);

		tbb::parallel_for(tbb::blocked_range<unsigned int>(0, input._len),
			[&](const tbb::blocked_range<unsigned int>& p) {

			for (unsigned int i = p.begin(); i < p.end(); ++i) input._data[i] = dis(gen);
		});
		

		memset(input._data, 1, input._len * input._elem_size);

		NN_Tensor<float> d_input({ 128, 3, 28, 28 });
		NN_Tensor<float> weight({ 32, 3, 3, 3 });
		NN_Tensor<float> d_output({ 128, 32, 26, 26 });

		set_uniform(weight);
		check_cuda(cudaMemcpy(d_input._data, input._data, input._elem_size * input._len, cudaMemcpyHostToDevice));

		cudaStream_t* s = new cudaStream_t[8];

		for (int i = 0; i < 8; ++i) check_cuda(cudaStreamCreate(&s[i]));

		CudaTensor din(
			d_input._data,
			d_input._shape[0],
			d_input._shape[2],
			d_input._shape[3],
			d_input._shape[1]
		);
		CudaTensor dk(
			weight._data,
			weight._shape[0],
			weight._shape[2],
			weight._shape[3],
			weight._shape[1]
		);
		CudaTensor dout(
			d_output._data,
			d_output._shape[0],
			d_output._shape[2],
			d_output._shape[3],
			d_output._shape[1]
		);

		clock_t start = clock();
		conv_2d(
			s,
			8,
			din,
			dk,
			dout,
			1, 1
		);
		check_cuda(cudaDeviceSynchronize());
		clock_t end = clock();

		check_cuda(cudaMemcpy(output._data, d_output._data, d_output._elem_size * d_output._len, cudaMemcpyDeviceToHost));
		
		for (int i = 0; i < 8; ++i) check_cuda(cudaStreamDestroy(s[i]));
		delete[] s;

		printf("elapsed time = %ldms\n", end - start);

		//std::cout << output;
		
	}
	catch (Exception& p) {
		p.Put();
	}

	return 0;
}