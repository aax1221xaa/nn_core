#include "../nn_core/cpp_source/nn_tensor.h"
#include "../nn_core/cuda_source/convolution.cuh"
#include <time.h>
#include "tbb/tbb.h"
#include "vld.h"

using namespace tbb;

int main() {
	try {
		Tensor<float> input({ 1, 2, 5, 5 });
		Tensor<float> output({ 1, 2, 5, 5 });


		//std::random_device rd;
		//std::mt19937 gen(rd());
		//std::uniform_real_distribution<float> dis(-0.1f, 0.1f);

		tbb::parallel_for(tbb::blocked_range<unsigned int>(0, input._len),
			[&](const tbb::blocked_range<unsigned int>& p) {

			for (unsigned int i = p.begin(); i < p.end(); ++i) input._data[i] = 0.1f;
		});

		NN_Tensor<float> d_input({ 1, 2, 5, 5 });
		NN_Tensor<float> d_pad({ STREAMS, 2, 7, 7 });
		NN_Tensor<float> weight({ 2, 2, 3, 3 });
		NN_Tensor<float> d_output({ 1, 2, 5, 5 });

		set_uniform(weight);
		check_cuda(cudaMemcpy(d_input._data, input._data, input._elem_size * input._len, cudaMemcpyHostToDevice));
		check_cuda(cudaMemset(d_pad._data, 0, d_pad._elem_size * d_pad._len));

		CudaTensor din(
			d_input._data,
			d_input._shape[0],
			d_input._shape[1],
			d_input._shape[2],
			d_input._shape[3]
		);
		CudaTensor dpad(
			d_pad._data,
			d_pad._shape[0],
			d_pad._shape[1],
			d_pad._shape[2],
			d_pad._shape[3]
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

		cudaStream_t streams[STREAMS] = { NULL, };
		for (int i = 0; i < STREAMS; ++i) check_cuda(cudaStreamCreate(&streams[i]));

		clock_t start = clock();
		padding_conv_2d(
			streams,
			din,
			dpad,
			dk,
			dout,
			1, 1
		);
		for (int i = 0; i < STREAMS; ++i) check_cuda(cudaStreamSynchronize(streams[i]));
		clock_t end = clock();

		printf("elapsed time= %ldms\n", end - start);
		
		check_cuda(cudaMemcpy(output._data, d_output._data, d_output._elem_size * d_output._len, cudaMemcpyDeviceToHost));
		
		std::cout << output;

		for (int i = 0; i < STREAMS; ++i) check_cuda(cudaStreamDestroy(streams[i]));
	}
	catch (Exception& p) {
		p.Put();
	}

	return 0;
}