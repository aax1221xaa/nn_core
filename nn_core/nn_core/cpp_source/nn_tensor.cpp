#include "nn_tensor.h"
#include <random>


#ifdef FIX_MODE

void set_uniform(NN_Tensor<nn_type>& p) {
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<float> dis(-0.1f, 0.1f);

	size_t size = p._len;

	float* tmp = new nn_type[size];

	for (size_t i = 0; i < size; ++i) tmp[i] = 0.1f;
	check_cuda(cudaMemcpy(p._data, tmp, sizeof(nn_type) * size, cudaMemcpyHostToDevice));

	delete[] tmp;
}

#endif

#ifndef FIX_MODE
void set_uniform(NN_Tensor<nn_type>& p) {
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<float> dis(-0.1f, 0.1f);

	size_t size = p._len;

	float* tmp = new float[size];

	for (size_t i = 0; i < size; ++i) tmp[i] = dis(gen);
	check_cuda(cudaMemcpy(p._data, tmp, sizeof(float) * size, cudaMemcpyHostToDevice));

	delete[] tmp;
}

#endif