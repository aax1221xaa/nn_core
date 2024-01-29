#include "nn_tensor.h"
#include <random>


#ifdef FIX_MODE

/**********************************************/
/*                                            */
/*                 TensorBase                 */
/*                                            */
/**********************************************/

TensorBase::TensorBase() {

}

TensorBase::TensorBase(const nn_shape& shape) :
	_shape(shape)
{
}

size_t TensorBase::get_len() const {
	size_t n = 1;

	for (cint i : _shape) {
		if (i < 1) {
			ErrorExcept(
				"Invalid shape. %s",
				put_shape(_shape)
			);
		}
		n *= i;
	}

	return n;
}

void set_uniform(DeviceTensor<nn_type>& p) {
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<float> dis(-0.1f, 0.1f);

	size_t size = p.get_len();

	float* tmp = new nn_type[size];

	for (size_t i = 0; i < size; ++i) tmp[i] = dis(gen);
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