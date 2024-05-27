#include "../nn_core/cpp_source/nn_tensor.h"
#include <opencv2/opencv.hpp>

#ifdef _DEBUG
#include "vld.h"
#endif


void xavier(Tensor<nn_type>& tensor, nn_type fan_in, nn_type fan_out) {
	std::random_device rd;
	cv::RNG rng(rd());
	cv::Mat mat(tensor.get_shape().get_vector(), CV_32FC1);
	
	rng.fill(mat, cv::RNG::UNIFORM, -0.1f, 0.1f, false);
	//mat *= std::sqrtf(fan_in);

	const size_t size = tensor.get_shape().total_size();
	nn_type* p_tensor = tensor.get_ptr();
	nn_type* p_mat = mat.ptr<nn_type>(0);

	for (size_t i = 0; i < size; ++i) {
		p_tensor[i] = p_mat[i];
	}
}


int main() {
	try {
		Tensor<nn_type> test(NN_Shape({ 5, 5 }));

		xavier(test, 5, 5);

		std::cout << test;
		
	}
	catch (const Exception& e) {
		e.put();
	}

	return 0;
}

