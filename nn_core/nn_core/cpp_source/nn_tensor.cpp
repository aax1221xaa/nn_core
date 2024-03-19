#include "nn_tensor.h"
#include <opencv2/opencv.hpp>



bool check_shape(const NN_Shape& shape) {
	bool is_valid = true;

	if (shape.get_size() > 0) {
		for (const int& n : shape) {
			if (n < 1) is_valid = false;
		}
	}
	else is_valid = false;

	return is_valid;
}

size_t calculate_length(const NN_Shape& shape) {
	size_t len = 1;

	for (const int& n : shape) len *= n;

	return len;
}

/**********************************************/
/*                                            */
/*					  Random                  */
/*                                            */
/**********************************************/

void set_random_uniform(Tensor<float>& tensor, float a, float b) {
	if (tensor.get_shape().get_size() == 0) return;

	cv::RNG rng(cv::getTickCount());
	cv::Mat mat(tensor.get_shape().get_size(), tensor.get_shape().get_dims(), CV_32FC1, tensor.get_data());

	rng.fill(mat, cv::RNG::UNIFORM, a, b);
}

void set_random_uniform(GpuTensor<float>& tensor, float a, float b) {
	if (tensor.get_shape().get_size() == 0) return;

	Tensor<float> tmp(tensor.get_shape());

	cv::RNG rng(cv::getTickCount());
	cv::Mat mat(tmp.get_shape().get_size(), tmp.get_shape().get_dims(), CV_32FC1, tmp.get_data());

	rng.fill(mat, cv::RNG::UNIFORM, a, b);

	host_to_gpu(tmp, tensor);
}