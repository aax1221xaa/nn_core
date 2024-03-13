#include "nn_tensor.h"
#include <opencv2/opencv.hpp>


/**********************************************/
/*                                            */
/*					  Random                  */
/*                                            */
/**********************************************/

void set_random_uniform(Tensor<float>& tensor, float a, float b) {
	if (tensor.get_shape().size() == 0) return;

	cv::RNG rng(cv::getTickCount());
	cv::Mat mat(tensor.get_shape(), CV_32FC1, tensor.get_data());

	rng.fill(mat, cv::RNG::UNIFORM, a, b);
}

void set_random_uniform(GpuTensor<float>& tensor, float a, float b) {
	if (tensor.get_shape().size() == 0) return;

	Tensor<float> tmp(tensor.get_shape());

	cv::RNG rng(cv::getTickCount());
	cv::Mat mat(tmp.get_shape(), CV_32FC1, tmp.get_data());

	rng.fill(mat, cv::RNG::UNIFORM, a, b);

	host_to_gpu(tmp, tensor);
}