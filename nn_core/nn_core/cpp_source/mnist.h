#pragma once

#include "../cpp_source/nn_tensor.h"
#include <opencv2/opencv.hpp>


class MNIST {
public:
	Tensor<uchar> train_x;
	Tensor<uchar> train_y;
	Tensor<uchar> test_x;
	Tensor<uchar> test_y;

	int _batch;

	MNIST(const char* dir, int batch);

	static void load_file(const char* image_file, const char* label_file, Tensor<uchar>& image, Tensor<uchar>& truth);
};