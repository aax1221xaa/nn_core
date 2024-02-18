#pragma once

#include "../cpp_source/nn_tensor.h"
#include <opencv2/opencv.hpp>


class MNIST {
	static void load_file(const char* image_file, const char* label_file, Tensor<uchar>& image, Tensor<uchar>& truth);

public:
	Tensor<uchar> train_x;
	Tensor<uchar> train_y;
	Tensor<uchar> test_x;
	Tensor<uchar> test_y;

	bool _do_shuffle;

	int _batch;

	int _train_max_iter;
	int _test_max_iter;

	int _train_iter_cnt;
	int _test_iter_cnt;

	MNIST(const char* dir, int batch, bool do_shuffle = true);

	//std::vector<Tensor<uchar>> train_iter();
	//std::vector<Tensor<uchar>> test_iter();
};