#pragma once

#include "../cpp_source/nn_tensor.h"
#include <opencv2/opencv.hpp>


class MNIST {
	static void load_file(const char* image_file, const char* label_file, HostTensor<uchar>& image, HostTensor<uchar>& truth);

public:
	HostTensor<uchar> train_x;
	HostTensor<uchar> train_y;
	HostTensor<uchar> test_x;
	HostTensor<uchar> test_y;

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