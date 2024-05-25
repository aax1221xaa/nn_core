#pragma once

#include "nn_sample.h"
#include <opencv2/opencv.hpp>


class MNIST {
private:
	DataSet<uchar, uchar> _train, _test;

	static DataSet<uchar, uchar> read_file(const std::string& img_path, const std::string& label_path);

public:
	MNIST(const std::string& dir_path);

	Sample<uchar, uchar> get_train_samples(int n_batch, int n_iter, bool shuffle = true) const;
	Sample<uchar, uchar> get_test_samples(int n_batch, int n_iter, bool shuffle = true) const;
};