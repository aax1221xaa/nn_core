#pragma once

#include "nn_sample.h"
#include <opencv2/opencv.hpp>



template <typename _xT, typename _yT>
class DataSetType {
public:
	typedef _xT x_type;
	typedef _yT y_type;
};


class MNIST : public DataSet<uchar, uchar> {
public:
	DataSet<uchar, uchar> _train, _test;

	static DataSet<uchar, uchar> read_file(const std::string& img_path, const std::string& label_path);

public:
	MNIST(const std::string& dir_path);

	std::vector<DataSet<uchar, uchar>> get_samples();
	Sample<uchar, uchar> get_train_samples(int n_batch, int n_iter, bool shuffle = true);
	Sample<uchar, uchar> get_test_samples(int n_batch, int n_iter, bool shuffle = true);
	const std::vector<DataSet<uchar, uchar>> get_samples() const;
	const Sample<uchar, uchar> get_train_samples(int n_batch, int n_iter, bool shuffle = true) const;
	const Sample<uchar, uchar> get_test_samples(int n_batch, int n_iter, bool shuffle = true) const;
};