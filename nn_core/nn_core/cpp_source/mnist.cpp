#include "mnist.h"
#include <fstream>
#include <opencv2/opencv.hpp>


DataSet<uchar, uchar> MNIST::read_file(const std::string& img_path, const std::string& label_path) {
	std::ifstream img_fp(img_path, std::ios::binary);
	std::ifstream label_fp(label_path, std::ios::binary);

	if (!img_fp.is_open() || !label_fp.is_open()) {
		img_fp.close();
		label_fp.close();

		ErrorExcept(
			"[MNIST::read_file] failed load files."
		);
	}

	union TransByte {
		uchar _byte[4];
		int _byte32;
	}param;

	int img_head[4];
	int label_head[2];

	for (int i = 0; i < 4; ++i) {
		for (int j = 0; j < 4; ++j) {
			img_fp >> param._byte[3 - j];
		}
		img_head[i] = param._byte32;
	}

	for (int i = 0; i < 2; ++i) {
		for (int j = 0; j < 4; ++j) {
			label_fp >> param._byte[3 - j];
		}
		label_head[i] = param._byte32;
	}

	std::cout << "==============================================\n";
	std::cout << "image path: " << img_path << std::endl;
	std::cout << "magic num: " << std::hex << "0x" << img_head[0] << std::dec << std::endl;
	std::cout << "amounts: " << img_head[1] << std::endl;
	std::cout << "rows: " << img_head[2] << std::endl;
	std::cout << "colums: " << img_head[3] << std::endl;

	std::cout << "==============================================\n";
	std::cout << "label path: " << label_path << std::endl;
	std::cout << "magic num: " << std::hex << "0x" << label_head[0] << std::dec << std::endl;
	std::cout << "amounts: " << label_head[1] << std::endl;
	std::cout << std::endl;
	
	DataSet<uchar, uchar> samples;

	samples._x.resize({ img_head[1], img_head[2], img_head[3] });
	samples._y.resize({ img_head[1] });

	img_fp.read((char*)samples._x.get_ptr(), sizeof(uchar) * samples._x.get_shape().total_size());
	label_fp.read((char*)samples._y.get_ptr(), sizeof(uchar) * samples._y.get_shape().total_size());

	img_fp.close();
	label_fp.close();

	return samples;
}

MNIST::MNIST(const std::string& dir_path) {
	std::string train_x = dir_path + "\\train-images.idx3-ubyte";
	std::string train_y = dir_path + "\\train-labels.idx1-ubyte";
	std::string test_x = dir_path + "\\t10k-images.idx3-ubyte";
	std::string test_y = dir_path + "\\t10k-labels.idx1-ubyte";
	
	try {
		_train = read_file(train_x, train_y);
		_test = read_file(test_x, test_y);
	}
	catch (const NN_Exception& e) {
		e.put();
	}
}

Sample<uchar, uchar> MNIST::get_train_samples(int n_batch, int n_iter, bool shuffle) const {
	return Sample<uchar, uchar>(_train, n_batch, n_iter, shuffle);
}

Sample<uchar, uchar> MNIST::get_test_samples(int n_batch, int n_iter, bool shuffle) const {
	return Sample<uchar, uchar>(_test, n_batch, n_iter, shuffle);
}