#include "mnist.h"
#include <fstream>
#include <opencv2/opencv.hpp>


MNIST::DataSet MNIST::read_file(const std::string& img_path, const std::string& label_path) {
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
	
	DataSet samples;

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
	catch (const Exception& e) {
		e.put();
	}
}

MNIST::Sample MNIST::get_train_samples(int n_batch, int n_iter, bool shuffle) const {
	return MNIST::Sample(_train, n_batch, n_iter, shuffle);
}

MNIST::Sample MNIST::get_test_samples(int n_batch, int n_iter, bool shuffle) const {
	return MNIST::Sample(_test, n_batch, n_iter, shuffle);
}

const MNIST::DataSet MNIST::Sample::get_batch_samples(const DataSet& origin, int index, int n_batch, bool shuffle) {
	DataSet sample;

	const NN_Shape& shape = origin._x.get_shape();

	const int amounts = shape[0];
	const int img_h = shape[1];
	const int img_w = shape[2];

	sample._x.resize({ n_batch, img_h, img_w });
	sample._y.resize({ n_batch });

	std::vector<int> batch_indice;

	if (shuffle) {
		batch_indice = random_choice(0, amounts, n_batch, false);
	}
	else {
		batch_indice.resize(n_batch);

		int start = (n_batch * index) % amounts;

		for (int i = 0; i < n_batch; ++i) {
			batch_indice[i] = (start + i) % amounts;
		}
	}

	sample._x = origin._x(batch_indice);

	return sample;
}

MNIST::Sample::Sample(const DataSet& current_samples, int n_batch, int n_iter, bool shuffle) :
	_origin(current_samples),
	_n_batch(n_batch),
	_n_iter(n_iter),
	_shuffle(shuffle)
{
}

typename MNIST::Sample::Iterator MNIST::Sample::begin() const {
	return MNIST::Sample::Iterator(*this, 0);
}

typename MNIST::Sample::Iterator MNIST::Sample::end() const {
	return MNIST::Sample::Iterator(*this, _n_iter);
}

const MNIST::DataSet MNIST::Sample::operator[](int index) const {
	if (index >= _n_iter) {
		ErrorExcept(
			"[MNIST::Sample::operator[]] Index[%d] is out of range. (%d ~ %d)",
			index,
			0, _n_iter
		);
	}

	return get_batch_samples(_origin, index, _n_batch, _shuffle);
}

MNIST::Sample::Iterator::Iterator(const Sample& samples, int n_iter) :
	_samples(samples),
	_n_iter(n_iter)
{
}

MNIST::Sample::Iterator::Iterator(const typename Iterator& p) :
	_samples(p._samples),
	_n_iter(p._n_iter)
{
}

bool MNIST::Sample::Iterator::operator!=(const typename Iterator& p) const {
	return _n_iter != p._n_iter;
}

void MNIST::Sample::Iterator::operator++() {
	++_n_iter;
}

const MNIST::DataSet MNIST::Sample::Iterator::operator*() const {
	return get_batch_samples(_samples._origin, _n_iter, _samples._n_batch, _samples._shuffle);
}