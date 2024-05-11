#include "mnist.h"



MNIST::DataSet MNIST::read_file(const std::string& img_path, const std::string& label_path) {
	FILE* img_fp = NULL;
	FILE* label_fp = NULL;

	errno_t err = fopen_s(&img_fp, img_path.c_str(), "rb");
	if (err < 0) {
		ErrorExcept(
			"[MNIST::read_file] failed load image file [%s]",
			img_path.c_str()
		);
	}

	err = fopen_s(&label_fp, label_path.c_str(), "rb");
	if (err < 0) {
		fclose(img_fp);
		ErrorExcept(
			"[MNIST::read_file] failed load label file [%s]",
			label_path.c_str()
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
			fread_s(&param._byte[3 - j], sizeof(uchar), sizeof(uchar), 1, img_fp);
		}
		img_head[i] = param._byte32;
	}

	for (int i = 0; i < 2; ++i) {
		for (int j = 0; j < 4; ++j) {
			fread_s(&param._byte[3 - j], sizeof(uchar), sizeof(uchar), 1, label_fp);
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

	samples._x = cv::Mat(std::vector<int>({ img_head[1], img_head[2], img_head[3] }), CV_8UC1);
	samples._y = cv::Mat(std::vector<int>({ label_head[1] }), CV_8UC1);

	fread_s(
		samples._x.data,
		sizeof(uchar) * samples._x.total(),
		sizeof(uchar),
		samples._x.total(),
		img_fp
	);
	fread_s(
		samples._y.data,
		sizeof(uchar) * samples._y.total(),
		sizeof(uchar),
		samples._y.total(),
		label_fp
	);

	fclose(img_fp);
	fclose(label_fp);

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

	const int amounts = origin._x.size[0];
	const int img_h = origin._x.size[1];
	const int img_w = origin._x.size[2];

	sample._x = cv::Mat(std::vector<int>({ n_batch, img_h, img_w }), CV_8UC1);
	sample._y = cv::Mat(std::vector<int>({ n_batch }), CV_8UC1);

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

	for (int i = 0; i < n_batch; ++i) {
		cv::Mat src_img(img_h, img_w, CV_8UC1, (uchar*)origin._x.ptr<uchar>(batch_indice[i]));
		cv::Mat dst_img(img_h, img_w, CV_8UC1, (uchar*)sample._x.ptr<uchar>(i));

		cv::Mat src_label({ 1 }, CV_8UC1, (uchar*)origin._y.ptr<uchar>(batch_indice[i]));
		cv::Mat dst_label({ 1 }, CV_8UC1, (uchar*)sample._y.ptr<uchar>(i));

		src_img.copyTo(dst_img);
		src_label.copyTo(dst_label);
	}

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