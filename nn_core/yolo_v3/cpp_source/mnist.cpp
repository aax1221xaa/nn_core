#include "mnist.h"
#include <fstream>



void MNIST::read_file(
	const std::string& img_path, 
	const std::string& label_path,
	Sample<uchar, uchar>& dst
) {
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

	Tensor<uchar> x({ img_head[1], img_head[2], img_head[3], 1 });
	Tensor<uchar> y({ img_head[1] });

	img_fp.read((char*)x.get_ptr(), sizeof(uchar) * x.get_shape().total_size());
	label_fp.read((char*)y.get_ptr(), sizeof(uchar) * y.get_shape().total_size());

	img_fp.close();
	label_fp.close();

	dst._x.push_back(x);
	dst._y.push_back(y);
}

void MNIST::Generator::generate_sample(const std::vector<int>& indices, Sample<xt, yt>& dst) const {
	dst._x.resize(_samples->_x.size());
	dst._y.resize(_samples->_y.size());

	std::vector<Tensor<xt>>::iterator src_x_iter = _samples->_x.begin();
	std::vector<Tensor<yt>>::iterator src_y_iter = _samples->_y.begin();

	for (Tensor<xt>& tensor : dst._x) {
		tensor = (*src_x_iter)(indices);
		++src_x_iter;
	}

	for (Tensor<yt>& tensor : dst._y) {
		tensor = (*src_y_iter)(indices);
		++src_y_iter;
	}
}

MNIST::Generator::Generator() :
	_samples(NULL)
{

}

MNIST::Generator::Generator(Sample<MNIST::xt, MNIST::yt>& samples, int batch, bool do_shuffle) :
	SampleGenerator(),
	_samples(&samples)
{
	const NN_Shape shape = _samples->_x[0].get_shape();

	_len = (uint)shape[0];
	_batch = batch;
	_max_index = _len / _batch;
	_do_shuffle = do_shuffle;
}

const MNIST::Generator& MNIST::Generator::operator=(Generator&& p) {
	if (this == &p) return *this;

	_max_index = p._max_index;
	_len = p._len;
	_batch = p._batch;
	_do_shuffle = p._do_shuffle;
	_samples = p._samples;

	p._max_index = 0;
	p._len = 0;
	p._batch = 0;
	p._samples = NULL;

	return *this;
}

MNIST::MNIST(const std::string& dir_path, int batch, int do_shuffle) {
	const std::string train_x = dir_path + "\\train-images.idx3-ubyte";
	const std::string train_y = dir_path + "\\train-labels.idx1-ubyte";
	const std::string test_x = dir_path + "\\t10k-images.idx3-ubyte";
	const std::string test_y = dir_path + "\\t10k-labels.idx1-ubyte";
	
	try {
		read_file(train_x, train_y, _train);
		read_file(test_x, test_y, _test);

		_train_gen = Generator(_train, batch, do_shuffle);
		_test_gen = Generator(_test, batch, do_shuffle);
	}
	catch (const NN_Exception& e) {
		e.put();
	}
}

const DataSet<MNIST::xt, MNIST::yt>::SampleGenerator& MNIST::get_train_dataset() const {
	return _train_gen;
}

const DataSet<MNIST::xt, MNIST::yt>::SampleGenerator& MNIST::get_test_dataset() const {
	return _test_gen;
}