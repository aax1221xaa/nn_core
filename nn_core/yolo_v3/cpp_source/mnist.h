#pragma once

#include "nn_sample.h"
#include <opencv2/opencv.hpp>



class MNIST {
	Sample<uchar, uchar> _train, _test;

	static void read_file(
		const std::string& img_path,
		const std::string& label_path,
		Sample<uchar, uchar>& dst
	);

	class Generator : public SampleGenerator<uchar, uchar> {
		Sample<uchar, uchar>* _samples;

	public:
		void generate_sample(const std::vector<int>& indices, Sample<uchar, uchar>& dst) const;

		Generator();
		Generator(Sample<uchar, uchar>& samples, int batch, bool do_shuffle);

		const Generator& operator=(Generator&& p);
	};

	Generator _train_gen, _test_gen;

public:
	MNIST(const std::string& dir_path, int batch, int do_shuffle);

	const SampleGenerator<uchar, uchar>& get_train_dataset() const;
	const SampleGenerator<uchar, uchar>& get_test_dataset() const;
};