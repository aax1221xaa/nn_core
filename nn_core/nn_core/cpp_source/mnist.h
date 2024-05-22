#pragma once

#include "nn_tensor.h"


class MNIST {
public:
	struct DataSet {
		Tensor<uchar> _x;
		Tensor<uchar> _y;
	}_train, _test;

private:
	static DataSet read_file(const std::string& img_path, const std::string& label_path);

public:
	class Sample {
	private:
		const DataSet& _origin;
		const int _n_batch;
		const int _n_iter;
		const bool _shuffle;

		static const DataSet get_batch_samples(const DataSet& origin, int index, int n_batch, bool shuffle);

	public:
		class Iterator {
		public:
			const Sample& _samples;
			int _n_iter;

			Iterator(const Sample& samples, int n_iter);
			Iterator(const typename Iterator& p);

			bool operator!=(const typename Iterator& p) const;
			void operator++();
			const DataSet operator*() const;
		};

		Sample(const DataSet& current_samples, int n_batch, int n_iter, bool shuffle);

		typename Iterator begin() const;
		typename Iterator end() const;

		const DataSet operator[](int index) const;
	};

	MNIST(const std::string& dir_path);

	Sample get_train_samples(int n_batch, int n_iter, bool shuffle = true) const;
	Sample get_test_samples(int n_batch, int n_iter, bool shuffle = true) const;
};