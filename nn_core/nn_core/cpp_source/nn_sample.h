#pragma once
#include "nn_tensor.h"


template <typename _xT, typename _yT>
struct DataSet {
	Tensor<_xT> _x;
	Tensor<_yT> _y;
};


template <typename _xT, typename _yT>
class Sample {
private:
	const DataSet<_xT, _yT>& _origin;
	const int _n_batch;
	const int _n_iter;
	const bool _shuffle;

	static const DataSet<_xT, _yT> get_batch_samples(const DataSet<_xT, _yT>& origin, int index, int n_batch, bool shuffle);

public:
	class Iterator {
	public:
		const Sample& _samples;
		int _n_iter;

		Iterator(const Sample& samples, int n_iter);
		Iterator(const typename Iterator& p);

		bool operator!=(const typename Iterator& p) const;
		void operator++();
		const DataSet<_xT, _yT> operator*() const;
	};

	Sample(const DataSet<_xT, _yT>& current_samples, int n_batch, int n_iter, bool shuffle);

	typename Iterator begin() const;
	typename Iterator end() const;

	const DataSet<_xT, _yT> operator[](int index) const;

	size_t get_amounts() const;
	size_t get_iteration() const;
};

template <typename _xT, typename _yT>
const DataSet<_xT, _yT> Sample<_xT, _yT>::get_batch_samples(const DataSet<_xT, _yT>& origin, int index, int n_batch, bool shuffle) {
	DataSet<_xT, _yT> sample;

	const NN_Shape& shape = origin._x.get_shape();

	const int amounts = shape[0];
	const int img_h = shape[1];
	const int img_w = shape[2];

	sample._x.resize({ n_batch, img_h, img_w });
	sample._y.resize({ n_batch });

	std::vector<int> batch_indice;

	if (shuffle) {
		batch_indice = random_choice(0, amounts, n_batch, shuffle);
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

template <typename _xT, typename _yT>
Sample<_xT, _yT>::Iterator::Iterator(const Sample& samples, int n_iter) :
	_samples(samples),
	_n_iter(n_iter)
{
}

template <typename _xT, typename _yT>
Sample<_xT, _yT>::Iterator::Iterator(const typename Iterator& p) :
	_samples(p._samples),
	_n_iter(p._n_iter)
{
}

template <typename _xT, typename _yT>
bool Sample<_xT, _yT>::Iterator::operator!=(const typename Iterator& p) const {
	return _n_iter != p._n_iter;
}

template <typename _xT, typename _yT>
void Sample<_xT, _yT>::Iterator::operator++() {
	++_n_iter;
}

template <typename _xT, typename _yT>
const DataSet<_xT, _yT> Sample<_xT, _yT>::Iterator::operator*() const {
	return get_batch_samples(_samples._origin, _n_iter, _samples._n_batch, _samples._shuffle);
}

template <typename _xT, typename _yT>
typename Sample<_xT, _yT>::Iterator Sample<_xT, _yT>::begin() const {
	return Iterator(*this, 0);
}

template <typename _xT, typename _yT>
typename Sample<_xT, _yT>::Iterator Sample<_xT, _yT>::end() const {
	return Iterator(*this, _n_iter);
}

template <typename _xT, typename _yT>
Sample<_xT, _yT>::Sample(const DataSet<_xT, _yT>& current_samples, int n_batch, int n_iter, bool shuffle) :
	_origin(current_samples),
	_n_batch(n_batch),
	_n_iter(n_iter),
	_shuffle(shuffle)
{
}

template <typename _xT, typename _yT>
const DataSet<_xT, _yT> Sample<_xT, _yT>::operator[](int index) const {
	if (index >= _n_iter) {
		ErrorExcept(
			"[MNIST::Sample::operator[]] Index[%d] is out of range. (%d ~ %d)",
			index,
			0, _n_iter
		);
	}

	return get_batch_samples(_origin, index, _n_batch, _shuffle);
}

template <typename _xT, typename _yT>
size_t Sample<_xT, _yT>::get_amounts() const {
	return _n_batch;
}

template <typename _xT, typename _yT>
size_t Sample<_xT, _yT>::get_iteration() const {
	return _n_iter;
}