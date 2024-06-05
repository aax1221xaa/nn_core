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

	static void set_batch_samples(const DataSet<_xT, _yT>& origin, DataSet<_xT, _yT>& batch_samples, int index, int n_batch, bool shuffle);

	DataSet<_xT, _yT> get_buffer() const;

public:
	class Iterator {
	public:
		const Sample& _p_samples;
		DataSet<_xT, _yT> _buff;
		int _n_iter;

		Iterator(const Sample& p_samples, int n_iter);
		Iterator(const typename Iterator& p);

		bool operator!=(const typename Iterator& p) const;
		void operator++();
		const DataSet<_xT, _yT>& operator*() const;
	};

	Sample(const DataSet<_xT, _yT>& current_object, int n_batch, int n_iter, bool shuffle);

	typename Iterator begin() const;
	typename Iterator end() const;

	DataSet<_xT, _yT> operator[](int index) const;

	int get_batch() const;
	int get_iteration() const;
};

template <typename _xT, typename _yT>
void Sample<_xT, _yT>::set_batch_samples(const DataSet<_xT, _yT>& origin, DataSet<_xT, _yT>& batch_samples, int index, int n_batch, bool shuffle) {
	std::vector<int> batch_indice;
	const int amounts = origin._x.get_shape()[0];

	if (shuffle) {
		batch_indice = random_choice(0, amounts, n_batch, shuffle);
	}
	else {
		batch_indice.resize(n_batch);

		for (int i = 0; i < n_batch; ++i) {
			batch_indice[i] = (n_batch * index + i) % amounts;
		}
	}

	batch_samples._x = origin._x(batch_indice);
	batch_samples._y = origin._y(batch_indice);
}

template <typename _xT, typename _yT>
DataSet<_xT, _yT> Sample<_xT, _yT>::get_buffer() const {
	NN_Shape x_shape = _origin._x.get_shape();
	NN_Shape y_shape = _origin._y.get_shape();

	x_shape[0] = _n_batch;
	y_shape[0] = _n_batch;

	DataSet<_xT, _yT> buffer;

	buffer._x.resize(x_shape);
	buffer._y.resize(y_shape);

	return buffer;
}

template <typename _xT, typename _yT>
Sample<_xT, _yT>::Iterator::Iterator(const Sample& p_samples, int n_iter) :
	_p_samples(p_samples),
	_n_iter(n_iter),
	_buff(p_samples.get_buffer())
{
	set_batch_samples(_p_samples._origin, _buff, _n_iter, _p_samples._n_batch, _p_samples._shuffle);
}

template <typename _xT, typename _yT>
Sample<_xT, _yT>::Iterator::Iterator(const typename Iterator& p) :
	_p_samples(p._p_samples),
	_n_iter(p._n_iter),
	_buff(p._buff)
{
}

template <typename _xT, typename _yT>
bool Sample<_xT, _yT>::Iterator::operator!=(const typename Iterator& p) const {
	return _n_iter != p._n_iter;
}

template <typename _xT, typename _yT>
void Sample<_xT, _yT>::Iterator::operator++() {
	set_batch_samples(_p_samples._origin, _buff, _n_iter, _p_samples._n_batch, _p_samples._shuffle);
	++_n_iter;
}

template <typename _xT, typename _yT>
const DataSet<_xT, _yT>& Sample<_xT, _yT>::Iterator::operator*() const {
	return _buff;
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
Sample<_xT, _yT>::Sample(const DataSet<_xT, _yT>& current_object, int n_batch, int n_iter, bool shuffle) :
	_origin(current_object),
	_n_batch(n_batch),
	_n_iter(n_iter),
	_shuffle(shuffle)
{
}

template <typename _xT, typename _yT>
DataSet<_xT, _yT> Sample<_xT, _yT>::operator[](int index) const {
	if (index >= _n_iter) {
		ErrorExcept(
			"[MNIST::Sample::operator[]] Index[%d] is out of range. (%d ~ %d)",
			index,
			0, _n_iter
		);
	}

	NN_Shape x_shape = _origin._x.get_shape();
	NN_Shape y_shape = _origin._y.get_shape();

	x_shape[0] = _n_batch;
	y_shape[0] = _n_batch;

	DataSet<_xT, _yT> samples;
	samples._x.resize(x_shape);
	samples._y.resize(y_shape);

	set_batch_samples(_origin, samples, index, _n_batch, _shuffle);

	return samples;
}

template <typename _xT, typename _yT>
int Sample<_xT, _yT>::get_batch() const {
	return _n_batch;
}

template <typename _xT, typename _yT>
int Sample<_xT, _yT>::get_iteration() const {
	return _n_iter;
}