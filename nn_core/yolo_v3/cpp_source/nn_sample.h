#pragma once
#include "nn_tensor_plus.h"
#include <random>



/**********************************************/
/*                                            */
/*                    Sample				  */
/*                                            */
/**********************************************/

template <typename _xT, typename _yT>
struct Sample {
	std::vector<Tensor<_xT>> _x;
	std::vector<Tensor<_yT>> _y;
};


/**********************************************/
/*                                            */
/*                   DataSet				  */
/*                                            */
/**********************************************/

template <typename _xT, typename _yT>
class SampleGenerator {
protected:
	bool _do_shuffle;
	int _len;
	int _batch;
	int _max_index;

public:
	typedef _xT xt;
	typedef _yT yt;

	class Iterator {
	public:
		const SampleGenerator& _gen;
		int _index;

		Iterator(const SampleGenerator& generator, int index);
		Iterator(const typename Iterator& p);

		bool operator!=(const typename Iterator& p);
		void operator++();
		Sample<xt, yt> operator*();
	};

	class ConstIterator {
	public:
		const SampleGenerator& _gen;
		int _index;

		ConstIterator(const SampleGenerator& generator, int index);
		ConstIterator(const typename ConstIterator& p);

		bool operator!=(const typename ConstIterator& p) const;
		void operator++();
		Sample<xt, yt> operator*() const;
	};

	void gen_indices(int index, std::vector<int>& indices) const;
	void gen_shuffle_indices(std::vector<int>& indices) const;
	void check_index(int index) const;

	SampleGenerator();
	SampleGenerator(int len, int batch, bool do_shuffle);
	SampleGenerator(const SampleGenerator& p);
	virtual ~SampleGenerator();

	virtual void generate_sample(const std::vector<int>& indices, Sample<xt, yt>& dst) const = 0;

	typename Iterator begin();
	typename Iterator end();

	typename ConstIterator begin() const;
	typename ConstIterator end() const;

	Sample<xt, yt> operator[](int index);
	Sample<xt, yt> operator[](int index) const;

	const SampleGenerator& operator=(const SampleGenerator& p);

	void set_params(bool do_shuffle, int len, int batch, int max_index);
};

template <typename _xT, typename _yT>
SampleGenerator<_xT, _yT>::Iterator::Iterator(const SampleGenerator& generator, int index) :
	_gen(generator),
	_index(index)
{

}

template <typename _xT, typename _yT>
SampleGenerator<_xT, _yT>::Iterator::Iterator(const typename SampleGenerator::Iterator& p) :
	_gen(p._gen),
	_index(p._index)
{
}

template <typename _xT, typename _yT>
bool SampleGenerator<_xT, _yT>::Iterator::operator!=(const typename SampleGenerator::Iterator& p) {
	return _index != p._index;
}

template <typename _xT, typename _yT>
void SampleGenerator<_xT, _yT>::Iterator::operator++() {
	++_index;
}

template <typename _xT, typename _yT>
Sample<_xT, _yT> SampleGenerator<_xT, _yT>::Iterator::operator*() {
	Sample<_xT, _yT> buffer;
	std::vector<int> indices;

	if (_gen._do_shuffle) _gen.gen_shuffle_indices(indices);
	else _gen.gen_indices(_index, indices);

	_gen.generate_sample(indices, buffer);

	return buffer;
}

template <typename _xT, typename _yT>
SampleGenerator<_xT, _yT>::ConstIterator::ConstIterator(const SampleGenerator& generator, int index) :
	_gen(generator),
	_index(index)
{

}

template <typename _xT, typename _yT>
SampleGenerator<_xT, _yT>::ConstIterator::ConstIterator(const typename SampleGenerator::ConstIterator& p) :
	_gen(p._gen),
	_index(p._index)
{
}

template <typename _xT, typename _yT>
bool SampleGenerator<_xT, _yT>::ConstIterator::operator!=(const typename SampleGenerator::ConstIterator& p) const {
	return _index != p._index;
}

template <typename _xT, typename _yT>
void SampleGenerator<_xT, _yT>::ConstIterator::operator++() {
	++_index;
}

template <typename _xT, typename _yT>
Sample<_xT, _yT> SampleGenerator<_xT, _yT>::ConstIterator::operator*() const {
	Sample<_xT, _yT> buffer;
	std::vector<int> indices;

	if (_gen._do_shuffle) _gen.gen_shuffle_indices(indices);
	else _gen.gen_indices(_index, indices);

	_gen.generate_sample(indices, buffer);

	return buffer;
}

template <typename _xT, typename _yT>
void SampleGenerator<_xT, _yT>::gen_indices(int index, std::vector<int>& indices) const {
	if (_batch > _len) {
		ErrorExcept(
			"Batch size is bigger then data size. (%d > %d)",
			_batch,
			_len
		);
	}
	else if (((index + 1) * _batch) > _len) {
		ErrorExcept(
			"Index is out of range. (0 ~ %d)",
			_len / _batch
		);
	}

	const int start = _batch * index;
	const int end = _batch * (index + 1);

	indices.reserve(_batch);

	for (int i = start; i < end; ++i) indices.push_back(i);
}

template <typename _xT, typename _yT>
void SampleGenerator<_xT, _yT>::gen_shuffle_indices(std::vector<int>& indices) const {
	if (_batch > _len) {
		ErrorExcept(
			"Batch size is bigger then data size. (%d > %d)",
			_batch,
			_len
		);
	}

	std::vector<int> selected;

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<int> dist(0, _len - 1);

	indices.resize(_batch);
	selected.reserve(_batch);

	for (int& n : indices) {
		bool is_equal = true;
		int num = 0;

		while (is_equal) {
			num = dist(gen);
			is_equal = false;

			for (int& j : selected) {
				if (num == j) is_equal = true;
			}
		}

		n = num;
		selected.push_back(num);
	}
}

template <typename _xT, typename _yT>
void SampleGenerator<_xT, _yT>::check_index(int index) const {
	if (index >= _max_index) {
		ErrorExcept(
			"Out of range. (%d <= %d)",
			_max_index,
			index
		);
	}
}

template <typename _xT, typename _yT>
SampleGenerator<_xT, _yT>::SampleGenerator() :
	_do_shuffle(true),
	_len(0),
	_batch(0),
	_max_index(0)
{

}

template <typename _xT, typename _yT>
SampleGenerator<_xT, _yT>::SampleGenerator(int len, int batch, bool do_shuffle) :
	_do_shuffle(do_shuffle),
	_len(len),
	_batch(batch),
	_max_index(len / batch)
{

}

template <typename _xT, typename _yT>
SampleGenerator<_xT, _yT>::SampleGenerator(const SampleGenerator& p) :
	_do_shuffle(p._do_shuffle),
	_len(p._len),
	_batch(p._batch),
	_max_index(p._max_index)
{

}

template <typename _xT, typename _yT>
SampleGenerator<_xT, _yT>::~SampleGenerator() {

}

template <typename _xT, typename _yT>
typename SampleGenerator<_xT, _yT>::Iterator SampleGenerator<_xT, _yT>::begin() {
	return Iterator(*this, 0);
}

template <typename _xT, typename _yT>
typename SampleGenerator<_xT, _yT>::Iterator SampleGenerator<_xT, _yT>::end() {
	return Iterator(*this, _max_index);
}

template <typename _xT, typename _yT>
typename SampleGenerator<_xT, _yT>::ConstIterator SampleGenerator<_xT, _yT>::begin() const {
	return ConstIterator(*this, 0);
}

template <typename _xT, typename _yT>
typename SampleGenerator<_xT, _yT>::ConstIterator SampleGenerator<_xT, _yT>::end() const {
	return ConstIterator(*this, _max_index);
}

template <typename _xT, typename _yT>
Sample<_xT, _yT> SampleGenerator<_xT, _yT>::operator[](int index) {
	Sample<_xT, _yT> buffer;
	std::vector<int> indices;

	index = index < 0 ? _max_index + index : index;

	check_index(index);

	if (_do_shuffle) gen_shuffle_indices(indices);
	else gen_indices(index, indices);

	generate_sample(indices, buffer);

	return buffer;
}

template <typename _xT, typename _yT>
Sample<_xT, _yT> SampleGenerator<_xT, _yT>::operator[](int index) const {
	Sample<_xT, _yT> buffer;
	std::vector<int> indices;

	index = index < 0 ? _max_index + index : index;

	check_index(index);

	if (_do_shuffle) gen_shuffle_indices(indices);
	else gen_indices(index, indices);

	generate_sample(indices, buffer);

	return buffer;
}

template <typename _xT, typename _yT>
void SampleGenerator<_xT, _yT>::set_params(bool do_shuffle, int len, int batch, int max_index) {
	_do_shuffle = do_shuffle;
	_len = len;
	_batch = batch;
	_max_index = max_index;
}

template <typename _xT, typename _yT>
const SampleGenerator<_xT, _yT>& SampleGenerator<_xT, _yT>::operator=(const typename SampleGenerator& p) {
	if (this == &p) *this;

	_do_shuffle = p._do_shuffle;
	_max_index = p._max_index;
	_batch = p._batch;
	_len = p._len;

	return *this;
}