#pragma once
#include "nn_shape.h"
#include <tbb/tbb.h>
#include <memory>


/**********************************************/
/*                                            */
/*                   Tensor                   */
/*                                            */
/**********************************************/

/*
	Tensor test(10000, 3, 512, 512);

	Tensor test2 = test(0, 64)(-1)(128, 384)(128, 384);   {64, 3, 256, 256}
	Tensor test3 = test(3)(-1);
	
	Tensor test4(3, 3, 3, 3};
	Tensor test5({3, 3});

	test = 1;
	test(0, 3) = 1;
	test[0][0] = 1;

	Tensor operator()(int begin, int end, int step = 1);

	Tensor operator=(int scalar);
	Tensor operator=(const Tensor& p);
*/

template <typename _T>
class Tensor {
	struct ROI {
		int _begin;
		int _end;
		int _step;
	};

	std::shared_ptr<_T[]> _data;
	std::vector<size_t> _steps;
	std::vector<ROI> _roi;

	int _count_op;

	static std::vector<size_t> calc_step(const NN_Shape& shape);
	static size_t calc_size(const std::vector<ROI>& roi);
	static size_t get_elem_index(const std::vector<size_t>& steps, const std::vector<ROI>& roi, int rank, int index);
	static size_t count_to_elem_index(const std::vector<size_t>& steps, const std::vector<ROI>& roi, size_t count);
	static NN_Shape calc_shape(const std::vector<ROI>& roi);
	static bool is_valid_src_shape(const NN_Shape& dst_shape, const NN_Shape& src_shape);

public:

	Tensor();
	Tensor(const NN_Shape& shape);
	Tensor(const Tensor& p);
	Tensor(Tensor&& p);

	Tensor& operator=(const Tensor& p);
	Tensor& operator=(_T scalar);
	Tensor operator()(int begin, int end, int step = 1);
	Tensor operator()(int index);
	Tensor operator[](int index);

	_T& val();
	const _T& val() const;

	void clear();
	NN_Shape get_shape();
	std::shared_ptr<_T[]>& get_data();
};

template <typename _T>
using tensor_t = std::shared_ptr<_T[]>;

template <typename _T>
std::vector<size_t> Tensor<_T>::calc_step(const NN_Shape& shape) {
	std::vector<size_t> steps(shape.get_len(), 0);

	for (int i = 0; i < shape.get_len(); ++i) {
		size_t size = 1;

		for (int j = i + 1; j < shape.get_len(); ++j) {
			size *= (size_t)shape[j];
		}

		steps[i] = size;
	}

	return steps;
}

template <typename _T>
size_t Tensor<_T>::calc_size(const std::vector<ROI>& roi) {
	size_t size = roi.size() > 0 ? 1 : 0;

	for (const ROI& m_roi : roi) {
		size *= (m_roi._end - m_roi._begin + m_roi._step - 1) / m_roi._step;
	}

	return size;
}

template <typename _T>
size_t Tensor<_T>::get_elem_index(const std::vector<size_t>& steps, const std::vector<ROI>& roi, int rank, int index) {
	const ROI& m_roi = roi[rank];
	const size_t& m_step = steps[rank];

	return m_step * (size_t)(m_roi._begin + (index * m_roi._step));
}

template <typename _T>
size_t Tensor<_T>::count_to_elem_index(const std::vector<size_t>& steps, const std::vector<ROI>& roi, size_t count) {
	size_t i = steps.size();
	size_t index = 0;

	while (i > 0) {
		--i;
		const size_t& step = steps[i];
		const ROI& m_roi = roi[i];
		const size_t dim = (m_roi._end - m_roi._begin + m_roi._step - 1) / m_roi._step;

		index += step * (m_roi._begin + (count % dim) * m_roi._step);

		count /= dim;
	}

	return index;
}

template <typename _T>
NN_Shape Tensor<_T>::calc_shape(const std::vector<ROI>& roi) {
	NN_Shape shape((int)roi.size());

	int i = 0;
	for (const ROI& m_roi : roi) {
		shape[i++] = (m_roi._end - m_roi._begin + m_roi._step - 1) / m_roi._step;
	}

	return shape;
}

template <typename _T>
bool Tensor<_T>::is_valid_src_shape(const NN_Shape& dst_shape, const NN_Shape& src_shape) {
	bool is_valid = true;
	const int dst_len = dst_shape.get_len();
	const int src_len = src_shape.get_len();

	if (dst_len < src_len) is_valid = false;
	if (src_len > 0) {
		for (int i = 0; i < src_len; ++i) {
			if (dst_shape[dst_len - i - 1] != src_shape[src_len - i - 1]) {
				is_valid = false;
				break;
			}
		}
	}
	else is_valid = false;

	return is_valid;
}

template <typename _T>
Tensor<_T>::Tensor() :
	_count_op(0)
{
}

template <typename _T>
Tensor<_T>::Tensor(const NN_Shape& shape) :
	_count_op(0)
{
	_roi.resize(shape.get_len());

	int i = 0;

	for (const int& n : shape) {
		ROI roi;

		roi._begin = 0;
		roi._end = n;
		roi._step = 1;

		_roi[i++] = roi;
	}

	_steps = calc_step(shape);
	_data = std::shared_ptr<_T[]>(new _T[shape.total_size()]);
}

template <typename _T>
Tensor<_T>::Tensor(const Tensor& p) :
	_count_op(p._count_op),
	_data(p._data),
	_steps(p._steps),
	_roi(p._roi)
{
}

template <typename _T>
Tensor<_T>::Tensor(Tensor&& p) :
	_count_op(p._count_op),
	_data(std::move(p._data)),
	_steps(std::move(p._steps)),
	_roi(std::move(p._roi))
{
}

template <typename _T>
Tensor<_T>& Tensor<_T>::operator=(const Tensor& p) {
	if (this == &p) return *this;

	_count_op = 0;

	if (_steps.empty()) {
		_data = p._data;
		_steps = p._steps;
		_roi = p._roi;
	}
	else {
		const NN_Shape origin_shape = calc_shape(_roi);
		const NN_Shape source_shape = calc_shape(p._roi);

		if (!is_valid_src_shape(origin_shape, source_shape)) {
			ErrorExcept(
				"[Tensor<_T>::operator=] Source shape is wrong. %s.",
				shape_to_str(source_shape)
			);
		}

		const size_t src_size = calc_size(p._roi);
		const size_t dst_size = calc_size(_roi);

		tbb::parallel_for<tbb::blocked_range<size_t>>(
			tbb::blocked_range<size_t>(0, dst_size),
			[&](const tbb::blocked_range<size_t>& q) {
			for (size_t i = q.begin(); i < q.end(); ++i) {
				const size_t src_idx = count_to_elem_index(p._steps, p._roi, i % src_size);
				const size_t dst_idx = count_to_elem_index(_steps, _roi, i);

				_data[dst_idx] = p._data[src_idx];
			}
		}
		);
	}

	return *this;
}

template <typename _T>
Tensor<_T>& Tensor<_T>::operator=(_T scalar) {
	_count_op = 0;

	if (_steps.empty()) {
		_data = std::shared_ptr<_T[]>(new _T[1]);
		_steps.push_back(1);
		_roi.push_back({ 0, 1, 1 });
	}
	else {
		const size_t size = calc_size(_roi);

		tbb::parallel_for<tbb::blocked_range<size_t>>(
			tbb::blocked_range<size_t>(0, size),
			[&](const tbb::blocked_range<size_t>& q) {
			for (size_t i = q.begin(); i < q.end(); ++i) {
				const size_t index = count_to_elem_index(_steps, _roi, i);

				_data[index] = scalar;
			}
		}
		);
	}

	return *this;
}

template <typename _T>
Tensor<_T> Tensor<_T>::operator()(int begin, int end, int step) {
	if (_count_op >= _roi.size()) {
		ErrorExcept(
			"[Tensor<_T>::operator()] %d rank is empty.",
			_count_op
		);
	}

	const ROI& roi = _roi[_count_op];
	const int n = (roi._end - roi._begin + roi._step - 1) / roi._step;

	begin = begin < 0 ? begin - n : begin;
	end = end < 0 ? end - n : end;

	if (begin < 0 || begin > n || end < 0 || end > n || end < begin) {
		ErrorExcept(
			"[Tensor<_T>::operator()] begin and end is out of range."
		);
	}

	Tensor<_T> tensor = *this;
	tensor._roi[_count_op] = ROI({ begin, end, step });
	tensor._count_op = _count_op + 1;

	_count_op = 0;

	return tensor;
}

template <typename _T>
Tensor<_T> Tensor<_T>::operator()(int index) {
	if (_count_op >= _roi.size()) {
		ErrorExcept(
			"[Tensor<_T>::operator()] %d rank is empty.",
			_count_op
		);
	}

	const ROI& roi = _roi[_count_op];
	const int n = (roi._end - roi._begin + roi._step - 1) / roi._step;

	index = index < 0 ? n - index : index;

	if (index < 0 || index > n) {
		ErrorExcept(
			"[Tensor<_T>::operator()] begin and end is out of range."
		);
	}

	Tensor<_T> tensor = *this;
	tensor._roi[_count_op] = ROI({ index, index + 1, 1 });
	tensor._count_op = _count_op + 1;

	_count_op = 0;

	return tensor;
}

template <typename _T>
Tensor<_T> Tensor<_T>::operator[](int index) {
	if (_count_op >= _roi.size()) {
		ErrorExcept(
			"[Tensor<_T>::operator[]] %d rank is empty.",
			_count_op
		);
	}

	const ROI& roi = _roi[_count_op];
	const int n = (roi._end - roi._begin + roi._step - 1) / roi._step;

	index = index < 0 ? n - index : index;

	if (index < 0 || index >= n) {
		ErrorExcept(
			"[Tensor<_T>::operator[]] begin and end is out of range."
		);
	}

	Tensor<_T> tensor = *this;
	tensor._roi[_count_op] = ROI({ index, index + 1, 1 });
	tensor._count_op = _count_op + 1;

	_count_op = 0;

	return tensor;
}

template <typename _T>
_T& Tensor<_T>::val() {
	if (calc_size(_roi) > 1) {
		ErrorExcept(
			"[Tensor<_T>::val] This value is not scalar."
		);
	}

	const size_t index = count_to_elem_index(_steps, _roi, 0);

	return _data[index];
}

template <typename _T>
const _T& Tensor<_T>::val() const {
	if (calc_size(_roi) > 1) {
		ErrorExcept(
			"[Tensor<_T>::val] This value is not scalar."
		);
	}

	const size_t index = count_to_elem_index(_steps, _roi, 0);

	return _data[index];
}

template <typename _T>
void Tensor<_T>::clear() {
	_data.reset();
	_steps.clear();
	_roi.clear();
	_count_op = 0;
}

template <typename _T>
NN_Shape Tensor<_T>::get_shape() {
	return calc_shape(_roi);
}

template <typename _T>
std::shared_ptr<_T[]>& Tensor<_T>::get_data() {
	return _data;
}