#pragma once
#include "nn_shape.h"
#include <tbb/tbb.h>
#include <memory>


template <typename _T>
class GpuTensor;

template <typename _T>
class Tensor {

	std::shared_ptr<_T[]> _data;
	std::vector<size_t> _steps;
	std::vector<std::vector<size_t>> _indice;

	static int _rank_cnt;

	static std::vector<size_t> calc_step(const NN_Shape& shape);
	static size_t calc_size(const std::vector<std::vector<size_t>>& indice);
	static size_t count_to_elem_index(const std::vector<size_t>& steps, const std::vector<std::vector<size_t>>& indice, size_t count);
	static NN_Shape calc_shape(const std::vector<std::vector<size_t>>& indice);
	static bool is_valid_src_shape(const NN_Shape& dst_shape, const NN_Shape& src_shape);
	static void count_indice(std::vector<size_t>& indice, int begin, int end, int step);
	static void put_tensor(std::ostream& os, const Tensor& tensor, size_t offset, int& rank);

public:

	Tensor();
	Tensor(const NN_Shape& shape);
	//	Tensor(const std::initializer_list<int>& shape);
	Tensor(const Tensor& p);
	Tensor(Tensor&& p);

	Tensor& operator=(const Tensor& p);
	Tensor& operator=(const GpuTensor<_T>& p);
	Tensor& operator=(_T scalar);
	Tensor operator()(int begin, int end, int step = 1);
	Tensor operator()(int index);
	Tensor operator()(const std::vector<int>& indice) const;
	Tensor operator[](int index);

	_T& val();
	const _T& val() const;

	void clear();
	NN_Shape get_shape();
	NN_Shape get_shape() const;
	_T* get_ptr();
	const _T* get_ptr() const;

	std::ostream& put(std::ostream& os);

	void resize(const NN_Shape& shape);
};

template <typename _T>
using tensor_t = std::shared_ptr<_T[]>;

template <typename _T>
int Tensor<_T>::_rank_cnt = 0;

template <typename _T>
std::ostream& operator<<(std::ostream& os, Tensor<_T>& tensor) {
	tensor.put(os);

	return os;
}

template <typename _T>
std::vector<size_t> Tensor<_T>::calc_step(const NN_Shape& shape) {
	std::vector<size_t> steps(shape.get_len(), 1);

	int n = 0;
	for (NN_Shape::c_iterator i = shape.begin(); i != shape.end(); ++i) {
		for (NN_Shape::c_iterator j = i + 1; j != shape.end(); ++j) steps[n] *= (size_t)(*j);
		++n;
	}

	return steps;
}

template <typename _T>
size_t Tensor<_T>::calc_size(const std::vector<std::vector<size_t>>& indice) {
	size_t size = indice.size() > 0 ? 1 : 0;

	for (const std::vector<size_t>& m : indice) {
		size *= m.size();
	}

	return size;
}

template <typename _T>
size_t Tensor<_T>::count_to_elem_index(const std::vector<size_t>& steps, const std::vector<std::vector<size_t>>& indice, size_t count) {
	size_t i = steps.size();
	size_t index = 0;

	while (i > 0) {
		--i;
		const size_t& step = steps[i];
		const std::vector<size_t>& m_indice = indice[i];
		const size_t dim = m_indice.size();

		index += step * m_indice[count % dim];

		count /= dim;
	}

	return index;
}

template <typename _T>
NN_Shape Tensor<_T>::calc_shape(const std::vector<std::vector<size_t>>& indice) {
	NN_Shape shape((int)indice.size());

	int i = 0;
	for (const std::vector<size_t>& m_indice : indice) {
		shape[i++] = (int)m_indice.size();
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
void Tensor<_T>::count_indice(std::vector<size_t>& indice, int begin, int end, int step) {
	const int n = (end - begin + step - 1) / step;
	std::vector<size_t> m_indice(n, 0);

	int i = 0;
	for (size_t& m : m_indice) m = indice[begin + step * i++];

	indice = m_indice;
}

template <typename _T>
void Tensor<_T>::put_tensor(std::ostream& os, const Tensor& tensor, size_t offset, int& rank) {
	const size_t step = tensor._steps[rank];

	if (rank < tensor._indice.size() - 1) {
		for (const size_t& index : tensor._indice[rank]) {
			os << '[';
			put_tensor(os, tensor, offset + (index * step), ++rank);
			--rank;
			os << ']' << std::endl;
		}
	}
	else {
		for (const size_t& index : tensor._indice[rank]) {
			os << tensor._data[offset + (index * step)] << ", ";
		}
	}
}

template <typename _T>
Tensor<_T>::Tensor()
{
}

template <typename _T>
Tensor<_T>::Tensor(const NN_Shape& shape) :
	_indice(shape.get_len())
{
	int i = 0;
	for (const int& n : shape) {
		if (n < 1) {
			ErrorExcept(
				"[Tensor<_T>::Tensor] Shape must be greater than 0. but %s",
				shape_to_str(shape)
			);
		}

		std::vector<size_t> m_indice(n, 0);

		size_t j = 0;
		for (size_t& m : m_indice) m = j++;

		_indice[i++] = m_indice;
	}

	_steps = calc_step(shape);
	_data = std::shared_ptr<_T[]>(new _T[shape.total_size()]);
}

/*
template <typename _T>
Tensor<_T>::Tensor(const std::initializer_list<int>& shape) :
	_indice(shape.size())
{
	int i = 0;
	for (const int& n : shape) {
		if (n < 1) {
			ErrorExcept(
				"[Tensor<_T>::Tensor] Shape must be greater than 0. but %s",
				shape_to_str(shape)
			);
		}

		std::vector<size_t> m_indice(n, 0);

		size_t j = 0;
		for (size_t& m : m_indice) m = j++;

		_indice[i++] = m_indice;
	}

	_steps = calc_step(shape);
	_data = std::shared_ptr<_T[]>(new _T[NN_Shape(shape).total_size()]);
}
*/
template <typename _T>
Tensor<_T>::Tensor(const Tensor& p) :
	_data(p._data),
	_steps(p._steps),
	_indice(p._indice)
{
}

template <typename _T>
Tensor<_T>::Tensor(Tensor&& p) :
	_data(std::move(p._data)),
	_steps(std::move(p._steps)),
	_indice(std::move(p._indice))
{
}

template <typename _T>
Tensor<_T>& Tensor<_T>::operator=(const Tensor& p) {
	if (this == &p) return *this;

	_rank_cnt = 0;

	if (_steps.empty()) {
		_data = p._data;
		_steps = p._steps;
		_indice = p._indice;
	}
	else {
		const NN_Shape origin_shape = calc_shape(_indice);
		const NN_Shape source_shape = calc_shape(p._indice);

		if (!is_valid_src_shape(origin_shape, source_shape)) {
			ErrorExcept(
				"[Tensor<_T>::operator=] Source shape is wrong. %s.",
				shape_to_str(source_shape)
			);
		}

		const size_t src_size = calc_size(p._indice);
		const size_t dst_size = calc_size(_indice);

		tbb::parallel_for<tbb::blocked_range<size_t>>(
			tbb::blocked_range<size_t>(0, dst_size),
			[&](const tbb::blocked_range<size_t>& q) {
			for (size_t i = q.begin(); i < q.end(); ++i) {
				const size_t src_idx = count_to_elem_index(p._steps, p._indice, i % src_size);
				const size_t dst_idx = count_to_elem_index(_steps, _indice, i);

				_data[dst_idx] = p._data[src_idx];
			}
		}
		);
	}

	return *this;
}

template <typename _T>
Tensor<_T>& Tensor<_T>::operator=(const GpuTensor<_T>& p) {
	const NN_Shape h_shape = calc_shape(_indice);
	const NN_Shape& g_shape = p.get_shape();

	if (h_shape != g_shape) {
		ErrorExcept(
			"[Tensor<_T>::operator=] GPU tensor and Host tensor shape is different. GPU: %s, Host: %s",
			shape_to_str(g_shape),
			shape_to_str(h_shape)
		);
	}

	Tensor tmp(g_shape);
	_T* g_ptr = p.get_ptr();
	_T* h_ptr = tmp.get_ptr();

	check_cuda(cudaMemcpy(h_ptr, g_ptr, sizeof(_T) * h_shape.total_size(), cudaMemcpyDeviceToHost));

	*this = tmp;

	return *this;
}

template <typename _T>
Tensor<_T>& Tensor<_T>::operator=(_T scalar) {
	_rank_cnt = 0;

	if (_steps.empty()) {
		_data = std::shared_ptr<_T[]>(new _T[1]);
		_steps.push_back(1);
		_indice.push_back({ 0 });
	}
	else {
		const size_t size = calc_size(_indice);

		tbb::parallel_for<tbb::blocked_range<size_t>>(
			tbb::blocked_range<size_t>(0, size),
			[&](const tbb::blocked_range<size_t>& q) {
			for (size_t i = q.begin(); i < q.end(); ++i) {
				const size_t index = count_to_elem_index(_steps, _indice, i);

				_data[index] = scalar;
			}
		}
		);
	}

	return *this;
}

template <typename _T>
Tensor<_T> Tensor<_T>::operator()(int begin, int end, int step) {
	if (_rank_cnt >= _indice.size()) {
		ErrorExcept(
			"[Tensor<_T>::operator()] %d rank is empty.",
			_rank_cnt
		);
	}

	const int n = (int)_indice[_rank_cnt].size();

	begin = begin < 0 ? n + begin : begin;
	end = end < 0 ? n + end : end;

	if (begin < 0 || begin >= n || end < 0 || end > n) {
		ErrorExcept(
			"[Tensor<_T>::operator()] begin and end is out of range. begin: %d, end: %d, step: %d",
			begin, end, step
		);
	}

	Tensor<_T> tensor = *this;
	count_indice(tensor._indice[_rank_cnt++], begin, end, step);

	return tensor;
}

template <typename _T>
Tensor<_T> Tensor<_T>::operator()(int index) {
	if (_rank_cnt >= _indice.size()) {
		ErrorExcept(
			"[Tensor<_T>::operator()] %d rank is empty.",
			_rank_cnt
		);
	}

	const int n = (int)_indice[_rank_cnt].size();

	index = index < 0 ? n + index : index;

	if (index < 0 || index >= n) {
		ErrorExcept(
			"[Tensor<_T>::operator()] index is out of range."
		);
	}

	Tensor<_T> tensor = *this;
	count_indice(tensor._indice[_rank_cnt++], index, index + 1, 1);

	return tensor;
}

template <typename _T>
Tensor<_T> Tensor<_T>::operator()(const std::vector<int>& indice) const {
	if (_rank_cnt >= _indice.size()) {
		ErrorExcept(
			"[Tensor<_T>::operator()] %d rank is empty.",
			_rank_cnt
		);
	}

	const std::vector<size_t>& curr_indice = _indice[_rank_cnt];
	const int n = (int)curr_indice.size();
	std::vector<size_t> m_indice(indice.size(), 0);

	int i = 0;
	for (const int& m : indice) {
		int index = m < 0 ? n + m : m;

		if (index < 0 || index >= n) {
			ErrorExcept(
				"[Tensor<_T>::operator()] indice is out of range."
			);
		}

		m_indice[i++] = curr_indice[index];
	}

	Tensor<_T> tensor = *this;
	tensor._indice[_rank_cnt++] = m_indice;

	return tensor;
}

template <typename _T>
Tensor<_T> Tensor<_T>::operator[](int index) {
	if (_rank_cnt >= _indice.size()) {
		ErrorExcept(
			"[Tensor<_T>::operator()] %d rank is empty.",
			_rank_cnt
		);
	}

	const int n = (int)_indice[_rank_cnt].size();

	index = index < 0 ? n + index : index;

	if (index < 0 || index >= n) {
		ErrorExcept(
			"[Tensor<_T>::operator()] begin and end is out of range."
		);
	}

	Tensor<_T> tensor = *this;
	count_indice(tensor._indice[_rank_cnt++], index, index + 1, 1);

	return tensor;
}

template <typename _T>
_T& Tensor<_T>::val() {
	_rank_cnt = 0;

	if (calc_size(_indice) > 1) {
		ErrorExcept(
			"[Tensor<_T>::val] This value is not scalar."
		);
	}

	const size_t index = count_to_elem_index(_steps, _indice, 0);

	return _data[index];
}

template <typename _T>
const _T& Tensor<_T>::val() const {
	if (calc_size(_indice) > 1) {
		ErrorExcept(
			"[Tensor<_T>::val] This value is not scalar."
		);
	}

	const size_t index = count_to_elem_index(_steps, _indice, 0);

	return _data[index];
}

template <typename _T>
void Tensor<_T>::clear() {
	_data.reset();
	_steps.clear();
	_indice.clear();
	_rank_cnt = 0;
}

template <typename _T>
NN_Shape Tensor<_T>::get_shape() {
	_rank_cnt = 0;

	return calc_shape(_indice);
}

template <typename _T>
NN_Shape Tensor<_T>::get_shape() const {
	return calc_shape(_indice);
}

template <typename _T>
_T* Tensor<_T>::get_ptr() {
	_rank_cnt = 0;

	return _data.get();
}

template <typename _T>
const _T* Tensor<_T>::get_ptr() const {
	_rank_cnt = 0;

	return _data.get();
}

template <typename _T>
std::ostream& Tensor<_T>::put(std::ostream& os) {
	_rank_cnt = 0;

	if (_indice.size() > 0) {
		int i = 0;

		put_tensor(os, *this, 0, i);
		os << shape_to_str(calc_shape(_indice)) << std::endl;
	}
	else os << "[]" << std::endl;

	return os;
}

template <typename _T>
void Tensor<_T>::resize(const NN_Shape& shape) {
	_rank_cnt = 0;

	if ((int)_indice.size() != shape.get_len()) _indice.resize(shape.get_len());

	int i = 0;
	for (const int& n : shape) {
		if (n < 1) {
			ErrorExcept(
				"[Tensor<_T>::Tensor] Shape must be greater than 0. but %s",
				shape_to_str(shape)
			);
		}

		std::vector<size_t> m_indice(n, 0);

		size_t j = 0;
		for (size_t& m : m_indice) m = j++;

		_indice[i++] = m_indice;
	}

	_steps = calc_step(shape);
	_data = std::shared_ptr<_T[]>(new _T[shape.total_size()]);
}


/**********************************************/
/*                                            */
/*                  GpuTensor                 */
/*                                            */
/**********************************************/

template <typename _T>
class GpuTensor {
	std::shared_ptr<_T> _data;
	NN_Shape _shape;

	static void del_func(_T* ptr);

public:
	GpuTensor();
	GpuTensor(const NN_Shape& shape);
	GpuTensor(const GpuTensor& p);
	GpuTensor(GpuTensor&& p);
	~GpuTensor();

	GpuTensor& operator=(const GpuTensor& p);
	GpuTensor& operator=(GpuTensor&& p);
	GpuTensor& operator=(Tensor<_T>& p);

	NN_Shape& get_shape();
	const NN_Shape& get_shape() const;
	_T* get_ptr() const;

	void reshape(const NN_Shape& shape);
	void resize(const NN_Shape& shape);

	static GpuTensor<_T> zeros(const NN_Shape& shape);
};

template <typename _T>
void GpuTensor<_T>::del_func(_T* ptr) {
	cudaFree(ptr);
}

template <typename _T>
GpuTensor<_T>::GpuTensor() {
}

template <typename _T>
GpuTensor<_T>::GpuTensor(const NN_Shape& shape) :
	_shape(shape)
{
	_T* ptr = NULL;

	check_cuda(cudaMalloc(&ptr, sizeof(_T) * _shape.total_size()));

	_data.reset(ptr, del_func);
}

template <typename _T>
GpuTensor<_T>::GpuTensor(const GpuTensor& p) :
	_data(p._data),
	_shape(p._shape)
{
}

template <typename _T>
GpuTensor<_T>::GpuTensor(GpuTensor&& p) :
	_data(std::move(p._data)),
	_shape(std::move(p._shape))
{
}

template <typename _T>
GpuTensor<_T>::~GpuTensor() {

}

template <typename _T>
GpuTensor<_T>& GpuTensor<_T>::operator=(const GpuTensor& p) {
	if (this == &p) return *this;

	_data = p._data;
	_shape = p._shape;

	return *this;
}

template <typename _T>
GpuTensor<_T>& GpuTensor<_T>::operator=(GpuTensor&& p) {
	if (this == &p) return *this;

	_data = std::move(p._data);
	_shape = std::move(p._shape);

	return *this;
}

template <typename _T>
GpuTensor<_T>& GpuTensor<_T>::operator=(Tensor<_T>& p) {
	const NN_Shape h_shape = p.get_shape();

	if (h_shape != _shape) {
		ErrorExcept(
			"[GpuTensor<_T>::operator=] GPU and Host shape are different. GPU: %s, Host: %s",
			shape_to_str(_shape),
			shape_to_str(h_shape)
		);
	}

	Tensor<_T> tmp(h_shape);

	tmp = p;

	_T* h_ptr = tmp.get_ptr();
	_T* g_ptr = get_ptr();

	check_cuda(cudaMemcpy(g_ptr, h_ptr, sizeof(_T) * h_shape.total_size(), cudaMemcpyHostToDevice));

	return *this;
}

template <typename _T>
NN_Shape& GpuTensor<_T>::get_shape() {
	return _shape;
}

template <typename _T>
const NN_Shape& GpuTensor<_T>::get_shape() const {
	return _shape;
}

template <typename _T>
_T* GpuTensor<_T>::get_ptr() const {
	return _data.get();
}

template <typename _T>
void GpuTensor<_T>::reshape(const NN_Shape& shape) {
	if (_shape.total_size() != shape.total_size()) {
		ErrorExcept(
			"[GpuTensor<_T>::reshape] Can't reshape tensor. %s != %s",
			shape_to_str(_shape),
			shape_to_str(shape)
		);
	}

	_shape = shape;
}

template <typename _T>
void GpuTensor<_T>::resize(const NN_Shape& shape) {
	const size_t size = shape.total_size();

	if (size == 0) {
		ErrorExcept(
			"[GpuTensor<_T>::resize] Can't resize this shape. %s",
			shape_to_str(shape)
		);
	}

	_T* ptr = NULL;

	_shape = shape;
	check_cuda(cudaMalloc(&ptr, sizeof(_T) * size));
	_data = std::shared_ptr<_T>(ptr, del_func);
}

template <typename _T>
GpuTensor<_T> GpuTensor<_T>::zeros(const NN_Shape& shape) {
	GpuTensor<_T> tmp(shape);

	check_cuda(cudaMemset(tmp.get_ptr(), 0, sizeof(_T) * shape.total_size()));

	return tmp;
}
