#pragma once
#include <tbb/tbb.h>
#include "nn_list.h"
#include "nn_shape.h"
#include "nn_common.h"


/**********************************************/
/*                                            */
/*                  NN_Tensor                 */
/*                                            */
/**********************************************/

template <typename _H>
class HostTensor;

template <typename _T>
class NN_Tensor {
	std::vector<uint> _steps;
	std::shared_ptr<_T[]> _data;
	std::vector<std::vector<uint>> _indices;

	int* _cnt_rank;

	struct Iterators {
		std::vector<uint>::const_reverse_iterator _steps_criter;
		std::vector<uint>::const_reverse_iterator _steps_crend;
		std::vector<std::vector<uint>>::const_reverse_iterator _indices_criter;
	};

	static std::vector<uint> calc_step(const NN_Shape& shape);
	static bool is_valid_src_shape(const NN_Shape& dst_shape, const NN_Shape& src_shape);
	static std::vector<std::vector<uint>> calc_indices(const NN_Shape& shape);
	static size_t calc_size(const std::vector<std::vector<uint>>& indice);
	static typename Iterators get_iterators(const NN_Tensor& tensor_base);
	static size_t count_to_elem_index(typename Iterators& iters, size_t count);
	static NN_Shape calc_shape(const std::vector<std::vector<uint>>& indice);
	static std::vector<uint> set_indice(const std::vector<uint>& indices, int begin, int end, int step);

	static void put_tensor(std::ostream& os, const NN_Tensor& tensor, size_t offset, int& rank);

	NN_Tensor(const NN_Tensor& p, int cnt_rank);

public:
	NN_Tensor();
	NN_Tensor(const NN_Shape& shape);
	NN_Tensor(const int* p_dims, int n_dims);
	NN_Tensor(const size_t* p_dims, int n_dims);
	NN_Tensor(const std::initializer_list<int>& shape);
	NN_Tensor(const NN_Tensor& p);
	NN_Tensor(NN_Tensor&& p);
	NN_Tensor(const std::shared_ptr<_T[]>& data, const NN_Shape& shape);
	NN_Tensor(const HostTensor<_T>& p);
	~NN_Tensor();

	NN_Tensor& operator=(const HostTensor<_T>& p);
	NN_Tensor& operator=(const NN_Tensor& p);
	NN_Tensor& operator=(_T scalar);
	NN_Tensor operator()(int begin, int end, int step = 1);
	NN_Tensor operator()(int begin, int end, int step = 1) const;
	NN_Tensor operator()(int index);
	NN_Tensor operator()(int index) const;
	NN_Tensor operator()(const std::vector<int>& indice);
	NN_Tensor operator()(const std::vector<int>& indice) const;
	NN_Tensor operator[](int index);
	NN_Tensor operator[](int index) const;

	NN_Tensor operator+(const NN_Tensor& p);
	NN_Tensor operator-(const NN_Tensor& p);
	NN_Tensor operator*(const NN_Tensor& p);
	NN_Tensor operator/(const NN_Tensor& p);

	NN_Tensor operator+(const _T& val);
	NN_Tensor operator-(const _T& val);
	NN_Tensor operator*(const _T& val);
	NN_Tensor operator/(const _T& val);

	void operator+=(const NN_Tensor& p);
	void operator-=(const NN_Tensor& p);
	void operator*=(const NN_Tensor& p);
	void operator/=(const NN_Tensor& p);

	void operator+=(const _T& val);
	void operator-=(const _T& val);
	void operator*=(const _T& val);
	void operator/=(const _T& val);

	NN_Tensor inverse(_T val);

	NN_Tensor transpose(const std::initializer_list<int>& orders);
	NN_Tensor swap_pose();

	_T& val();
	const _T& val() const;

	void clear();
	std::vector<std::vector<uint>>& get_indices();
	const std::vector<std::vector<uint>>& get_indices() const;
	NN_Shape get_shape();
	NN_Shape get_shape() const;
	_T* get_ptr();
	const _T* get_ptr() const;

	std::ostream& put(std::ostream& os);
	std::ostream& put(std::ostream& os) const;

	void resize(const NN_Shape& shape);
	NN_Tensor reshape(const NN_Shape& shape);

	template <typename _cT>
	NN_Tensor<_cT> cast() const;

	static NN_Tensor expand_dims(const NN_Tensor& tensor, int axis = 0);
	static NN_Tensor expand_dims(const NN_Tensor& tensor, std::initializer_list<int>& axis);

	static NN_Tensor squeeze(const NN_Tensor& tensor, int axis = 0);
	static NN_Tensor squeeze(const NN_Tensor& tensor, std::initializer_list<int>& axis);

	static NN_Tensor zeros(const NN_Shape& shape);

	std::shared_ptr<_T[]> get_shared_ptr();
	const std::shared_ptr<_T[]> get_shared_ptr() const;
};

template <typename _T>
std::vector<uint> NN_Tensor<_T>::calc_step(const NN_Shape& shape) {
	std::vector<uint> steps(shape.ranks(), 1);
	std::vector<uint>::reverse_iterator steps_iter = steps.rbegin();

	for (NN_Shape::cr_iter i = shape.rbegin(); i != shape.rend(); ++i) {
		for (NN_Shape::cr_iter j = shape.rbegin(); j != i; ++j) *steps_iter *= (uint)(*j);
		++steps_iter;
	}

	return steps;
}

template <typename _T>
bool NN_Tensor<_T>::is_valid_src_shape(const NN_Shape& dst_shape, const NN_Shape& src_shape) {
	bool is_valid = true;
	const int dst_len = dst_shape.ranks();
	const int src_len = src_shape.ranks();

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
std::vector<std::vector<uint>> NN_Tensor<_T>::calc_indices(const NN_Shape& shape) {
	std::vector<std::vector<uint>> indices;

	indices.reserve((size_t)shape.ranks());

	for (const int& n : shape) {
		if (n < 1) {
			ErrorExcept(
				"[TensorBase<_T>::calc_indices] Shape must be greater than 0. but %s",
				shape_to_str(shape)
			);
		}

		uint i = 0;
		std::vector<uint> indice(n, 0);

		for (uint& j : indice) j = i++;

		indices.push_back(indice);
	}

	return indices;
}

template <typename _T>
size_t NN_Tensor<_T>::calc_size(const std::vector<std::vector<uint>>& indice) {
	size_t size = indice.size() > 0 ? 1 : 0;

	for (const std::vector<uint>& m : indice) {
		size *= m.size();
	}

	return size;
}

template <typename _T>
typename NN_Tensor<_T>::Iterators NN_Tensor<_T>::get_iterators(const NN_Tensor& tensor_base) {
	NN_Tensor<_T>::Iterators iters;

	iters._steps_criter = tensor_base._steps.crbegin();
	iters._steps_crend = tensor_base._steps.crend();
	iters._indices_criter = tensor_base._indices.crbegin();

	return iters;
}

template <typename _T>
size_t NN_Tensor<_T>::count_to_elem_index(typename Iterators& iters, size_t count) {
	size_t index = 0;

	while (iters._steps_criter != iters._steps_crend) {
		cuint curr_dims = (uint)(*iters._indices_criter).size();

		index += (*iters._steps_criter) * (*iters._indices_criter)[count % curr_dims];
		count /= curr_dims;

		++iters._steps_criter;
		++iters._indices_criter;
	}

	return index;
}

template <typename _T>
NN_Shape NN_Tensor<_T>::calc_shape(const std::vector<std::vector<uint>>& indice) {
	NN_Shape shape((int)indice.size());

	int i = 0;
	for (const std::vector<uint>& m_indice : indice) {
		shape[i++] = (int)m_indice.size();
	}

	return shape;
}

template <typename _T>
std::vector<uint> NN_Tensor<_T>::set_indice(const std::vector<uint>& indices, int begin, int end, int step) {
	const int n = (end - begin + step - 1) / step;
	std::vector<uint> m_indices(n, 0);

	int i = 0;
	for (uint& m : m_indices) m = indices[begin + step * i++];

	return m_indices;
}

template <typename _T>
void NN_Tensor<_T>::put_tensor(std::ostream& os, const NN_Tensor& tensor, size_t offset, int& rank) {
	cuint step = tensor._steps[rank];

	if (rank < tensor._indices.size() - 1) {
		for (cuint& index : tensor._indices[rank]) {
			os << '[';
			put_tensor(os, tensor, offset + (index * step), ++rank);
			--rank;
			os << ']' << std::endl;
		}
	}
	else {
		for (cuint& index : tensor._indices[rank]) {
			os << tensor._data[offset + (index * step)] << ", ";
		}
	}
}

template <typename _T>
using tensor_t = std::shared_ptr<_T[]>;

template <typename _T>
std::ostream& operator<<(std::ostream& os, const NN_Tensor<_T>& tensor) {
	tensor.put(os);

	return os;
}

template <typename _T>
NN_Tensor<_T>::NN_Tensor(const NN_Tensor& p, int cnt_rank) :
	_data(p._data),
	_steps(p._steps),
	_indices(p._indices),
	_cnt_rank(new int(cnt_rank))
{
}

template <typename _T>
NN_Tensor<_T>::NN_Tensor(const std::shared_ptr<_T[]>& data, const NN_Shape& shape) :
	_data(data),
	_cnt_rank(new int(0))
{
	_steps = calc_step(shape);
	_indices = calc_indices(shape);
}

template <typename _T>
NN_Tensor<_T>::NN_Tensor() :
	_cnt_rank(new int(0))
{

}

template <typename _T>
NN_Tensor<_T>::NN_Tensor(const NN_Shape& shape) :
	_cnt_rank(new int(0))
{
	_steps = calc_step(shape);
	_indices = calc_indices(shape);
	_data = std::shared_ptr<_T[]>(new _T[shape.total_size()]);
}

template <typename _T>
NN_Tensor<_T>::NN_Tensor(const int* p_dims, int n_dims) :
	_cnt_rank(new int(0))
{
	NN_Shape shape(p_dims, n_dims);

	_steps = calc_step(shape);
	_indices = calc_indices(shape);
	_data = std::shared_ptr<_T[]>(new _T[shape.total_size()]);
}

template <typename _T>
NN_Tensor<_T>::NN_Tensor(const size_t* p_dims, int n_dims) :
	_cnt_rank(new int(0))
{
	int* dims = new int[n_dims];

	for (int i = 0; i < n_dims; ++i) dims[i] = (int)(p_dims[i]);

	NN_Shape shape(dims, n_dims);

	_steps = calc_step(shape);
	_indices = calc_indices(shape);
	_data = std::shared_ptr<_T[]>(new _T[shape.total_size()]);

	delete[] dims;
}


template <typename _T>
NN_Tensor<_T>::NN_Tensor(const std::initializer_list<int>& shape) :
	_cnt_rank(new int(0))
{
	NN_Shape m_shape(shape);

	_steps = calc_step(m_shape);
	_indices = calc_indices(m_shape);
	_data = std::shared_ptr<_T[]>(new _T[m_shape.total_size()]);
}

template <typename _T>
NN_Tensor<_T>::NN_Tensor(const NN_Tensor& p) :
	_steps(p._steps),
	_indices(p._indices),
	_data(p._data),
	_cnt_rank(p._cnt_rank)
{

}

template <typename _T>
NN_Tensor<_T>::NN_Tensor(NN_Tensor&& p) :
	_steps(std::move(p._steps)),
	_indices(std::move(p._indices)),
	_data(std::move(p._data)),
	_cnt_rank(p._cnt_rank)
{
	p._cnt_rank = NULL;
}

template <typename _T>
NN_Tensor<_T>::NN_Tensor(const HostTensor<_T>& p) :
	_cnt_rank(new int(0))
{
	_steps = calc_step(p.get_shape());
	_indices = calc_indices(p.get_shape());
	_data = p.get_shared_ptr();
}

template <typename _T>
NN_Tensor<_T>::~NN_Tensor() {
	delete _cnt_rank;
}

template <typename _T>
NN_Tensor<_T>& NN_Tensor<_T>::operator=(const HostTensor<_T>& p) {
	*_cnt_rank = 0;
	
	if (_data == NULL) {
		_data = p.get_shared_ptr();
	}
	else {
		const NN_Shape origin_shape = calc_shape(_indices);
		const NN_Shape source_shape = p.get_shape();

		if (!is_valid_src_shape(origin_shape, source_shape)) {
			ErrorExcept(
				"[NN_Tensor<_T>::operator=] Shape of Right operand are unsuitable. %s.",
				shape_to_str(source_shape)
			);
		}

		const size_t src_size = source_shape.total_size();
		const size_t dst_size = origin_shape.total_size();
		const _T* src_data = p.get_ptr();

		tbb::parallel_for<tbb::blocked_range<size_t>>(
			tbb::blocked_range<size_t>(0, dst_size),
			[&](const tbb::blocked_range<size_t>& q) {
			Iterators src_iters = get_iterators(p);
			Iterators dst_iters = get_iterators(*this);

			for (size_t i = q.begin(); i < q.end(); ++i) {
				const size_t src_idx = count_to_elem_index(src_iters, i % src_size);
				const size_t dst_idx = count_to_elem_index(dst_iters, i);

				_data[dst_idx] = src_data[src_idx];
			}
		},
			tbb::auto_partitioner()
			);
	}
	_steps = calc_step(p.get_shape());
	_indices = calc_indices(p.get_shape());

	return *this;
}

template <typename _T>
NN_Tensor<_T>& NN_Tensor<_T>::operator=(const NN_Tensor& p) {
	if (this == &p) return *this;

	if (_data == NULL) {
		_data = p._data;
		_steps = p._steps;
		_indices = p._indices;
		*_cnt_rank = *p._cnt_rank;
	}
	else {
		const NN_Shape origin_shape = calc_shape(_indices);
		const NN_Shape source_shape = calc_shape(p._indices);

		if (!is_valid_src_shape(origin_shape, source_shape)) {
			ErrorExcept(
				"[NN_Tensor<_T>::operator=] Shape of Right operand are unsuitable. %s.",
				shape_to_str(source_shape)
			);
		}

		const size_t src_size = source_shape.total_size();
		const size_t dst_size = origin_shape.total_size();

		tbb::parallel_for<tbb::blocked_range<size_t>>(
			tbb::blocked_range<size_t>(0, dst_size),
			[&](const tbb::blocked_range<size_t>& q) {
				Iterators src_iters = get_iterators(p);
				Iterators dst_iters = get_iterators(*this);
				const _T* src_data = p._data.get();

				for (size_t i = q.begin(); i < q.end(); ++i) {
					const size_t src_idx = count_to_elem_index(src_iters, i % src_size);
					const size_t dst_idx = count_to_elem_index(dst_iters, i);

					_data[dst_idx] = src_data[src_idx];
				}
			},
			tbb::auto_partitioner()
		);
	}

	*_cnt_rank = 0;
	*(p._cnt_rank) = 0;

	return *this;
}

template <typename _T>
NN_Tensor<_T>& NN_Tensor<_T>::operator=(_T scalar) {
	if (_data == NULL) {
		_data = std::shared_ptr<_T[]>(new _T[1]);
		_steps.clear();
		_steps.push_back(1);
		_indices.clear();
		_indices.push_back({ 0 });

		_data[0] = scalar;
	}
	else {
		const size_t size = calc_size(_indices);

		tbb::parallel_for<tbb::blocked_range<size_t>>(
			tbb::blocked_range<size_t>(0, size),
			[&](const tbb::blocked_range<size_t>& q) {
				Iterators iters = get_iterators(*this);

				for (size_t i = q.begin(); i < q.end(); ++i) {
					const size_t index = count_to_elem_index(iters, i);

					_data[index] = scalar;
				}
			},
			tbb::auto_partitioner()
		);
	}

	*_cnt_rank = 0;

	return *this;
}

template <typename _T>
NN_Tensor<_T> NN_Tensor<_T>::operator()(int begin, int end, int step) {
	if (*_cnt_rank >= _indices.size()) {
		ErrorExcept(
			"[NN_Tensor<_T>::operator()] %d rank is empty.",
			_cnt_rank
		);
	}

	const int n = (int)_indices[*_cnt_rank].size();

	begin = begin < 0 ? n + begin : begin;
	end = end < 0 ? n + end : end;

	if (begin < 0 || begin >= n || end < 0 || end > n) {
		ErrorExcept(
			"[NN_Tensor<_T>::operator()] begin and end is out of range. begin: %d, end: %d, step: %d",
			begin, end, step
		);
	}

	NN_Tensor<_T> tensor(*this, *_cnt_rank + 1);

	set_indice(tensor._indices[*_cnt_rank], begin, end, step);
	*_cnt_rank = 0;

	return tensor;
}

template <typename _T>
NN_Tensor<_T> NN_Tensor<_T>::operator()(int begin, int end, int step) const {
	if (*_cnt_rank >= _indices.size()) {
		ErrorExcept(
			"[NN_Tensor<_T>::operator()] %d rank is empty.",
			_cnt_rank
		);
	}

	const int n = (int)_indices[*_cnt_rank].size();

	begin = begin < 0 ? n + begin : begin;
	end = end < 0 ? n + end : end;

	if (begin < 0 || begin >= n || end < 0 || end > n) {
		ErrorExcept(
			"[NN_Tensor<_T>::operator()] begin and end is out of range. begin: %d, end: %d, step: %d",
			begin, end, step
		);
	}

	NN_Tensor<_T> tensor(*this, *_cnt_rank + 1);

	set_indice(tensor._indices[*_cnt_rank], begin, end, step);
	*_cnt_rank = 0;

	return tensor;
}

template <typename _T>
NN_Tensor<_T> NN_Tensor<_T>::operator()(int index) {
	if (*_cnt_rank >= _indices.size()) {
		ErrorExcept(
			"[NN_Tensor<_T>::operator()] %d rank is empty.",
			_cnt_rank
		);
	}

	const int n = (int)_indices[*_cnt_rank].size();

	index = index < 0 ? n + index : index;

	if (index < 0 || index >= n) {
		ErrorExcept(
			"[NN_Tensor<_T>::operator()] index is out of range."
		);
	}

	NN_Tensor<_T> tensor(*this, *_cnt_rank + 1);

	set_indice(tensor._indices[*_cnt_rank], index, index + 1, 1);
	*_cnt_rank = 0;

	return tensor;
}

template <typename _T>
NN_Tensor<_T> NN_Tensor<_T>::operator()(int index) const {
	if (*_cnt_rank >= _indices.size()) {
		ErrorExcept(
			"[NN_Tensor<_T>::operator()] %d rank is empty.",
			_cnt_rank
		);
	}

	const int n = (int)_indices[*_cnt_rank].size();

	index = index < 0 ? n + index : index;

	if (index < 0 || index >= n) {
		ErrorExcept(
			"[NN_Tensor<_T>::operator()] index is out of range."
		);
	}

	NN_Tensor<_T> tensor(*this, *_cnt_rank + 1);

	set_indice(tensor._indices[*_cnt_rank], index, index + 1, 1);
	*_cnt_rank = 0;

	return tensor;
}

template <typename _T>
NN_Tensor<_T> NN_Tensor<_T>::operator()(const std::vector<int>& indice) {
	if (*_cnt_rank >= _indices.size()) {
		ErrorExcept(
			"[NN_Tensor<_T>::operator()] %d rank is empty.",
			_cnt_rank
		);
	}

	const std::vector<uint>& curr_indice = _indices[*_cnt_rank];
	const int n = (int)curr_indice.size();
	std::vector<uint> m_indice(indice.size(), 0);

	int i = 0;
	for (const int& m : indice) {
		int index = m < 0 ? n + m : m;

		if (index < 0 || index >= n) {
			ErrorExcept(
				"[NN_Tensor<_T>::operator()] indice is out of range."
			);
		}

		m_indice[i++] = curr_indice[index];
	}

	NN_Tensor<_T> tensor(*this, *_cnt_rank + 1);

	tensor._indices[*_cnt_rank] = m_indice;
	*_cnt_rank = 0;

	return tensor;
}

template <typename _T>
NN_Tensor<_T> NN_Tensor<_T>::operator()(const std::vector<int>& indice) const {
	if (*_cnt_rank >= _indices.size()) {
		ErrorExcept(
			"[NN_Tensor<_T>::operator()] %d rank is empty.",
			_cnt_rank
		);
	}

	const std::vector<uint>& curr_indice = _indices[*_cnt_rank];
	const int n = (int)curr_indice.size();
	std::vector<uint> m_indice(indice.size(), 0);

	int i = 0;
	for (const int& m : indice) {
		int index = m < 0 ? n + m : m;

		if (index < 0 || index >= n) {
			ErrorExcept(
				"[NN_Tensor<_T>::operator()] indice is out of range."
			);
		}

		m_indice[i++] = curr_indice[index];
	}

	NN_Tensor<_T> tensor(*this, *_cnt_rank + 1);

	tensor._indices[*_cnt_rank] = m_indice;
	*_cnt_rank = 0;

	return tensor;
}

template <typename _T>
NN_Tensor<_T> NN_Tensor<_T>::operator[](int index) {
	if (*_cnt_rank >= _indices.size()) {
		ErrorExcept(
			"[NN_Tensor<_T>::operator()] %d rank is empty.",
			*_cnt_rank
		);
	}

	const int n = (int)_indices[*_cnt_rank].size();

	index = index < 0 ? n + index : index;

	if (index < 0 || index >= n) {
		ErrorExcept(
			"[NN_Tensor<_T>::operator()] begin and end is out of range."
		);
	}

	NN_Tensor<_T> tensor(*this, *_cnt_rank + 1);

	set_indice(tensor._indices[*_cnt_rank], index, index + 1, 1);
	*_cnt_rank = 0;

	return tensor;
}

template <typename _T>
NN_Tensor<_T> NN_Tensor<_T>::operator[](int index) const {
	if (*_cnt_rank >= _indices.size()) {
		ErrorExcept(
			"[NN_Tensor<_T>::operator()] %d rank is empty.",
			_cnt_rank
		);
	}

	const int n = (int)_indices[*_cnt_rank].size();

	index = index < 0 ? n + index : index;

	if (index < 0 || index >= n) {
		ErrorExcept(
			"[NN_Tensor<_T>::operator()] begin and end is out of range."
		);
	}

	NN_Tensor<_T> tensor(*this, *_cnt_rank + 1);

	set_indice(tensor._indices[*_cnt_rank], index, index + 1, 1);
	*_cnt_rank = 0;

	return tensor;
}

template <typename _T>
NN_Tensor<_T> NN_Tensor<_T>::operator+(const NN_Tensor& p) {
	NN_Tensor<_T> value(calc_shape(_indices));

	if (_data != NULL && p._data != NULL) {
		const NN_Shape origin_shape = calc_shape(_indices);
		const NN_Shape source_shape = calc_shape(p._indices);

		if (!is_valid_src_shape(origin_shape, source_shape)) {
			ErrorExcept(
				"[NN_Tensor<_T>::operator+] Shape of Right operand are unsuitable. %s.",
				shape_to_str(source_shape)
			);
		}

		const size_t src_size = calc_size(p._indices);
		const size_t dst_size = calc_size(_indices);

		tbb::parallel_for<tbb::blocked_range<size_t>>(
			tbb::blocked_range<size_t>(0, dst_size),
			[&](const tbb::blocked_range<size_t>& q) {
				Iterators src_iters = get_iterators(p);
				Iterators dst_iters = get_iterators(*this);

				for (size_t i = q.begin(); i < q.end(); ++i) {
					const size_t src_idx = count_to_elem_index(src_iters, i % src_size);
					const size_t dst_idx = count_to_elem_index(dst_iters, i);

					value._data[dst_idx] = _data[dst_idx] + p._data[src_idx];
				}
			},
			tbb::auto_partitioner()
		);
	}
	else {
		ErrorExcept(
			"[NN_Tensor<_T>::operator+] None tensor can't this operator."
		);
	}

	*_cnt_rank = 0;

	return value;
}

template <typename _T>
NN_Tensor<_T> NN_Tensor<_T>::operator-(const NN_Tensor& p) {
	NN_Tensor<_T> value(calc_shape(_indices));

	if (_data != NULL && p._data != NULL) {
		const NN_Shape origin_shape = calc_shape(_indices);
		const NN_Shape source_shape = calc_shape(p._indices);

		if (!is_valid_src_shape(origin_shape, source_shape)) {
			ErrorExcept(
				"[NN_Tensor<_T>::operator-] Shape of Right operand are unsuitable. %s.",
				shape_to_str(source_shape)
			);
		}

		const size_t src_size = calc_size(p._indices);
		const size_t dst_size = calc_size(_indices);

		tbb::parallel_for<tbb::blocked_range<size_t>>(
			tbb::blocked_range<size_t>(0, dst_size),
			[&](const tbb::blocked_range<size_t>& q) {
				Iterators src_iters = get_iterators(p);
				Iterators dst_iters = get_iterators(*this);

				for (size_t i = q.begin(); i < q.end(); ++i) {
					const size_t src_idx = count_to_elem_index(src_iters, i % src_size);
					const size_t dst_idx = count_to_elem_index(dst_iters, i);

					value._data[dst_idx] = _data[dst_idx] - p._data[src_idx];
				}
			},
			tbb::auto_partitioner()
		);
	}
	else {
		ErrorExcept(
			"[NN_Tensor<_T>::operator-] None tensor can't this operator."
		);
	}

	*_cnt_rank = 0;

	return value;
}

template <typename _T>
NN_Tensor<_T> NN_Tensor<_T>::operator*(const NN_Tensor& p) {
	NN_Tensor<_T> value(calc_shape(_indices));

	if (_data != NULL && p._data != NULL) {
		const NN_Shape origin_shape = calc_shape(_indices);
		const NN_Shape source_shape = calc_shape(p._indices);

		if (!is_valid_src_shape(origin_shape, source_shape)) {
			ErrorExcept(
				"[NN_Tensor<_T>::operator*] Shape of Right operand are unsuitable. %s.",
				shape_to_str(source_shape)
			);
		}

		const size_t src_size = calc_size(p._indices);
		const size_t dst_size = calc_size(_indices);

		tbb::parallel_for<tbb::blocked_range<size_t>>(
			tbb::blocked_range<size_t>(0, dst_size),
			[&](const tbb::blocked_range<size_t>& q) {
				Iterators src_iters = get_iterators(p);
				Iterators dst_iters = get_iterators(*this);

				for (size_t i = q.begin(); i < q.end(); ++i) {
					const size_t src_idx = count_to_elem_index(src_iters, i % src_size);
					const size_t dst_idx = count_to_elem_index(dst_iters, i);

					value._data[dst_idx] = _data[dst_idx] * p._data[src_idx];
				}
			},
			tbb::auto_partitioner()
		);
	}
	else {
		ErrorExcept(
			"[NN_Tensor<_T>::operator*] None tensor can't this operator."
		);
	}

	*_cnt_rank = 0;

	return value;
}

template <typename _T>
NN_Tensor<_T> NN_Tensor<_T>::operator/(const NN_Tensor& p) {
	NN_Tensor<_T> value(calc_shape(_indices));

	if (_data != NULL && p._data != NULL) {
		const NN_Shape origin_shape = calc_shape(_indices);
		const NN_Shape source_shape = calc_shape(p._indices);

		if (!is_valid_src_shape(origin_shape, source_shape)) {
			ErrorExcept(
				"[NN_Tensor<_T>::operator/] Shape of Right operand are unsuitable. %s.",
				shape_to_str(source_shape)
			);
		}

		const size_t src_size = calc_size(p._indices);
		const size_t dst_size = calc_size(_indices);

		tbb::parallel_for<tbb::blocked_range<size_t>>(
			tbb::blocked_range<size_t>(0, dst_size),
			[&](const tbb::blocked_range<size_t>& q) {
				Iterators src_iters = get_iterators(p);
				Iterators dst_iters = get_iterators(*this);

				for (size_t i = q.begin(); i < q.end(); ++i) {
					const size_t src_idx = count_to_elem_index(src_iters, i % src_size);
					const size_t dst_idx = count_to_elem_index(dst_iters, i);

					value._data[dst_idx] = _data[dst_idx] / p._data[src_idx];
				}
			},
			tbb::auto_partitioner()
		);
	}
	else {
		ErrorExcept(
			"[NN_Tensor<_T>::operator/] None tensor can't this operator."
		);
	}

	*_cnt_rank = 0;

	return value;
}

template <typename _T>
NN_Tensor<_T> NN_Tensor<_T>::operator+(const _T& val) {
	NN_Tensor<_T> value(calc_shape(_indices));

	if (_data != NULL) {
		const size_t dst_size = calc_size(_indices);

		tbb::parallel_for<tbb::blocked_range<size_t>>(
			tbb::blocked_range<size_t>(0, dst_size),
			[&](const tbb::blocked_range<size_t>& q) {
				Iterators iters = get_iterators(*this);

				for (size_t i = q.begin(); i < q.end(); ++i) {
					const size_t dst_idx = count_to_elem_index(iters, i);

					value._data[dst_idx] = _data[dst_idx] + val;
				}
			},
			tbb::auto_partitioner()
		);
	}
	else {
		ErrorExcept(
			"[NN_Tensor<_T>::operator+] None tensor can't this operator."
		);
	}

	*_cnt_rank = 0;

	return value;
}

template <typename _T>
NN_Tensor<_T> NN_Tensor<_T>::operator-(const _T& val) {
	NN_Tensor<_T> value(calc_shape(_indices));

	if (_data != NULL) {
		const size_t dst_size = calc_size(_indices);

		tbb::parallel_for<tbb::blocked_range<size_t>>(
			tbb::blocked_range<size_t>(0, dst_size),
			[&](const tbb::blocked_range<size_t>& q) {
				Iterators iters = get_iterators(*this);
				
				for (size_t i = q.begin(); i < q.end(); ++i) {
					const size_t dst_idx = count_to_elem_index(iters, i);

					value._data[dst_idx] = _data[dst_idx] - val;
				}
			},
			tbb::auto_partitioner()
		);
	}
	else {
		ErrorExcept(
			"[NN_Tensor<_T>::operator-] None tensor can't this operator."
		);
	}

	*_cnt_rank = 0;

	return value;
}

template <typename _T>
NN_Tensor<_T> NN_Tensor<_T>::operator*(const _T& val) {
	NN_Tensor<_T> value(calc_shape(_indices));

	if (_data != NULL) {
		const size_t dst_size = calc_size(_indices);

		tbb::parallel_for<tbb::blocked_range<size_t>>(
			tbb::blocked_range<size_t>(0, dst_size),
			[&](const tbb::blocked_range<size_t>& q) {
				Iterators iters = get_iterators(*this);

				for (size_t i = q.begin(); i < q.end(); ++i) {
					const size_t dst_idx = count_to_elem_index(iters, i);

					value._data[dst_idx] = _data[dst_idx] * val;
				}
			},
			tbb::auto_partitioner()
		);
	}
	else {
		ErrorExcept(
			"[NN_Tensor<_T>::operator*] None tensor can't this operator."
		);
	}

	*_cnt_rank = 0;

	return value;
}

template <typename _T>
NN_Tensor<_T> NN_Tensor<_T>::operator/(const _T& val) {
	NN_Tensor<_T> value(calc_shape(_indices));

	if (_data != NULL) {
		const size_t dst_size = calc_size(_indices);

		tbb::parallel_for<tbb::blocked_range<size_t>>(
			tbb::blocked_range<size_t>(0, dst_size),
			[&](const tbb::blocked_range<size_t>& q) {
				Iterators iters = get_iterators(*this);

				for (size_t i = q.begin(); i < q.end(); ++i) {
					const size_t dst_idx = count_to_elem_index(iters, i);

					value._data[dst_idx] = _data[dst_idx] / val;
				}
			},
			tbb::auto_partitioner()
		);
	}
	else {
		ErrorExcept(
			"[NN_Tensor<_T>::operator/] None tensor can't this operator."
		);
	}

	*_cnt_rank = 0;

	return value;
}

template <typename _T>
void NN_Tensor<_T>::operator+=(const NN_Tensor& p) {
	if (_data != NULL && p._data != NULL) {
		const NN_Shape origin_shape = calc_shape(_indices);
		const NN_Shape source_shape = calc_shape(p._indices);

		if (!is_valid_src_shape(origin_shape, source_shape)) {
			ErrorExcept(
				"[NN_Tensor<_T>::operator+=] Shape of Right operand are unsuitable. %s.",
				shape_to_str(source_shape)
			);
		}

		const size_t src_size = calc_size(p._indices);
		const size_t dst_size = calc_size(_indices);

		tbb::parallel_for<tbb::blocked_range<size_t>>(
			tbb::blocked_range<size_t>(0, dst_size),
			[&](const tbb::blocked_range<size_t>& q) {
				Iterators src_iters = get_iterators(p);
				Iterators dst_iters = get_iterators(*this);

				for (size_t i = q.begin(); i < q.end(); ++i) {
					const size_t src_idx = count_to_elem_index(src_iters, i % src_size);
					const size_t dst_idx = count_to_elem_index(dst_iters, i);

					_data[dst_idx] += p._data[src_idx];
				}
			},
			tbb::auto_partitioner()
		);
	}
	else {
		ErrorExcept(
			"[NN_Tensor<_T>::operator+=] None tensor can't this operator."
		);
	}

	*_cnt_rank = 0;
}

template <typename _T>
void NN_Tensor<_T>::operator-=(const NN_Tensor& p) {
	if (_data != NULL && p._data != NULL) {
		const NN_Shape origin_shape = calc_shape(_indices);
		const NN_Shape source_shape = calc_shape(p._indices);

		if (!is_valid_src_shape(origin_shape, source_shape)) {
			ErrorExcept(
				"[NN_Tensor<_T>::operator-=] Shape of Right operand are unsuitable. %s.",
				shape_to_str(source_shape)
			);
		}

		const size_t src_size = calc_size(p._indices);
		const size_t dst_size = calc_size(_indices);

		tbb::parallel_for<tbb::blocked_range<size_t>>(
			tbb::blocked_range<size_t>(0, dst_size),
			[&](const tbb::blocked_range<size_t>& q) {
				Iterators src_iters = get_iterators(p);
				Iterators dst_iters = get_iterators(*this);

				for (size_t i = q.begin(); i < q.end(); ++i) {
					const size_t src_idx = count_to_elem_index(src_iters, i % src_size);
					const size_t dst_idx = count_to_elem_index(dst_iters, i);

					_data[dst_idx] -= p._data[src_idx];
				}
			},
			tbb::auto_partitioner()
		);
	}
	else {
		ErrorExcept(
			"[NN_Tensor<_T>::operator-=] None tensor can't this operator."
		);
	}

	*_cnt_rank = 0;
}

template <typename _T>
void NN_Tensor<_T>::operator*=(const NN_Tensor& p) {
	if (_data != NULL && p._data != NULL) {
		const NN_Shape origin_shape = calc_shape(_indices);
		const NN_Shape source_shape = calc_shape(p._indices);

		if (!is_valid_src_shape(origin_shape, source_shape)) {
			ErrorExcept(
				"[NN_Tensor<_T>::operator*=] Shape of Right operand are unsuitable. %s.",
				shape_to_str(source_shape)
			);
		}

		const size_t src_size = calc_size(p._indices);
		const size_t dst_size = calc_size(_indices);

		tbb::parallel_for<tbb::blocked_range<size_t>>(
			tbb::blocked_range<size_t>(0, dst_size),
			[&](const tbb::blocked_range<size_t>& q) {
				Iterators src_iters = get_iterators(p);
				Iterators dst_iters = get_iterators(*this);

			for (size_t i = q.begin(); i < q.end(); ++i) {
				const size_t src_idx = count_to_elem_index(src_iters, i % src_size);
				const size_t dst_idx = count_to_elem_index(dst_iters, i);

					_data[dst_idx] *= p._data[src_idx];
				}
			},
			tbb::auto_partitioner()
		);
	}
	else {
		ErrorExcept(
			"[NN_Tensor<_T>::operator*=] None tensor can't this operator."
		);
	}

	*_cnt_rank = 0;
}

template <typename _T>
void NN_Tensor<_T>::operator/=(const NN_Tensor& p) {
	if (_data != NULL && p._data != NULL) {
		const NN_Shape origin_shape = calc_shape(_indices);
		const NN_Shape source_shape = calc_shape(p._indices);

		if (!is_valid_src_shape(origin_shape, source_shape)) {
			ErrorExcept(
				"[NN_Tensor<_T>::operator/=] Shape of Right operand are unsuitable. %s.",
				shape_to_str(source_shape)
			);
		}

		const size_t src_size = calc_size(p._indices);
		const size_t dst_size = calc_size(_indices);

		tbb::parallel_for<tbb::blocked_range<size_t>>(
			tbb::blocked_range<size_t>(0, dst_size),
			[&](const tbb::blocked_range<size_t>& q) {
				Iterators src_iters = get_iterators(p);
				Iterators dst_iters = get_iterators(*this);

				for (size_t i = q.begin(); i < q.end(); ++i) {
					const size_t src_idx = count_to_elem_index(src_iters, i % src_size);
					const size_t dst_idx = count_to_elem_index(dst_iters, i);

					_data[dst_idx] /= p._data[src_idx];
				}
			},
			tbb::auto_partitioner()
		);
	}
	else {
		ErrorExcept(
			"[NN_Tensor<_T>::operator/=] None tensor can't this operator."
		);
	}

	*_cnt_rank = 0;
}

template <typename _T>
void NN_Tensor<_T>::operator+=(const _T& val) {
	if (_data != NULL) {
		const size_t dst_size = calc_size(_indices);

		tbb::parallel_for<tbb::blocked_range<size_t>>(
			tbb::blocked_range<size_t>(0, dst_size),
			[&](const tbb::blocked_range<size_t>& q) {
				Iterators iters = get_iterators(*this);

				for (size_t i = q.begin(); i < q.end(); ++i) {
					const size_t dst_idx = count_to_elem_index(iters, i);

					_data[dst_idx] += val;
				}
			},
			tbb::auto_partitioner()
		);
	}
	else {
		ErrorExcept(
			"[NN_Tensor<_T>::operator+=] None tensor can't this operator."
		);
	}
}

template <typename _T>
void NN_Tensor<_T>::operator-=(const _T& val) {
	if (_data != NULL) {
		const size_t dst_size = calc_size(_indices);

		tbb::parallel_for<tbb::blocked_range<size_t>>(
			tbb::blocked_range<size_t>(0, dst_size),
			[&](const tbb::blocked_range<size_t>& q) {
				Iterators iters = get_iterators(*this);

				for (size_t i = q.begin(); i < q.end(); ++i) {
					const size_t dst_idx = count_to_elem_index(iters, i);

					_data[dst_idx] -= val;
				}
			},
			tbb::auto_partitioner()
		);
	}
	else {
		ErrorExcept(
			"[NN_Tensor<_T>::operator-=] None tensor can't this operator."
		);
	}

	*_cnt_rank = 0;
}

template <typename _T>
void NN_Tensor<_T>::operator*=(const _T& val) {
	if (_data != NULL) {
		const size_t dst_size = calc_size(_indices);

		tbb::parallel_for<tbb::blocked_range<size_t>>(
			tbb::blocked_range<size_t>(0, dst_size),
			[&](const tbb::blocked_range<size_t>& q) {
				Iterators iters = get_iterators(*this);

				for (size_t i = q.begin(); i < q.end(); ++i) {
					const size_t dst_idx = count_to_elem_index(iters, i);

					_data[dst_idx] *= val;
				}
			},
			tbb::auto_partitioner()
		);
	}
	else {
		ErrorExcept(
			"[NN_Tensor<_T>::operator*=] None tensor can't this operator."
		);
	}

	*_cnt_rank = 0;
}

template <typename _T>
void NN_Tensor<_T>::operator/=(const _T& val) {
	if (_data != NULL) {
		const size_t dst_size = calc_size(_indices);

		tbb::parallel_for<tbb::blocked_range<size_t>>(
			tbb::blocked_range<size_t>(0, dst_size),
			[&](const tbb::blocked_range<size_t>& q) {
				Iterators iters = get_iterators(*this);

				for (size_t i = q.begin(); i < q.end(); ++i) {
					const size_t dst_idx = count_to_elem_index(iters, i);

					_data[dst_idx] /= val;
				}
			},
			tbb::auto_partitioner()
		);
	}
	else {
		ErrorExcept(
			"[NN_Tensor<_T>::operator/=] None tensor can't this operator."
		);
	}

	*_cnt_rank = 0;
}

template <typename _T>
NN_Tensor<_T> NN_Tensor<_T>::inverse(_T val) {
	NN_Tensor<_T> tmp(calc_shape(_indices));

	if (_data != NULL) {
		const size_t dst_size = calc_size(_indices);

		tbb::parallel_for<tbb::blocked_range<size_t>>(
			tbb::blocked_range<size_t>(0, dst_size),
			[&](const tbb::blocked_range<size_t>& q) {
				Iterators iters = get_iterators(*this);

				for (size_t i = q.begin(); i < q.end(); ++i) {
					const size_t dst_idx = count_to_elem_index(iters, i);

					tmp._data[dst_idx] = val / _data[dst_idx];
				}
			},
			tbb::auto_partitioner()
		);
	}
	else {
		ErrorExcept(
			"[NN_Tensor<_T>::inverse] None tensor can't inverse operator."
		);
	}

	*_cnt_rank = 0;

	return tmp;
}

template <typename _T>
NN_Tensor<_T> NN_Tensor<_T>::transpose(const std::initializer_list<int>& orders) {
	NN_Tensor<_T> tmp = *this;

	std::vector<uint> m_steps(_steps.size());
	std::vector<std::vector<uint>> m_indice(_indices.size());

	int i = 0;
	for (const int& n : orders) {
		m_steps[i] = _steps[n];
		m_indice[i++] = _indices[n];
	}

	tmp._steps = m_steps;
	tmp._indices = m_indice;

	*_cnt_rank = 0;

	return tmp;
}

template <typename _T>
NN_Tensor<_T> NN_Tensor<_T>::swap_pose() {
	NN_Tensor<_T> tmp = *this;

	std::vector<uint> m_steps(_steps.rbegin(), _steps.rend());
	std::vector<std::vector<uint>> m_indice(_indices.rbegin(), _indices.rend());

	tmp._steps = m_steps;
	tmp._indices = m_indice;

	*_cnt_rank = 0;

	return tmp;
}

template <typename _T>
_T& NN_Tensor<_T>::val() {
	if (calc_size(_indices) > 1) {
		ErrorExcept(
			"[NN_Tensor<_T>::val] This value is not scalar."
		);
	}

	Iterators iters = get_iterators(*this);
	const size_t index = count_to_elem_index(iters, 0);

	*_cnt_rank = 0;

	return _data[index];
}

template <typename _T>
const _T& NN_Tensor<_T>::val() const {
	if (calc_size(_indices) > 1) {
		ErrorExcept(
			"[NN_Tensor<_T>::val] This value is not scalar."
		);
	}

	Iterators iters = get_iterators(*this);
	const size_t index = count_to_elem_index(iters, 0);

	*_cnt_rank = 0;

	return _data[index];
}

template <typename _T>
void NN_Tensor<_T>::clear() {
	_data.reset();
	_steps.clear();
	_indices.clear();
	_cnt_rank = 0;
}

template <typename _T>
std::vector<std::vector<uint>>& NN_Tensor<_T>::get_indices() {
	return _indices;
}

template <typename _T>
const std::vector<std::vector<uint>>& NN_Tensor<_T>::get_indices() const {
	return _indices;
}

template <typename _T>
NN_Shape NN_Tensor<_T>::get_shape() {
	*_cnt_rank = 0;

	return calc_shape(_indices);
}

template <typename _T>
NN_Shape NN_Tensor<_T>::get_shape() const {
	*_cnt_rank = 0;

	return calc_shape(_indices);
}

template <typename _T>
_T* NN_Tensor<_T>::get_ptr() {
	*_cnt_rank = 0;

	return _data.get();
}

template <typename _T>
const _T* NN_Tensor<_T>::get_ptr() const {
	*_cnt_rank = 0;

	return _data.get();
}

template <typename _T>
std::ostream& NN_Tensor<_T>::put(std::ostream& os) {
	if (_indices.size() > 0) {
		int i = 0;

		put_tensor(os, *this, 0, i);
		os << "shape: " << shape_to_str(calc_shape(_indices)) << std::endl;
	}
	else os << "[]" << std::endl;

	*_cnt_rank = 0;

	return os;
}

template <typename _T>
std::ostream& NN_Tensor<_T>::put(std::ostream& os) const {
	if (_indices.size() > 0) {
		int i = 0;

		put_tensor(os, *this, 0, i);
		os << "shape: " << shape_to_str(calc_shape(_indices)) << std::endl;
	}
	else os << "[]" << std::endl;

	*_cnt_rank = 0;

	return os;
}

template <typename _T>
void NN_Tensor<_T>::resize(const NN_Shape& shape) {
	if ((int)_indices.size() != shape.ranks()) _indices.resize(shape.ranks());

	int i = 0;
	for (const int& n : shape) {
		if (n < 1) {
			ErrorExcept(
				"[NN_Tensor<_T>::NN_Tensor] Shape must be greater than 0. but %s",
				shape_to_str(shape)
			);
		}

		std::vector<uint> m_indice(n, 0);

		uint j = 0;
		for (uint& m : m_indice) m = j++;

		_indices[i++] = m_indice;
	}

	_steps = calc_step(shape);
	_data = std::shared_ptr<_T[]>(new _T[shape.total_size()]);

	*_cnt_rank = 0;
}


template <typename _T>
NN_Tensor<_T> NN_Tensor<_T>::reshape(const NN_Shape& shape) {
	NN_Shape origin = calc_shape(_indices);
	const size_t origin_size = origin.total_size();
	NN_Shape new_shape(shape);
	int minus_cnt = 0;
	size_t new_size = 1;

	for (const int& n : shape) {
		if (n == 0 || minus_cnt > 1) {
			ErrorExcept(
				"[NN_Tensor<_T>::reshape] Can't reshape %s to %s",
				shape_to_str(origin),
				shape_to_str(shape)
			);
		}
		else if (n < 0) ++minus_cnt;
		else new_size *= n;
	}

	if (origin_size % new_size != 0) {
		ErrorExcept(
			"[NN_Tensor<_T>::reshape] Can't reshape %s to %s",
			shape_to_str(origin),
			shape_to_str(shape)
		);
	}

	for (int& n : new_shape) {
		if (n < 0) n = (int)(origin_size / new_size);
	}

	NN_Tensor tensor;

	tensor._data = _data;
	tensor._steps = calc_step(shape);

	NN_Shape::r_iter m_iter = new_shape.rbegin();
	NN_Shape::r_iter m_end = new_shape.rend();
	NN_Shape::r_iter o_iter = origin.rbegin();
	NN_Shape::r_iter o_end = origin.rend();
	std::vector<std::vector<uint>>::reverse_iterator i_iter = _indices.rbegin();
	std::vector<std::vector<uint>>::reverse_iterator i_end = _indices.rend();

	/*
	NN_Tensor A(10, 10, 10);

	A = A(1, 9)(1, 9)(1, 9); // (8, 8, 8)
	A = A.reshape(-1, 64) // (8, 64)
	A = A.reshape(64, -1) // (64, 8)
	A = A.reshape(2, 2, 2, 64)

	NN_Tensor B = A.reshape({-1});
	NN_Tensor C = A.reshape({100, 10});
	NN_Tensor D = A.reshape({-1, 10});
*/

	for (; m_iter != m_end; ++m_iter){
		while (*o_iter > 0) {

			--(*o_iter);
			--(*m_iter);
		}
	}

	return NN_Tensor(_data, shape);
}

template <typename _T>
template <typename _cT>
NN_Tensor<_cT> NN_Tensor<_T>::cast() const {
	NN_Tensor<_cT> dst(calc_shape(_indices));
	NN_Tensor<_T> src(calc_shape(_indices));

	src = *this;

	tbb::parallel_for(
		tbb::blocked_range<size_t>(0, calc_size(_indices)),
		[&](const tbb::blocked_range<size_t>& q) {

		const _T* p_src = src.get_ptr();
		_cT* p_dst = dst.get_ptr();

		for (size_t i = q.begin(); i < q.end(); ++i) {
			p_dst[i] = (_cT)(p_src[i]);
		}
	}
	);

	*_cnt_rank = 0;

	return dst;
}

template <typename _T>
NN_Tensor<_T> NN_Tensor<_T>::expand_dims(const NN_Tensor& tensor, int axis) {
	NN_Shape shape = calc_shape(tensor._indices);
	const int ranks = shape.ranks();
	const int curr_axis = axis < 0 ? ranks - axis : axis;

	if (ranks <= curr_axis || 0 > curr_axis) {
		ErrorExcept(
			"[NN_Tensor<_T>::expand_dims] %d axis is out of range.",
			axis
		);
	}

	NN_Tensor<_T> m_tensor = tensor;

	m_tensor._steps.insert(m_tensor._steps.begin() + curr_axis, m_tensor._steps[curr_axis] * shape[curr_axis]);
	m_tensor._indices.insert(m_tensor._indices.begin() + curr_axis, { 0 });

	*tensor._cnt_rank = 0;

	return m_tensor;
}

template <typename _T>
NN_Tensor<_T> NN_Tensor<_T>::expand_dims(const NN_Tensor& tensor, std::initializer_list<int>& axis) {
	NN_Shape shape = calc_shape(tensor._indices);
	const int ranks = shape.ranks();

	NN_Tensor<_T> m_tensor = tensor;

	for (const int& n : axis) {
		const int curr_axis = n < 0 ? ranks - n : n;

		if (ranks <= curr_axis || 0 > curr_axis) {
			ErrorExcept(
				"[NN_Tensor<_T>::expand_dims] %s axis is out of range.",
				shape_to_str(axis)
			);
		}

		m_tensor._steps.insert(m_tensor._steps.begin() + curr_axis, m_tensor._steps[curr_axis] * shape[curr_axis]);
		m_tensor._indices.insert(m_tensor._indices.begin() + curr_axis, { 0 });
	}

	*tensor._cnt_rank = 0;

	return m_tensor;
}

template <typename _T>
NN_Tensor<_T> NN_Tensor<_T>::squeeze(const NN_Tensor& tensor, int axis) {
	NN_Shape shape = calc_shape(tensor._indices);
	const int ranks = shape.ranks();
	const int curr_axis = axis < 0 ? ranks - axis : axis;

	if (ranks <= curr_axis || 0 > curr_axis) {
		ErrorExcept(
			"[NN_Tensor<_T>::squeeze] %d axis is out of range.",
			axis
		);
	}
	else if (shape[curr_axis] > 1) {
		ErrorExcept(
			"[NN_Tensor<_T>::squeeze] Can't squeeze %d axis.",
			axis
		);
	}

	NN_Tensor<_T> m_tensor = tensor;

	m_tensor._steps.erase(m_tensor._steps.begin() + curr_axis);
	m_tensor._indices.erase(m_tensor._indices.begin() + curr_axis);

	*tensor._cnt_rank = 0;

	return m_tensor;
}

template <typename _T>
NN_Tensor<_T> NN_Tensor<_T>::squeeze(const NN_Tensor& tensor, std::initializer_list<int>& axis) {
	NN_Shape shape = calc_shape(tensor._indices);
	const int ranks = shape.ranks();

	NN_Tensor<_T> m_tensor = tensor;

	for (const int& n : axis) {
		const int curr_axis = n < 0 ? ranks - n : n;

		if (ranks <= curr_axis || 0 > curr_axis) {
			ErrorExcept(
				"[NN_Tensor<_T>::squeeze] %s axis is out of range.",
				shape_to_str(axis)
			);
		}
		else if (shape[curr_axis] > 1) {
			ErrorExcept(
				"[NN_Tensor<_T>::squeeze] Can't squeeze %d axis.",
				shape_to_str(axis)
			);
		}

		m_tensor._steps.erase(m_tensor._steps.begin() + curr_axis);
		m_tensor._indices.erase(m_tensor._indices.begin() + curr_axis);
	}

	*tensor._cnt_rank = 0;

	return m_tensor;
}

template <typename _T>
NN_Tensor<_T> NN_Tensor<_T>::zeros(const NN_Shape& shape) {
	NN_Tensor<_T> tensor(shape);

	tensor = _T(0);

	return tensor;
}

template <typename _T>
std::shared_ptr<_T[]> NN_Tensor<_T>::get_shared_ptr() {
	return _data;
}

template <typename _T>
const std::shared_ptr<_T[]> NN_Tensor<_T>::get_shared_ptr() const {
	return _data;
}