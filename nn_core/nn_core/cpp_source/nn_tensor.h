#pragma once
#include <tbb/tbb.h>
#include <memory>
#include "Exception.h"
#include "nn_list.h"
#include "nn_shape.h"
#include "mat_dtype.h"


template <typename _T>
class GpuTensor;

/**********************************************/
/*                                            */
/*                    Tensor                  */
/*                                            */
/**********************************************/

template <typename _T>
class Tensor {
	std::shared_ptr<_T[]> _data;
	std::vector<size_t> _steps;
	std::vector<std::vector<size_t>> _indice;

	int* _cnt_rank;

	static std::vector<std::vector<size_t>> calc_indices(const NN_Shape& shape);
	static std::vector<size_t> calc_step(const NN_Shape& shape);
	static size_t calc_size(const std::vector<std::vector<size_t>>& indice);
	static size_t count_to_elem_index(const std::vector<size_t>& steps, const std::vector<std::vector<size_t>>& indice, size_t count);
	static NN_Shape calc_shape(const std::vector<std::vector<size_t>>& indice);
	static bool is_valid_src_shape(const NN_Shape& dst_shape, const NN_Shape& src_shape);
	static void count_indice(std::vector<size_t>& indice, int begin, int end, int step);
	static void put_tensor(std::ostream& os, const Tensor& tensor, size_t offset, int& rank);

	Tensor(const Tensor& p, int cnt_rank);

public:
	Tensor();
	Tensor(const NN_Shape& shape);
	Tensor(const size_t* p_dims, int n_dims);
	Tensor(const std::initializer_list<int>& shape);
	Tensor(const Tensor& p);
	Tensor(Tensor&& p);
	Tensor(const GpuTensor<_T>& p);
	Tensor(const cv::Mat& mat);
	~Tensor();

	Tensor& operator=(const Tensor& p);
	Tensor& operator=(const GpuTensor<_T>& p);
	Tensor& operator=(_T scalar);
	Tensor operator()(int begin, int end, int step = 1);
	Tensor operator()(int begin, int end, int step = 1) const;
	Tensor operator()(int index);
	Tensor operator()(int index) const;
	Tensor operator()(const std::vector<int>& indice);
	Tensor operator()(const std::vector<int>& indice) const;
	Tensor operator[](int index);
	Tensor operator[](int index) const;

	Tensor operator+(const Tensor& p);
	Tensor operator-(const Tensor& p);
	Tensor operator*(const Tensor& p);
	Tensor operator/(const Tensor& p);

	Tensor operator+(const _T& val);
	Tensor operator-(const _T& val);
	Tensor operator*(const _T& val);
	Tensor operator/(const _T& val);

	void operator+=(const Tensor& p);
	void operator-=(const Tensor& p);
	void operator*=(const Tensor& p);
	void operator/=(const Tensor& p);

	void operator+=(const _T& val);
	void operator-=(const _T& val);
	void operator*=(const _T& val);
	void operator/=(const _T& val);

	Tensor inverse(_T val);

	Tensor transpose(const std::initializer_list<int>& orders);
	Tensor swap_pose();
	Tensor reshape(const NN_Shape& new_shape);
	cv::Mat&& get_mat();

	_T& val();
	const _T& val() const;

	void clear();
	NN_Shape get_shape();
	NN_Shape get_shape() const;
	_T* get_ptr();
	_T* get_ptr() const;

	std::vector<size_t>& get_steps();
	const std::vector<size_t>& get_steps() const;

	std::vector<std::vector<size_t>>& get_indices();
	const std::vector<std::vector<size_t>>& get_indices() const;

	std::ostream& put(std::ostream& os);
	std::ostream& put(std::ostream& os) const;

	void resize(const NN_Shape& shape);

	template <typename _cT>
	void cast(const Tensor<_cT>& src);

	template <typename _cT>
	void cast(const Tensor<_cT>& src) const;

	static Tensor expand_dims(const Tensor& tensor, int axis = 0);
	static Tensor expand_dims(const Tensor& tensor, std::initializer_list<int>& axis);

	static Tensor squeeze(const Tensor& tensor, int axis = 0);
	static Tensor squeeze(const Tensor& tensor, std::initializer_list<int>& axis);

	static Tensor zeros(const NN_Shape& shape);
};


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
	GpuTensor(const Tensor<_T>& p);
	GpuTensor(const GpuTensor& p, const NN_Shape& shape);
	~GpuTensor();

	GpuTensor& operator=(const GpuTensor& p);
	GpuTensor& operator=(const Tensor<_T>& p);

	NN_Shape get_shape();
	NN_Shape get_shape() const;

	_T* get_ptr();
	_T* get_ptr() const;

	void reshape(const NN_Shape& shape);
	void resize(const NN_Shape& shape);
	void resize(const std::shared_ptr<_T>& g_data, const NN_Shape& shape);

	static GpuTensor<_T> zeros(const NN_Shape& shape);
};


/**********************************************/
/*                                            */
/*                    Tensor                  */
/*                                            */
/**********************************************/

template <typename _T>
using tensor_t = std::shared_ptr<_T[]>;

template <typename _T>
std::ostream& operator<<(std::ostream& os, const Tensor<_T>& tensor) {
	tensor.put(os);

	return os;
}

template <typename _T>
std::vector<std::vector<size_t>> Tensor<_T>::calc_indices(const NN_Shape& shape) {
	std::vector<std::vector<size_t>> indices;

	for (const int& n : shape) {
		if (n < 1) {
			ErrorExcept(
				"[Tensor<_T>::calc_indices] Shape must be greater than 0. but %s",
				shape_to_str(shape)
			);
		}

		size_t i = 0;
		std::vector<size_t> indice(n, 0);

		for (size_t& j : indice) j = i++;

		indices.push_back(indice);
	}

	return indices;
}

template <typename _T>
std::vector<size_t> Tensor<_T>::calc_step(const NN_Shape& shape) {
	std::vector<size_t> steps(shape.ranks(), 1);

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
	size_t index = 0;

	std::vector<size_t>::const_reverse_iterator step_iter = steps.crbegin();
	std::vector<std::vector<size_t>>::const_reverse_iterator indice_iter = indice.crbegin();

	for (; step_iter != steps.crend(); ++step_iter, ++indice_iter) {
		const size_t& step = *step_iter;
		const std::vector<size_t>& indices = *indice_iter;
		const size_t dims = indices.size();

		index += step * indices[count % dims];
		count /= dims;
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
Tensor<_T>::Tensor(const Tensor& p, int cnt_rank) :
	_data(p._data),
	_steps(p._steps),
	_indice(p._indice),
	_cnt_rank(new int(cnt_rank))
{
}

template <typename _T>
Tensor<_T>::Tensor() :
	_cnt_rank(new int(0))
{
}

template <typename _T>
Tensor<_T>::Tensor(const NN_Shape& shape) :
	_indice(shape.ranks()),
	_cnt_rank(new int(0))
{
	_indice = calc_indices(shape);
	_steps = calc_step(shape);
	_data = std::shared_ptr<_T[]>(new _T[shape.total_size()]);
}

template <typename _T>
Tensor<_T>::Tensor(const size_t* p_dims, int n_dims) :
	_indice(n_dims),
	_cnt_rank(new int(0))
{
	NN_Shape shape(n_dims);

	for (int i = 0; i < n_dims; ++i) {
		std::vector<size_t> m_indice(p_dims[i], 0);

		size_t j = 0;
		for (size_t& m : m_indice) m = j++;

		_indice[i] = m_indice;
		shape[i] = (int)p_dims[i];
	}

	_steps = calc_step(shape);
	_data = std::shared_ptr<_T[]>(new _T[shape.total_size()]);
}


template <typename _T>
Tensor<_T>::Tensor(const std::initializer_list<int>& shape) :
	_indice(shape.size()),
	_cnt_rank(new int(0))
{
	const NN_Shape m_shape(shape);

	_indice = calc_indices(m_shape);
	_steps = calc_step(m_shape);
	_data = std::shared_ptr<_T[]>(new _T[m_shape.total_size()]);
}

template <typename _T>
Tensor<_T>::Tensor(const Tensor& p) :
	_data(p._data),
	_steps(p._steps),
	_indice(p._indice),
	_cnt_rank(new int(0))
{

}

template <typename _T>
Tensor<_T>::Tensor(Tensor&& p) :
	_data(p._data),
	_steps(p._steps),
	_indice(p._indice),
	_cnt_rank(p._cnt_rank)
{
	p._cnt_rank = NULL;
}

template <typename _T>
Tensor<_T>::Tensor(const GpuTensor<_T>& p) :
	_cnt_rank(new int(0))
{
	const NN_Shape shape = p.get_shape();

	_steps = calc_step(shape);
	_indice = calc_indices(shape);
	_data = std::shared_ptr<_T[]>(new _T[shape.total_size()]);

	const _T* src_ptr = p.get_ptr();
	_T* dst_ptr = _data.get();

	check_cuda(cudaMemcpy(dst_ptr, src_ptr, sizeof(_T) * shape.total_size(), cudaMemcpyDeviceToHost));
}

template <typename _T>
Tensor<_T>::Tensor(const cv::Mat& mat) {
	const NN_Shape shape({ mat.rows, mat.cols, mat.channels() });
	const size_t t_size = shape.total_size();

	_steps = calc_step(shape);
	_indice = calc_indices(shape);
	_data = std::shared_ptr<_T[]>(new _T[t_size]);

	const _T* src = mat.data;
	_T* dst = _data.get();

	memcpy_s(dst, sizeof(_T) * t_size, src, sizeof(_T) * t_size);
}

template <typename _T>
Tensor<_T>::~Tensor() {
	delete _cnt_rank;
}

template <typename _T>
Tensor<_T>& Tensor<_T>::operator=(const Tensor& p) {
	if (this == &p) return *this;

	if (_data == NULL) {
		_data = p._data;
		_steps = p._steps;
		_indice = p._indice;
	}
	else {
		const NN_Shape dst_shape = calc_shape(_indice);
		const NN_Shape src_shape = calc_shape(p._indice);

		if (src_shape != dst_shape) {
			ErrorExcept(
				"[Tensor<_T>::operator=] different of src and dst tensors. %s != %s",
				shape_to_str(src_shape),
				shape_to_str(dst_shape)
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

	*_cnt_rank = 0;

	return *this;
}

template <typename _T>
Tensor<_T>& Tensor<_T>::operator=(const GpuTensor<_T>& p) {
	const NN_Shape src_shape = p.get_shape();

	if (_data == NULL) {
		_steps = calc_step(src_shape);
		_indice = calc_indices(src_shape);
		_data = std::shared_ptr<_T[]>(new _T[src_shape.total_size()]);
	}
	else {
		const NN_Shape dst_shape = calc_shape(_indice);

		if (dst_shape != src_shape) {
			ErrorExcept(
				"[Tensor<_T>::operator=] Different src and dst tensors. %s != %s",
				shape_to_str(src_shape),
				shape_to_str(dst_shape)
			);
		}
	}

	check_cuda(cudaMemcpy(_data.get(), p.get_ptr(), sizeof(_T) * src_shape.total_size(), cudaMemcpyDeviceToHost));
	*_cnt_rank = 0;

	return *this;
}

template <typename _T>
Tensor<_T>& Tensor<_T>::operator=(_T scalar) {
	if (_data == NULL) {
		_data = std::shared_ptr<_T[]>(new _T[1]);
		_steps.clear();
		_steps.push_back(1);
		_indice.clear();
		_indice.push_back({ 0 });

		_data[0] = scalar;
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

	*_cnt_rank = 0;

	return *this;
}

template <typename _T>
Tensor<_T> Tensor<_T>::operator()(int begin, int end, int step) {
	if (*_cnt_rank >= _indice.size()) {
		ErrorExcept(
			"[Tensor<_T>::operator()] %d rank is empty.",
			*_cnt_rank
		);
	}

	const int n = (int)_indice[*_cnt_rank].size();

	begin = begin < 0 ? n + begin : begin;
	end = end < 0 ? n + end : end;

	if (begin < 0 || begin >= n || end < 0 || end > n) {
		ErrorExcept(
			"[Tensor<_T>::operator()] begin and end is out of range. begin: %d, end: %d, step: %d",
			begin, end, step
		);
	}

	Tensor<_T> tensor(*this, *_cnt_rank + 1);

	count_indice(tensor._indice[*_cnt_rank], begin, end, step);

	return tensor;
}

template <typename _T>
Tensor<_T> Tensor<_T>::operator()(int begin, int end, int step) const {
	if (*_cnt_rank >= _indice.size()) {
		ErrorExcept(
			"[Tensor<_T>::operator()] %d rank is empty.",
			*_cnt_rank
		);
	}

	const int n = (int)_indice[*_cnt_rank].size();

	begin = begin < 0 ? n + begin : begin;
	end = end < 0 ? n + end : end;

	if (begin < 0 || begin >= n || end < 0 || end > n) {
		ErrorExcept(
			"[Tensor<_T>::operator()] begin and end is out of range. begin: %d, end: %d, step: %d",
			begin, end, step
		);
	}

	Tensor<_T> tensor(*this, *_cnt_rank + 1);

	count_indice(tensor._indice[*_cnt_rank], begin, end, step);
	*_cnt_rank = 0;

	return tensor;
}

template <typename _T>
Tensor<_T> Tensor<_T>::operator()(int index) {
	if (*_cnt_rank >= _indice.size()) {
		ErrorExcept(
			"[Tensor<_T>::operator()] %d rank is empty.",
			_cnt_rank
		);
	}

	const int n = (int)_indice[*_cnt_rank].size();

	index = index < 0 ? n + index : index;

	if (index < 0 || index >= n) {
		ErrorExcept(
			"[Tensor<_T>::operator()] index is out of range."
		);
	}

	Tensor<_T> tensor(*this, *_cnt_rank + 1);

	count_indice(tensor._indice[*_cnt_rank], index, index + 1, 1);
	*_cnt_rank = 0;

	return tensor;
}

template <typename _T>
Tensor<_T> Tensor<_T>::operator()(int index) const {
	if (*_cnt_rank >= _indice.size()) {
		ErrorExcept(
			"[Tensor<_T>::operator()] %d rank is empty.",
			_cnt_rank
		);
	}

	const int n = (int)_indice[*_cnt_rank].size();

	index = index < 0 ? n + index : index;

	if (index < 0 || index >= n) {
		ErrorExcept(
			"[Tensor<_T>::operator()] index is out of range."
		);
	}

	Tensor<_T> tensor(*this, *_cnt_rank + 1);

	count_indice(tensor._indice[*_cnt_rank], index, index + 1, 1);
	*_cnt_rank = 0;

	return tensor;
}

template <typename _T>
Tensor<_T> Tensor<_T>::operator()(const std::vector<int>& indice) {
	if (*_cnt_rank >= _indice.size()) {
		ErrorExcept(
			"[Tensor<_T>::operator()] %d rank is empty.",
			_cnt_rank
		);
	}

	const std::vector<size_t>& curr_indice = _indice[*_cnt_rank];
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

	Tensor<_T> tensor(*this, *_cnt_rank + 1);

	tensor._indice[*_cnt_rank] = m_indice;
	*_cnt_rank = 0;

	return tensor;
}

template <typename _T>
Tensor<_T> Tensor<_T>::operator()(const std::vector<int>& indice) const {
	if (*_cnt_rank >= _indice.size()) {
		ErrorExcept(
			"[Tensor<_T>::operator()] %d rank is empty.",
			_cnt_rank
		);
	}

	const std::vector<size_t>& curr_indice = _indice[*_cnt_rank];
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

	Tensor<_T> tensor(*this, *_cnt_rank + 1);

	tensor._indice[*_cnt_rank] = m_indice;
	*_cnt_rank = 0;

	return tensor;
}

template <typename _T>
Tensor<_T> Tensor<_T>::operator[](int index) {
	if (*_cnt_rank >= _indice.size()) {
		ErrorExcept(
			"[Tensor<_T>::operator()] %d rank is empty.",
			*_cnt_rank
		);
	}

	const int n = (int)_indice[*_cnt_rank].size();

	index = index < 0 ? n + index : index;

	if (index < 0 || index >= n) {
		ErrorExcept(
			"[Tensor<_T>::operator()] begin and end is out of range."
		);
	}

	Tensor<_T> tensor(*this, *_cnt_rank + 1);

	count_indice(tensor._indice[*_cnt_rank], index, index + 1, 1);
	*_cnt_rank = 0;

	return tensor;
}

template <typename _T>
Tensor<_T> Tensor<_T>::operator[](int index) const {
	if (*_cnt_rank >= _indice.size()) {
		ErrorExcept(
			"[Tensor<_T>::operator()] %d rank is empty.",
			_cnt_rank
		);
	}

	const int n = (int)_indice[*_cnt_rank].size();

	index = index < 0 ? n + index : index;

	if (index < 0 || index >= n) {
		ErrorExcept(
			"[Tensor<_T>::operator()] begin and end is out of range."
		);
	}

	Tensor<_T> tensor(*this, *_cnt_rank + 1);

	count_indice(tensor._indice[*_cnt_rank], index, index + 1, 1);
	*_cnt_rank = 0;

	return tensor;
}

template <typename _T>
Tensor<_T> Tensor<_T>::operator+(const Tensor& p) {
	Tensor<_T> value(calc_shape(_indice));

	if (_data != NULL && p._data != NULL) {
		const NN_Shape origin_shape = calc_shape(_indice);
		const NN_Shape source_shape = calc_shape(p._indice);

		if (!is_valid_src_shape(origin_shape, source_shape)) {
			ErrorExcept(
				"[Tensor<_T>::operator+] Shape of Right operand are unsuitable. %s.",
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

				value._data[dst_idx] = _data[dst_idx] + p._data[src_idx];
			}
		}
		);
	}
	else {
		ErrorExcept(
			"[Tensor<_T>::operator+] None tensor can't this operator."
		);
	}

	*_cnt_rank = 0;

	return value;
}

template <typename _T>
Tensor<_T> Tensor<_T>::operator-(const Tensor& p) {
	Tensor<_T> value(calc_shape(_indice));

	if (_data != NULL && p._data != NULL) {
		const NN_Shape origin_shape = calc_shape(_indice);
		const NN_Shape source_shape = calc_shape(p._indice);

		if (!is_valid_src_shape(origin_shape, source_shape)) {
			ErrorExcept(
				"[Tensor<_T>::operator-] Shape of Right operand are unsuitable. %s.",
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

				value._data[dst_idx] = _data[dst_idx] - p._data[src_idx];
			}
		}
		);
	}
	else {
		ErrorExcept(
			"[Tensor<_T>::operator-] None tensor can't this operator."
		);
	}

	*_cnt_rank = 0;

	return value;
}

template <typename _T>
Tensor<_T> Tensor<_T>::operator*(const Tensor& p) {
	Tensor<_T> value(calc_shape(_indice));

	if (_data != NULL && p._data != NULL) {
		const NN_Shape origin_shape = calc_shape(_indice);
		const NN_Shape source_shape = calc_shape(p._indice);

		if (!is_valid_src_shape(origin_shape, source_shape)) {
			ErrorExcept(
				"[Tensor<_T>::operator*] Shape of Right operand are unsuitable. %s.",
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

				value._data[dst_idx] = _data[dst_idx] * p._data[src_idx];
			}
		}
		);
	}
	else {
		ErrorExcept(
			"[Tensor<_T>::operator*] None tensor can't this operator."
		);
	}

	*_cnt_rank = 0;

	return value;
}

template <typename _T>
Tensor<_T> Tensor<_T>::operator/(const Tensor& p) {
	Tensor<_T> value(calc_shape(_indice));

	if (_data != NULL && p._data != NULL) {
		const NN_Shape origin_shape = calc_shape(_indice);
		const NN_Shape source_shape = calc_shape(p._indice);

		if (!is_valid_src_shape(origin_shape, source_shape)) {
			ErrorExcept(
				"[Tensor<_T>::operator/] Shape of Right operand are unsuitable. %s.",
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

				value._data[dst_idx] = _data[dst_idx] / p._data[src_idx];
			}
		}
		);
	}
	else {
		ErrorExcept(
			"[Tensor<_T>::operator/] None tensor can't this operator."
		);
	}

	*_cnt_rank = 0;

	return value;
}

template <typename _T>
Tensor<_T> Tensor<_T>::operator+(const _T& val) {
	Tensor<_T> value(calc_shape(_indice));

	if (_data != NULL) {
		const size_t dst_size = calc_size(_indice);

		tbb::parallel_for<tbb::blocked_range<size_t>>(
			tbb::blocked_range<size_t>(0, dst_size),
			[&](const tbb::blocked_range<size_t>& q) {

			for (size_t i = q.begin(); i < q.end(); ++i) {
				const size_t dst_idx = count_to_elem_index(_steps, _indice, i);

				value._data[dst_idx] = _data[dst_idx] + val;
			}
		}
		);
	}
	else {
		ErrorExcept(
			"[Tensor<_T>::operator+] None tensor can't this operator."
		);
	}

	*_cnt_rank = 0;

	return value;
}

template <typename _T>
Tensor<_T> Tensor<_T>::operator-(const _T& val) {
	Tensor<_T> value(calc_shape(_indice));

	if (_data != NULL) {
		const size_t dst_size = calc_size(_indice);

		tbb::parallel_for<tbb::blocked_range<size_t>>(
			tbb::blocked_range<size_t>(0, dst_size),
			[&](const tbb::blocked_range<size_t>& q) {

			for (size_t i = q.begin(); i < q.end(); ++i) {
				const size_t dst_idx = count_to_elem_index(_steps, _indice, i);

				value._data[dst_idx] = _data[dst_idx] - val;
			}
		}
		);
	}
	else {
		ErrorExcept(
			"[Tensor<_T>::operator-] None tensor can't this operator."
		);
	}

	*_cnt_rank = 0;

	return value;
}

template <typename _T>
Tensor<_T> Tensor<_T>::operator*(const _T& val) {
	Tensor<_T> value(calc_shape(_indice));

	if (_data != NULL) {
		const size_t dst_size = calc_size(_indice);

		tbb::parallel_for<tbb::blocked_range<size_t>>(
			tbb::blocked_range<size_t>(0, dst_size),
			[&](const tbb::blocked_range<size_t>& q) {

			for (size_t i = q.begin(); i < q.end(); ++i) {
				const size_t dst_idx = count_to_elem_index(_steps, _indice, i);

				value._data[dst_idx] = _data[dst_idx] * val;
			}
		}
		);
	}
	else {
		ErrorExcept(
			"[Tensor<_T>::operator*] None tensor can't this operator."
		);
	}

	*_cnt_rank = 0;

	return value;
}

template <typename _T>
Tensor<_T> Tensor<_T>::operator/(const _T& val) {
	Tensor<_T> value(calc_shape(_indice));

	if (_data != NULL) {
		const size_t dst_size = calc_size(_indice);

		tbb::parallel_for<tbb::blocked_range<size_t>>(
			tbb::blocked_range<size_t>(0, dst_size),
			[&](const tbb::blocked_range<size_t>& q) {

			for (size_t i = q.begin(); i < q.end(); ++i) {
				const size_t dst_idx = count_to_elem_index(_steps, _indice, i);

				value._data[dst_idx] = _data[dst_idx] / val;
			}
		}
		);
	}
	else {
		ErrorExcept(
			"[Tensor<_T>::operator/] None tensor can't this operator."
		);
	}

	*_cnt_rank = 0;

	return value;
}

template <typename _T>
void Tensor<_T>::operator+=(const Tensor& p) {
	if (_data != NULL && p._data != NULL) {
		const NN_Shape origin_shape = calc_shape(_indice);
		const NN_Shape source_shape = calc_shape(p._indice);

		if (!is_valid_src_shape(origin_shape, source_shape)) {
			ErrorExcept(
				"[Tensor<_T>::operator+=] Shape of Right operand are unsuitable. %s.",
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

				_data[dst_idx] += p._data[src_idx];
			}
		}
		);
	}
	else {
		ErrorExcept(
			"[Tensor<_T>::operator+=] None tensor can't this operator."
		);
	}

	*_cnt_rank = 0;
}

template <typename _T>
void Tensor<_T>::operator-=(const Tensor& p) {
	if (_data != NULL && p._data != NULL) {
		const NN_Shape origin_shape = calc_shape(_indice);
		const NN_Shape source_shape = calc_shape(p._indice);

		if (!is_valid_src_shape(origin_shape, source_shape)) {
			ErrorExcept(
				"[Tensor<_T>::operator-=] Shape of Right operand are unsuitable. %s.",
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

				_data[dst_idx] -= p._data[src_idx];
			}
		}
		);
	}
	else {
		ErrorExcept(
			"[Tensor<_T>::operator-=] None tensor can't this operator."
		);
	}

	*_cnt_rank = 0;
}

template <typename _T>
void Tensor<_T>::operator*=(const Tensor& p) {
	if (_data != NULL && p._data != NULL) {
		const NN_Shape origin_shape = calc_shape(_indice);
		const NN_Shape source_shape = calc_shape(p._indice);

		if (!is_valid_src_shape(origin_shape, source_shape)) {
			ErrorExcept(
				"[Tensor<_T>::operator*=] Shape of Right operand are unsuitable. %s.",
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

				_data[dst_idx] *= p._data[src_idx];
			}
		}
		);
	}
	else {
		ErrorExcept(
			"[Tensor<_T>::operator*=] None tensor can't this operator."
		);
	}

	*_cnt_rank = 0;
}

template <typename _T>
void Tensor<_T>::operator/=(const Tensor& p) {
	if (_data != NULL && p._data != NULL) {
		const NN_Shape origin_shape = calc_shape(_indice);
		const NN_Shape source_shape = calc_shape(p._indice);

		if (!is_valid_src_shape(origin_shape, source_shape)) {
			ErrorExcept(
				"[Tensor<_T>::operator/=] Shape of Right operand are unsuitable. %s.",
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

				_data[dst_idx] /= p._data[src_idx];
			}
		}
		);
	}
	else {
		ErrorExcept(
			"[Tensor<_T>::operator/=] None tensor can't this operator."
		);
	}

	*_cnt_rank = 0;
}

template <typename _T>
void Tensor<_T>::operator+=(const _T& val) {
	if (_data != NULL) {
		const size_t dst_size = calc_size(_indice);

		tbb::parallel_for<tbb::blocked_range<size_t>>(
			tbb::blocked_range<size_t>(0, dst_size),
			[&](const tbb::blocked_range<size_t>& q) {

			for (size_t i = q.begin(); i < q.end(); ++i) {
				const size_t dst_idx = count_to_elem_index(_steps, _indice, i);

				_data[dst_idx] += val;
			}
		}
		);
	}
	else {
		ErrorExcept(
			"[Tensor<_T>::operator+=] None tensor can't this operator."
		);
	}
}

template <typename _T>
void Tensor<_T>::operator-=(const _T& val) {
	if (_data != NULL) {
		const size_t dst_size = calc_size(_indice);

		tbb::parallel_for<tbb::blocked_range<size_t>>(
			tbb::blocked_range<size_t>(0, dst_size),
			[&](const tbb::blocked_range<size_t>& q) {

			for (size_t i = q.begin(); i < q.end(); ++i) {
				const size_t dst_idx = count_to_elem_index(_steps, _indice, i);

				_data[dst_idx] -= val;
			}
		}
		);
	}
	else {
		ErrorExcept(
			"[Tensor<_T>::operator-=] None tensor can't this operator."
		);
	}

	*_cnt_rank = 0;
}

template <typename _T>
void Tensor<_T>::operator*=(const _T& val) {
	if (_data != NULL) {
		const size_t dst_size = calc_size(_indice);

		tbb::parallel_for<tbb::blocked_range<size_t>>(
			tbb::blocked_range<size_t>(0, dst_size),
			[&](const tbb::blocked_range<size_t>& q) {

			for (size_t i = q.begin(); i < q.end(); ++i) {
				const size_t dst_idx = count_to_elem_index(_steps, _indice, i);

				_data[dst_idx] *= val;
			}
		}
		);
	}
	else {
		ErrorExcept(
			"[Tensor<_T>::operator*=] None tensor can't this operator."
		);
	}

	*_cnt_rank = 0;
}

template <typename _T>
void Tensor<_T>::operator/=(const _T& val) {
	if (_data != NULL) {
		const size_t dst_size = calc_size(_indice);

		tbb::parallel_for<tbb::blocked_range<size_t>>(
			tbb::blocked_range<size_t>(0, dst_size),
			[&](const tbb::blocked_range<size_t>& q) {

			for (size_t i = q.begin(); i < q.end(); ++i) {
				const size_t dst_idx = count_to_elem_index(_steps, _indice, i);

				_data[dst_idx] /= val;
			}
		}
		);
	}
	else {
		ErrorExcept(
			"[Tensor<_T>::operator/=] None tensor can't this operator."
		);
	}

	*_cnt_rank = 0;
}

template <typename _T>
Tensor<_T> Tensor<_T>::inverse(_T val) {
	Tensor<_T> tmp(calc_shape(_indice));

	if (_data != NULL) {
		const size_t dst_size = calc_size(_indice);

		tbb::parallel_for<tbb::blocked_range<size_t>>(
			tbb::blocked_range<size_t>(0, dst_size),
			[&](const tbb::blocked_range<size_t>& q) {

			for (size_t i = q.begin(); i < q.end(); ++i) {
				const size_t dst_idx = count_to_elem_index(_steps, _indice, i);

				tmp._data[dst_idx] = val / _data[dst_idx];
			}
		}
		);
	}
	else {
		ErrorExcept(
			"[Tensor<_T>::inverse] None tensor can't inverse operator."
		);
	}

	*_cnt_rank = 0;

	return tmp;
}

template <typename _T>
Tensor<_T> Tensor<_T>::transpose(const std::initializer_list<int>& orders) {
	Tensor<_T> tmp = *this;

	std::vector<size_t> m_steps(_steps.size());
	std::vector<std::vector<size_t>> m_indice(_indice.size());

	int i = 0;
	for (const int& n : orders) {
		m_steps[i] = _steps[n];
		m_indice[i++] = _indice[n];
	}

	tmp._steps = m_steps;
	tmp._indice = m_indice;

	*_cnt_rank = 0;

	return tmp;
}

template <typename _T>
Tensor<_T> Tensor<_T>::swap_pose() {
	Tensor<_T> tmp = *this;

	std::vector<size_t> m_steps(_steps.rbegin(), _steps.rend());
	std::vector<std::vector<size_t>> m_indice(_indice.rbegin(), _indice.rend());

	tmp._steps = m_steps;
	tmp._indice = m_indice;

	*_cnt_rank = 0;

	return tmp;
}

template <typename _T>
Tensor<_T> Tensor<_T>::reshape(const NN_Shape& new_shape) {
	Tensor<_T> tmp = *this;

	std::vector<std::vector<size_t>> new_indices = calc_indices(new_shape);
	size_t new_size = calc_size(new_indices);
	size_t old_size = calc_size(_indice);

	if (new_size != old_size) {
		ErrorExcept(
			"[Tensor<_T>::reshape] different size. %ld != %ld",
			new_size, old_size
		);
	}

	tmp._steps = calc_step(new_shape);
	tmp._indice = new_indices;

	return tmp;
}

template <typename _T>
cv::Mat&& Tensor<_T>::get_mat() {
	const NN_Shape shape = calc_shape(_indice);
	const int dtype = get_type(_T(), shape[-1]);
	
	return cv::Mat(shape.ranks(), shape.get_dims().data(), dtype, _data.get());
}

template <typename _T>
_T& Tensor<_T>::val() {
	if (calc_size(_indice) > 1) {
		ErrorExcept(
			"[Tensor<_T>::val] This value is not scalar."
		);
	}

	const size_t index = count_to_elem_index(_steps, _indice, 0);

	*_cnt_rank = 0;

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

	*_cnt_rank = 0;

	return _data[index];
}

template <typename _T>
void Tensor<_T>::clear() {
	_data.reset();
	_steps.clear();
	_indice.clear();
	*_cnt_rank = 0;
}

template <typename _T>
NN_Shape Tensor<_T>::get_shape() {
	*_cnt_rank = 0;

	return calc_shape(_indice);
}

template <typename _T>
NN_Shape Tensor<_T>::get_shape() const {
	*_cnt_rank = 0;

	return calc_shape(_indice);
}

template <typename _T>
_T* Tensor<_T>::get_ptr() {
	*_cnt_rank = 0;

	return _data.get();
}

template <typename _T>
_T* Tensor<_T>::get_ptr() const {
	*_cnt_rank = 0;

	return _data.get();
}

template <typename _T>
std::vector<size_t>& Tensor<_T>::get_steps() {
	return _steps;
}

template <typename _T>
const std::vector<size_t>& Tensor<_T>::get_steps() const {
	return _steps;
}

template <typename _T>
std::vector<std::vector<size_t>>& Tensor<_T>::get_indices() {
	return _indice;
}

template <typename _T>
const std::vector<std::vector<size_t>>& Tensor<_T>::get_indices() const {
	return _indice;
}

template <typename _T>
std::ostream& Tensor<_T>::put(std::ostream& os) {
	if (_indice.size() > 0) {
		int i = 0;

		put_tensor(os, *this, 0, i);
		os << "shape: " << shape_to_str(calc_shape(_indice)) << std::endl;
	}
	else os << "[]" << std::endl;

	*_cnt_rank = 0;

	return os;
}

template <typename _T>
std::ostream& Tensor<_T>::put(std::ostream& os) const {
	if (_indice.size() > 0) {
		int i = 0;

		put_tensor(os, *this, 0, i);
		os << "shape: " << shape_to_str(calc_shape(_indice)) << std::endl;
	}
	else os << "[]" << std::endl;

	*_cnt_rank = 0;

	return os;
}

template <typename _T>
void Tensor<_T>::resize(const NN_Shape& shape) {
	if ((int)_indice.size() != shape.ranks()) _indice.resize(shape.ranks());

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

	*_cnt_rank = 0;
}

template <typename _T>
template <typename _cT>
void Tensor<_T>::cast(const Tensor<_cT>& src) {
	if (get_shape() != src.get_shape()) resize(src.get_shape());

	tbb::parallel_for(
		tbb::blocked_range<size_t>(0, calc_size(_indice)),
		[&](const tbb::blocked_range<size_t>& q) {

		const _cT* p_src = src.get_ptr();
		_T* p_dst = get_ptr();

		for (size_t i = q.begin(); i < q.end(); ++i) {
			size_t src_idx = count_to_elem_index(src.get_steps(), src.get_indices(), i);
			p_dst[i] = (_T)p_src[src_idx];
		}
	}
	);

	*_cnt_rank = 0;
}

template <typename _T>
template <typename _cT>
void Tensor<_T>::cast(const Tensor<_cT>& src) const {
	if (get_shape() != src.get_shape()) resize(src.get_shape());

	tbb::parallel_for(
		tbb::blocked_range<size_t>(0, calc_size(_indice)),
		[&](const tbb::blocked_range<size_t>& q) {

		const _cT* p_src = src.get_ptr();
		_T* p_dst = get_ptr();

		for (size_t i = q.begin(); i < q.end(); ++i) {
			size_t src_idx = count_to_elem_index(src.get_steps(), src.get_indices(), i);
			p_dst[i] = (_T)p_src[src_idx];
		}
	}
	);

	*_cnt_rank = 0;
}

template <typename _T>
Tensor<_T> Tensor<_T>::expand_dims(const Tensor& tensor, int axis) {
	NN_Shape shape = calc_shape(tensor._indice);
	const int ranks = shape.ranks();
	const int curr_axis = axis < 0 ? ranks - axis : axis;

	if (ranks <= curr_axis || 0 > curr_axis) {
		ErrorExcept(
			"[Tensor<_T>::expand_dims] %d axis is out of range.",
			axis
		);
	}

	Tensor<_T> m_tensor = tensor;

	m_tensor._steps.insert(m_tensor._steps.begin() + curr_axis, m_tensor._steps[curr_axis] * shape[curr_axis]);
	m_tensor._indice.insert(m_tensor._indice.begin() + curr_axis, { 0 });

	*tensor._cnt_rank = 0;

	return m_tensor;
}

template <typename _T>
Tensor<_T> Tensor<_T>::expand_dims(const Tensor& tensor, std::initializer_list<int>& axis) {
	NN_Shape shape = calc_shape(tensor._indice);
	const int ranks = shape.ranks();

	Tensor<_T> m_tensor = tensor;

	for (const int& n : axis) {
		const int curr_axis = n < 0 ? ranks - n : n;

		if (ranks <= curr_axis || 0 > curr_axis) {
			ErrorExcept(
				"[Tensor<_T>::expand_dims] %s axis is out of range.",
				shape_to_str(axis)
			);
		}

		m_tensor._steps.insert(m_tensor._steps.begin() + curr_axis, m_tensor._steps[curr_axis] * shape[curr_axis]);
		m_tensor._indice.insert(m_tensor._indice.begin() + curr_axis, { 0 });
	}

	*tensor._cnt_rank = 0;

	return m_tensor;
}

template <typename _T>
Tensor<_T> Tensor<_T>::squeeze(const Tensor& tensor, int axis) {
	NN_Shape shape = calc_shape(tensor._indice);
	const int ranks = shape.ranks();
	const int curr_axis = axis < 0 ? ranks - axis : axis;

	if (ranks <= curr_axis || 0 > curr_axis) {
		ErrorExcept(
			"[Tensor<_T>::squeeze] %d axis is out of range.",
			axis
		);
	}
	else if (shape[curr_axis] > 1) {
		ErrorExcept(
			"[Tensor<_T>::squeeze] Can't squeeze %d axis.",
			axis
		);
	}

	Tensor<_T> m_tensor = tensor;

	m_tensor._steps.erase(m_tensor._steps.begin() + curr_axis);
	m_tensor._indice.erase(m_tensor._indice.begin() + curr_axis);

	*tensor._cnt_rank = 0;

	return m_tensor;
}

template <typename _T>
Tensor<_T> Tensor<_T>::squeeze(const Tensor& tensor, std::initializer_list<int>& axis) {
	NN_Shape shape = calc_shape(tensor._indice);
	const int ranks = shape.ranks();

	Tensor<_T> m_tensor = tensor;

	for (const int& n : axis) {
		const int curr_axis = n < 0 ? ranks - n : n;

		if (ranks <= curr_axis || 0 > curr_axis) {
			ErrorExcept(
				"[Tensor<_T>::squeeze] %s axis is out of range.",
				shape_to_str(axis)
			);
		}
		else if (shape[curr_axis] > 1) {
			ErrorExcept(
				"[Tensor<_T>::squeeze] Can't squeeze %d axis.",
				shape_to_str(axis)
			);
		}

		m_tensor._steps.erase(m_tensor._steps.begin() + curr_axis);
		m_tensor._indice.erase(m_tensor._indice.begin() + curr_axis);
	}

	*tensor._cnt_rank = 0;

	return m_tensor;
}

template <typename _T>
Tensor<_T> Tensor<_T>::zeros(const NN_Shape& shape) {
	Tensor<_T> tensor(shape);

	tensor = _T(0);

	return tensor;
}


/**********************************************/
/*                                            */
/*                  GpuTensor                 */
/*                                            */
/**********************************************/

template <typename _T>
void GpuTensor<_T>::del_func(_T* ptr) {
	try {
		check_cuda(cudaFree(ptr));
	}
	catch (const NN_Exception& e) {
		e.put();
	}
}

template <typename _T>
GpuTensor<_T>::GpuTensor() {
}

template <typename _T>
GpuTensor<_T>::GpuTensor(const NN_Shape& shape) :
	_shape(shape)
{
	_T* ptr = NULL;

	try {
		check_cuda(cudaMalloc(&ptr, sizeof(_T) * _shape.total_size()));

		_data = std::shared_ptr<_T>(ptr, del_func);
	}
	catch (const NN_Exception& e) {
		cudaFree(ptr);

		throw e;
	}
}

template <typename _T>
GpuTensor<_T>::GpuTensor(const GpuTensor& p) :
	_shape(p._shape),
	_data(p._data)
{

}

template <typename _T>
GpuTensor<_T>::GpuTensor(GpuTensor&& p) :
	_shape(p._shape),
	_data(p._data)
{

}

template <typename _T>
GpuTensor<_T>::GpuTensor(const Tensor<_T>& p) :
	_shape(p.get_shape())
{
	_T* dst_ptr = NULL;

	try {
		check_cuda(cudaMalloc(&dst_ptr, sizeof(_T) * _shape.total_size()));
		check_cuda(cudaMemcpy(dst_ptr, p.get_ptr(), sizeof(_T) * _shape.total_size(), cudaMemcpyHostToDevice));
	}
	catch (const NN_Exception& e) {
		cudaFree(dst_ptr);
		e.put();
	}

	_data = std::shared_ptr<_T>(dst_ptr, del_func);
}

template <typename _T>
GpuTensor<_T>::GpuTensor(const GpuTensor& p, const NN_Shape& shape) :
	_data(p._data),
	_shape(shape)
{

}

template <typename _T>
GpuTensor<_T>::~GpuTensor() {

}

template <typename _T>
GpuTensor<_T>& GpuTensor<_T>::operator=(const GpuTensor& p) {
	if (this == &p) return *this;

	if (_data == NULL) {
		_data = p._data;
		_shape = p._shape;
	}
	else {
		if (_shape != p._shape) {
			ErrorExcept(
				"[GpuTensor<_T>::operator=] differnet of src and dst tensors. %s != %s",
				shape_to_str(p._shape),
				shape_to_str(_shape)
			);
		}

		const _T* src_ptr = p.get_ptr();
		_T* dst_ptr = _data.get();

		check_cuda(cudaMemcpy(dst_ptr, src_ptr, sizeof(_T) * _shape.total_size(), cudaMemcpyDeviceToDevice));
	}

	return *this;
}

template <typename _T>
GpuTensor<_T>& GpuTensor<_T>::operator=(const Tensor<_T>& p) {
	if (_data == NULL) {
		_T* dst_ptr = NULL;

		_shape = p.get_shape();
		check_cuda(cudaMalloc(&dst_ptr, sizeof(_T) * _shape.total_size()));
		_data = std::shared_ptr<_T>(dst_ptr, del_func);
	}
	else {
		const NN_Shape src_shape = p.get_shape();

		if (_shape != src_shape) {
			ErrorExcept(
				"[GpuTensor<_T>::operator=] different of src and dst tensors. %s != %s",
				shape_to_str(src_shape),
				shape_to_str(_shape)
			);
		}
	}

	check_cuda(cudaMemcpy(_data.get(), p.get_ptr(), sizeof(_T) * _shape.total_size(), cudaMemcpyHostToDevice));

	return *this;
}

template <typename _T>
NN_Shape GpuTensor<_T>::get_shape() {
	return _shape;
}

template <typename _T>
NN_Shape GpuTensor<_T>::get_shape() const {
	return _shape;
}

template <typename _T>
_T* GpuTensor<_T>::get_ptr() {
	return _data.get();
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

	if (size != _shape.total_size()) {
		_T* ptr = NULL;

		check_cuda(cudaMalloc(&ptr, sizeof(_T) * size));
		_data = std::shared_ptr<_T>(ptr, del_func);
	}

	_shape = shape;
}

template <typename _T>
void GpuTensor<_T>::resize(const std::shared_ptr<_T>& g_data, const NN_Shape& shape) {
	_data = g_data;
	_shape = shape;
}

template <typename _T>
GpuTensor<_T> GpuTensor<_T>::zeros(const NN_Shape& shape) {
	GpuTensor<_T> tmp(shape);

	check_cuda(cudaMemset(tmp.get_ptr(), 0, sizeof(_T) * shape.total_size()));

	return tmp;
}