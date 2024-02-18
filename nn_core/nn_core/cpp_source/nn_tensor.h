#pragma once

#include "cuda_common.h"


/**********************************************/
/*                                            */
/*                 TensorBase                 */
/*                                            */
/**********************************************/

class TensorBase {
public:
	bool _is_valid;
	size_t _len;
	nn_shape _shape;

	TensorBase();
	TensorBase(const nn_shape& shape);
	
	static size_t get_len(const nn_shape& shape);
	static bool check_shape(const nn_shape& shape);
};

/**********************************************/
/*                                            */
/*                    Tensor                  */
/*                                            */
/**********************************************/

template <typename _T>
class Tensor : public NN_Shared_Ptr, public TensorBase {
protected:
	static void put_tensor(std::ostream& os, const Tensor<_T>& tensor, uint& current_lank, uint& index);

public:
	_T* _data;

	Tensor();
	Tensor(const nn_shape& shape);
	Tensor(const Tensor& p);
	Tensor(Tensor&& p);
	~Tensor();

	const Tensor& operator=(const Tensor& p);
	const Tensor& operator=(Tensor&& p);

	void clear();
	void set(const nn_shape& shape);
	void put(std::ostream& os) const;
};

template <typename _T>
void Tensor<_T>::put_tensor(std::ostream& os, const Tensor<_T>& tensor, uint& current_lank, uint& index) {
	if (current_lank < tensor._shape.size() - 1) {
		for (int i = 0; i < tensor._shape[current_lank]; ++i) {
			++current_lank;

			os << '[';
			put_tensor(os, tensor, current_lank, index);
			os << "]\n";

			--current_lank;
		}
	}
	else {
		os << '[';
		for(int i = 0; i < tensor._shape[current_lank - 1]; ++i)
			os << tensor._data[index++] << ", ";
		os << "]\n";
	}
}

template <typename _T>
Tensor<_T>::Tensor() :
	_data(NULL)
{
	id = NULL;
}

template <typename _T>
Tensor<_T>::Tensor(const nn_shape& shape) :
	_data(NULL),
	TensorBase(shape)
{
	if (_is_valid) {
		_data = new _T[_len];
		id = linker.create();
	}
	else id = NULL;
}

template <typename _T>
Tensor<_T>::Tensor(const Tensor& p) :
	_data(p._data)
{
	_shape = p._shape;
	_is_valid = p._is_valid;
	_len = p._len;

	id = p.id;

	if (id) ++id->ref_cnt;
}

template <typename _T>
Tensor<_T>::Tensor(Tensor&& p) :
	_data(p._data)
{
	_shape = p._shape;
	_is_valid = p._is_valid;
	_len = p._len;

	id = p.id;

	p._data = NULL;
	p.id = NULL;
}

template <typename _T>
Tensor<_T>::~Tensor() {
	clear();
}

template <typename _T>
const Tensor<_T>& Tensor<_T>::operator=(const Tensor& p) {
	if (this == &p) return *this;

	clear();

	_data = p._data;
	_shape = p._shape;
	_is_valid = p._is_valid;
	_len = p._len;

	id = p.id;

	if (id) ++id->ref_cnt;

	return *this;
}

template <typename _T>
const Tensor<_T>& Tensor<_T>::operator=(Tensor&& p) {
	clear();

	_data = p._data;
	_shape = p._shape;
	_is_valid = p._is_valid;
	_len = p._len;

	id = p.id;

	p._data = NULL;
	p.id = NULL;

	return *this;
}

template <typename _T>
void Tensor<_T>::put(std::ostream& os) const {
	uint lank = 0;
	uint index = 0;
	
	if (_shape.size() > 0) {
		os << "Dimenstion: " << put_shape(_shape) << std::endl << std::endl;

		put_tensor(os, *this, lank, index);
	}
	else os << "[]\n";
}

template <typename _T>
void Tensor<_T>::clear() {
	if (id) {
		if (id->ref_cnt > 1) --id->ref_cnt;
		else {
			delete[] _data;
			linker.erase(id);
		}
	}
	id = NULL;
	_data = NULL;
	_shape.clear();
	_is_valid = false;
	_len = 0;
}

template <typename _T>
void Tensor<_T>::set(const nn_shape& shape) {
	clear();

	_is_valid = check_shape(shape);

	if (_is_valid) {
		_shape = shape;
		_len = get_len(shape);
		_data = new _T[_len];

		id = linker.create();
	}
}

template <typename _T>
std::ostream& operator<<(std::ostream& os, const Tensor<_T>& tensor) {
	tensor.put(os);

	return os;
}


template <typename _T>
Tensor<_T> zeros(const nn_shape& shape) {
	Tensor<_T> tensor(shape);

	memset(tensor._data, 0, sizeof(_T) * tensor._len);

	return tensor;
}

template <typename _T>
Tensor<_T> zeros_like(const Tensor<_T>& src) {
	Tensor<_T> tensor(src._shape);

	memset(tensor._data, 0, sizeof(_T) * tensor._len);

	return tensor;
}


/**********************************************/
/*                                            */
/*                  GpuTensor                 */
/*                                            */
/**********************************************/

template <typename _T>
class GpuTensor : public NN_Shared_Ptr, public TensorBase {
public:
	_T* _data;

	GpuTensor();
	GpuTensor(const nn_shape& shape);
	GpuTensor(const GpuTensor& p);
	GpuTensor(GpuTensor&& p);
	~GpuTensor();

	const GpuTensor& operator=(const GpuTensor& p);
	const GpuTensor& operator=(GpuTensor&& p);

	void clear();
	void set(const nn_shape& shape);
};

template <typename _T>
GpuTensor<_T>::GpuTensor() :
	_data(NULL)
{
	id = NULL;
}

template <typename _T>
GpuTensor<_T>::GpuTensor(const nn_shape& shape) :
	_data(NULL),
	TensorBase(shape)
{
	if (_is_valid && (cudaMalloc(&_data, sizeof(_T) * _len) == cudaSuccess)) {
		id = linker.create();
	}
	else {
		_is_valid = false;
		id = NULL;
	}
}

template <typename _T>
GpuTensor<_T>::GpuTensor(const GpuTensor& p) :
	_data(p._data)
{
	_shape = p._shape;
	_is_valid = p._is_valid;
	_len = p._len;

	id = p.id;

	if (id) ++id->ref_cnt;
}

template <typename _T>
GpuTensor<_T>::GpuTensor(GpuTensor&& p) :
	_data(p._data)
{
	_shape = p._shape;
	_is_valid = p._is_valid;
	_len = p._len;

	id = p.id;

	p._data = NULL;
	p.id = NULL;
}

template <typename _T>
GpuTensor<_T>::~GpuTensor() {
	clear();
}

template <typename _T>
const GpuTensor<_T>& GpuTensor<_T>::operator=(const GpuTensor& p) {
	if (this == &p) return *this;

	clear();

	_data = p._data;
	_shape = p._shape;
	_is_valid = p._is_valid;
	_len = p._len;

	id = p.id;

	if (id) ++id->ref_cnt;

	return *this;
}

template <typename _T>
const GpuTensor<_T>& GpuTensor<_T>::operator=(GpuTensor&& p) {
	clear();

	_data = p._data;
	_shape = p._shape;
	_is_valid = p._is_valid;
	_len = p._len;

	id = p.id;

	p._data = NULL;
	p.id = NULL;

	return *this;
}

template <typename _T>
void GpuTensor<_T>::clear() {
	if (id) {
		if (id->ref_cnt > 1) --id->ref_cnt;
		else {
			cudaFree(_data);
			linker.erase(id);
		}
	}
	id = NULL;
	_data = NULL;
	_shape.clear();
	_is_valid = false;
	_len = 0;
}

template <typename _T>
void GpuTensor<_T>::set(const nn_shape& shape) {
	clear();

	_is_valid = check_shape(shape);
	_len = get_len(shape);

	if (_is_valid && (cudaMalloc(&_data, sizeof(_T) * _len) == cudaSuccess)) {
		_shape = shape;
		id = linker.create();
	}
	else {
		_is_valid = false;
		_len = 0;

		id = NULL;
	}
}

template <class _T>
GpuTensor<_T> zeros(const nn_shape& shape) {
	GpuTensor<_T> tensor(shape);

	check_cuda(cudaMemset(tensor._data, 0, sizeof(_T) * tensor._len));

	return tensor;
}

template <class _T>
GpuTensor<_T> zeros_like(const GpuTensor<_T>& src) {
	GpuTensor<_T> tensor(src._shape);

	check_cuda(cudaMemset(tensor._data, 0, sizeof(_T) * tensor._len));

	return tensor;
}


/**********************************************/
/*                                            */
/*                 Host & Gpu                 */
/*                                            */
/**********************************************/

template <class _T>
void copy_to_gpu(const Tensor<_T>& src, GpuTensor<_T>& dst) {
	if (src._len != dst._len) {
		ErrorExcept(
			"[copy_to_gpu] The space of sdt and dst are different. %ld != %ld",
			src._len, dst._len
		);
	}

	check_cuda(cudaMemcpy(dst._data, src._data, sizeof(_T) * src._len, cudaMemcpyHostToDevice));
}

template <class _T>
void copy_to_host(const GpuTensor<_T>& src, Tensor<_T>& dst) {
	if (src._len != dst._len) {
		ErrorExcept(
			"[copy_to_host] The space of sdt and dst are different. %ld != %ld",
			src._len, dst._len
		);
	}

	check_cuda(cudaMemcpy(dst._data, src._data, sizeof(_T) * src._len, cudaMemcpyDeviceToHost));
}