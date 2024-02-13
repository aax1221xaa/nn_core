#pragma once

#include "cuda_common.h"
#include "../cuda_source/cast.cuh"


/**********************************************/
/*                                            */
/*                 TensorBase                 */
/*                                            */
/**********************************************/

class TensorBase {
public:
	nn_shape _shape;

	TensorBase();
	TensorBase(const nn_shape& shape);
	size_t get_len() const;
};

/**********************************************/
/*                                            */
/*                    Tensor                  */
/*                                            */
/**********************************************/

template <typename _T>
class HostTensor : public NN_Shared_Ptr, public TensorBase {
public:
	_T* _data;

	HostTensor();
	HostTensor(const nn_shape& shape);
	HostTensor(const HostTensor& p);
	HostTensor(HostTensor&& p);
	~HostTensor();

	const HostTensor& operator=(const HostTensor& p);
	const HostTensor& operator=(HostTensor&& p);

	void clear();
	void set(const nn_shape& shape);
	void put(std::ostream& os) const;

	static HostTensor zeros(const nn_shape& shape);

	template <typename iT>
	static HostTensor zeros_like(const HostTensor<iT>& p);
};

template <typename _T>
HostTensor<_T>::HostTensor() :
	_data(NULL)
{
	id = NULL;
}

template <typename _T>
HostTensor<_T>::HostTensor(const nn_shape& shape) :
	_data(NULL),
	TensorBase(shape)
{
	try {
		size_t size = get_len();
		_data = new _T[size];

		id = linker.create();
	}
	catch (const Exception& e) {
		_data = NULL;
		_shape.clear();
		id = NULL;

		e.Put();
	}
}

template <typename _T>
HostTensor<_T>::HostTensor(const HostTensor& p) :
	_data(p._data),
	TensorBase(p._shape)
{
	id = p.id;

	if (id) ++id->ref_cnt;
}

template <typename _T>
HostTensor<_T>::HostTensor(HostTensor&& p) :
	_data(p._data),
	TensorBase(p._shape)
{
	id = p.id;

	p._data = NULL;
	p.id = NULL;
}

template <typename _T>
HostTensor<_T>::~HostTensor() {
	clear();
}

template <typename _T>
const HostTensor<_T>& HostTensor<_T>::operator=(const HostTensor& p) {
	if (this == &p) return *this;

	clear();

	_data = p._data;
	_shape = p._shape;

	id = p.id;

	if (id) ++id->ref_cnt;

	return *this;
}

template <typename _T>
const HostTensor<_T>& HostTensor<_T>::operator=(HostTensor&& p) {
	clear();

	_data = p._data;
	_shape = p._shape;

	id = p.id;

	p._data = NULL;
	p.id = NULL;

	return *this;
}

template <typename _T>
void HostTensor<_T>::put(std::ostream& os) const {
	std::vector<size_t> indicator;
	size_t step = 1;
	bool end_flag = false;

	for (nn_shape::reverse_iterator i = _shape.rbegin(); i != _shape.rend(); ++i) {
		step *= *i;
		indicator.push_back(step);
	}

	os << "Tensor: " << put_shape(_shape) << std::endl << std::endl;

	size_t len = get_len();

	for (size_t i = 0; i < len;) {
		for (const uint& n : indicator) {
			if (i % n == 0) os << '[';
		}

		os << _data[i] << ", ";
		++i;

		for (const uint& n : indicator) {
			if (i % n == 0) {
				os << "], ";
				end_flag = true;
			}
		}
		if (end_flag) {
			os << std::endl;
			end_flag = false;
		}
	}
}

template <typename _T>
void HostTensor<_T>::clear() {
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
}

template <typename _T>
void HostTensor<_T>::set(const nn_shape& shape) {
	try {
		clear();

		_shape = shape;
		_data = new _T[get_len()];

		id = linker.create();
	}
	catch (const Exception& e) {
		_shape.clear();
		_data = NULL;

		throw e;
	}
}

template <typename _T>
HostTensor<_T> HostTensor<_T>::zeros(const nn_shape& shape) {
	HostTensor<_T> tensor(shape);

	memset(tensor._data, 0, sizeof(_T) * tensor.get_len());

	return tensor;
}

template <typename _T>
template <typename iT>
HostTensor<_T> HostTensor<_T>::zeros_like(const HostTensor<iT>& p) {
	HostTensor<_T> tensor(p._shape);

	memset(tensor._data, 0, sizeof(_T) * tensor.get_len());

	return tensor;
}

template <typename _T>
std::ostream& operator<<(std::ostream& os, const HostTensor<_T>& tensor) {
	tensor.put(os);

	return os;
}

/**********************************************/
/*                                            */
/*                 DeviceTensor               */
/*                                            */
/**********************************************/

template <typename _T>
class DeviceTensor : public NN_Shared_Ptr, public TensorBase {
public:
	_T* _data;

	DeviceTensor();
	DeviceTensor(const nn_shape& shape);
	DeviceTensor(const DeviceTensor& p);
	DeviceTensor(DeviceTensor&& p);
	~DeviceTensor();

	const DeviceTensor& operator=(const DeviceTensor& p);
	const DeviceTensor& operator=(DeviceTensor&& p);

	void clear();
	void set(const nn_shape& shape);

	static DeviceTensor zeros(const nn_shape& shape);

	template <typename iT>
	static DeviceTensor<_T> zeros_like(const DeviceTensor<iT>& p);
};

template <typename _T>
DeviceTensor<_T>::DeviceTensor() :
	_data(NULL)
{
	id = NULL;
}

template <typename _T>
DeviceTensor<_T>::DeviceTensor(const nn_shape& shape) :
	_data(NULL),
	TensorBase(shape)
{
	try {
		size_t size = sizeof(_T) * get_len();
		check_cuda(cudaMalloc(&_data, size));

		id = linker.create();
	}
	catch (const Exception& e) {
		_data = NULL;
		_shape.clear();
		id = NULL;

		e.Put();
	}
}

template <typename _T>
DeviceTensor<_T>::DeviceTensor(const DeviceTensor& p) :
	_data(p._data),
	TensorBase(p._shape)
{
	id = p.id;

	if (id) ++id->ref_cnt;
}

template <typename _T>
DeviceTensor<_T>::DeviceTensor(DeviceTensor&& p) :
	_data(p._data),
	TensorBase(p._shape)
{
	id = p.id;

	p._data = NULL;
	p.id = NULL;
}

template <typename _T>
DeviceTensor<_T>::~DeviceTensor() {
	clear();
}

template <typename _T>
const DeviceTensor<_T>& DeviceTensor<_T>::operator=(const DeviceTensor& p) {
	if (this == &p) return *this;

	clear();

	_data = p._data;
	_shape = p._shape;

	id = p.id;

	if (id) ++id->ref_cnt;

	return *this;
}

template <typename _T>
const DeviceTensor<_T>& DeviceTensor<_T>::operator=(DeviceTensor&& p) {
	clear();

	_data = p._data;
	_shape = p._shape;

	id = p.id;

	p._data = NULL;
	p.id = NULL;

	return *this;
}

template <typename _T>
void DeviceTensor<_T>::clear() {
	if (id) {
		if (id->ref_cnt > 1) --id->ref_cnt;
		else {
			check_cuda(cudaFree(_data));
			linker.erase(id);
		}
	}

	_data = NULL;
	_shape.clear();

	id = NULL;
}

template <typename _T>
void DeviceTensor<_T>::set(const nn_shape& shape) {
	try {
		clear();

		_shape = shape;
		check_cuda(cudaMalloc(&_data, sizeof(_T) * get_len()));


		id = linker.create();
	}
	catch (const Exception& e) {
		_shape.clear();
		_data = NULL;

		throw e;
	}
}

template <typename _T>
DeviceTensor<_T> DeviceTensor<_T>::zeros(const nn_shape& shape) {
	DeviceTensor<_T> tensor(shape);

	check_cuda(cudaMemset(tensor._data, 0, sizeof(_T) * tensor.get_len()));

	return tensor;
}

template <typename _T>
template <typename iT>
DeviceTensor<_T> DeviceTensor<_T>::zeros_like(const DeviceTensor<iT>& p) {
	DeviceTensor<_T> tensor(p._shape);

	check_cuda(cudaMemset(tensor._data, 0, sizeof(_T) * tensor.get_len()));

	return tensor;
}

/**********************************************/
/*                                            */
/*             mem copy function              */
/*                                            */
/**********************************************/

/*          copy Tensor to NN_Tensor          */

template <typename _T>
void to_device_tensor(const HostTensor<_T>& src, DeviceTensor<_T>& dst) {
	if (src.get_len() != dst.get_len()) {
		ErrorExcept(
			"[copy_to_nn_tensor()] src and dst length are different. %llu != %llu.",
			src.get_len(), dst.get_len()
		);
	}
	check_cuda(cudaMemcpy(dst._data, src._data, sizeof(_T) * src.get_len(), cudaMemcpyHostToDevice));
}

/*          copy NN_Tensor to Tensor          */

template <typename _T>
void to_host_tensor(const DeviceTensor<_T>& src, HostTensor<_T>& dst) {
	if (src.get_len() != dst.get_len()) {
		ErrorExcept(
			"[copy_to_tensor()] src and dst length are different. %llu != %llu.",
			src.get_len(), dst.get_len()
		);
	}
	check_cuda(cudaMemcpy(dst._data, src._data, sizeof(_T) * src.get_len(), cudaMemcpyDeviceToHost));
}

void set_uniform(DeviceTensor<nn_type>& p);

template <typename iT, typename oT>
void cast(const DeviceTensor<iT>& src, DeviceTensor<oT>& dst) {
	if (src.get_len() != dst.get_len()) {
		ErrorExcept(
			"[cast()] src and dst length are different. %llu != %llu.",
			src.get_len(), dst.get_len()
		);
	}

	type_cast(get_type(src._data), src._data, get_type(dst._data), dst_data, dst.get_len());
}