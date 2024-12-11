#pragma once
#include "nn_tensor.h"


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
	GpuTensor& operator=(const Tensor<_T>& p);

	NN_Shape& get_shape();
	const NN_Shape& get_shape() const;

	_T* get_ptr();
	const _T* get_ptr() const;

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

	_data = std::shared_ptr<_T>(ptr, del_func);
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
GpuTensor<_T>& GpuTensor<_T>::operator=(const Tensor<_T>& p) {
	const NN_Shape h_shape = p.get_shape();

	if (!get_ptr()) {
		_shape = h_shape;

		_T* dev_ptr = NULL;

		check_cuda(cudaMalloc(&dev_ptr, sizeof(_T) * _shape.total_size()));
		_data = std::shared_ptr<_T>(dev_ptr, del_func);
	}
	else if (h_shape != _shape) {
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
_T* GpuTensor<_T>::get_ptr() {
	return _data.get();
}

template <typename _T>
const _T* GpuTensor<_T>::get_ptr() const {
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