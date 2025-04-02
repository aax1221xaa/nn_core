#pragma once
#include "nn_tensor.h"



/**********************************************/
/*                                            */
/*                 HostTensor                 */
/*                                            */
/**********************************************/

template <typename _H>
class HostTensor {
	std::shared_ptr<_H[]> _data;
	NN_Shape _shape;

public:
	HostTensor();
	HostTensor(const NN_Shape& shape);
	HostTensor(const HostTensor& tensor);
	HostTensor(HostTensor&& tensor);
	HostTensor(const NN_Tensor<_H>& tensor);
	~HostTensor();

	HostTensor& operator=(const HostTensor& tensor);
	HostTensor& operator=(HostTensor&& tensor);
	HostTensor& operator=(const NN_Tensor<_H>& tensor);
	HostTensor& operator=(NN_Tensor<_H>&& tensor);

	_H* get_ptr();
	const _H* get_ptr() const;

	const NN_Shape& get_shape();
	const NN_Shape& get_shape() const;

	const std::shared_ptr<_H[]>& get_shared_ptr();
	const std::shared_ptr<_H[]>& get_shared_ptr() const;
};

template <typename _H>
HostTensor<_H>::HostTensor() {

}

template <typename _H>
HostTensor<_H>::HostTensor(const NN_Shape& shape) :
	_shape(shape)
{
	_data = std::shared_ptr<_H[]>(new _H[shape.total_size()]);
}

template <typename _H>
HostTensor<_H>::HostTensor(const HostTensor& tensor) :
	_data(tensor._data),
	_shape(tensor._shape)
{

}

template <typename _H>
HostTensor<_H>::HostTensor(HostTensor&& tensor) :
	_data(std::move(tensor._data)),
	_shape(std::move(tensor._shape))
{

}

template <typename _H>
HostTensor<_H>::HostTensor(const NN_Tensor<_H>& tensor) {
	_shape = tensor.get_shape();

	NN_Tensor<_H> tmp(_shape);

	tmp = tensor;
	_data = tmp.get_shared_ptr();
}

template <typename _H>
HostTensor<_H>::~HostTensor() {

}

template <typename _H>
HostTensor<_H>& HostTensor<_H>::operator=(const HostTensor& tensor) {
	if (this == &tensor) return *this;

	if (_data == NULL) {
		_data = tensor._data;
		_shape = tensor._shape;
	}
	else {
		if (_shape != tensor._shape) {
			ErrorExcept(
				"[HostTensor<_T>::operator=] src%s and dst%s tensors are different.",
				shape_to_str(tensor._shape),
				shape_to_str(_shape)
			);
		}
		
		const size_t len = _shape.total_size();

		memcpy_s(_data.get(), sizeof(_H) * len, tensor._data.get(), sizeof(_H) * len);
	}

	return *this;
}

template <typename _H>
HostTensor<_H>& HostTensor<_H>::operator=(HostTensor&& tensor) {
	if (_data == NULL) {
		_data = tensor._data;
		_shape = tensor._shape;
	}
	else {
		if (_shape != tensor._shape) {
			ErrorExcept(
				"[HostTensor<_T>::operator=] src%s and dst%s tensors are different.",
				shape_to_str(tensor._shape),
				shape_to_str(_shape)
			);
		}

		const size_t len = _shape.total_size();

		memcpy_s(_data.get(), sizeof(_H) * len, tensor._data.get(), sizeof(_H) * len);
	}

	return *this;
}

template <typename _H>
HostTensor<_H>& HostTensor<_H>::operator=(const NN_Tensor<_H>& tensor) {
	NN_Tensor<_H> tmp(tensor.get_shape());

	tmp = tensor;
	
	if (_data == NULL) {
		_data = tmp.get_shared_ptr();
		_shape = tmp.get_shape();
	}
	else {
		if (_shape != tmp.get_shape()) {
			ErrorExcept(
				"[HostTensor<_T>::operator=] src%s and dst%s tensors are differnent.",
				shape_to_str(_shape),
				shape_to_str(tmp.get_shape())
			);
		}

		const size_t len = _shape.total_size();

		memcpy_s(_data.get(), sizeof(_H) * len, tmp.get_ptr(), sizeof(_H) * len);
	}

	return *this;
}

template <typename _H>
HostTensor<_H>& HostTensor<_H>::operator=(NN_Tensor<_H>&& tensor) {
	if (_data == NULL) {
		_data = tensor.get_shared_ptr();
		_shape = tensor.get_shape();
	}
	else {
		if (_shape != tensor.get_shape()) {
			ErrorExcept(
				"[HostTensor<_T>::operator=] src%s and dst%s tensors are differnent.",
				shape_to_str(_shape),
				shape_to_str(tensor.get_shape())
			);
		}

		const size_t len = _shape.total_size();

		memcpy_s(_data.get(), sizeof(_H) * len, tensor.get_ptr(), sizeof(_H) * len);
	}

	return *this;
}

template <typename _H>
_H* HostTensor<_H>::get_ptr() {
	return _data.get();
}

template <typename _H>
const _H* HostTensor<_H>::get_ptr() const {
	return _data.get();
}

template <typename _H>
const NN_Shape& HostTensor<_H>::get_shape() {
	return _shape;
}

template <typename _H>
const NN_Shape& HostTensor<_H>::get_shape() const {
	return _shape;
}

template <typename _H>
const std::shared_ptr<_H[]>& HostTensor<_H>::get_shared_ptr() {
	return _data;
}

template <typename _H>
const std::shared_ptr<_H[]>& HostTensor<_H>::get_shared_ptr() const {
	return _data;
}