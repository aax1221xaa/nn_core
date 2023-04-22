#pragma once
#include "cuda_common.h"
#include "../cuda_source/cast.cuh"


/**********************************************/
/*                                            */
/*                    Tensor                  */
/*                                            */
/**********************************************/

template <typename _T>
class Tensor : public NN_Shared_Ptr {
protected:
	static uint calc_len_size(const nn_shape& shape);
	static std::vector<uint> calc_steps(const nn_shape& shape);
	static uint calc_addr(
		const nn_shape& start,
		const nn_shape& shape,
		const std::vector<uint>& steps, 
		const nn_shape& indices
	);

public:
	_T* _data;

	cuint _elem_size;
	uint _len;

	nn_shape _shape;
	std::vector<uint> _steps;

	nn_shape _start;

	Tensor();
	Tensor(const nn_shape& shape);
	Tensor(const Tensor& p);
	Tensor(Tensor&& p);
	~Tensor();

	const Tensor& operator=(const Tensor& p);
	const Tensor& operator=(Tensor&& p);

	void clear();
	void set(const nn_shape& shape);

	template <typename cast_type>
	Tensor<cast_type> cast();

	Tensor& slice(nn_shape&& start, nn_shape&& end);
	_T& get(const nn_shape& indices);
	void put(std::ostream& os) const;

	static Tensor zeros(const nn_shape& shape);

	template <typename in_type>
	static Tensor zeros_like(const Tensor<in_type>& p);
};

template <typename _T>
uint Tensor<_T>::calc_len_size(const nn_shape& shape) {
	uint size = 1;

	for (const int& n : shape) {
		if (n < 1) {
			ErrorExcept(
				"[Tensor::calc_mem_size()] invalid shapes %s.",
				dimension_to_str(shape)
			);
		}
		size *= (uint)n;
	}

	return size;
}

template <typename _T>
std::vector<uint> Tensor<_T>::calc_steps(const nn_shape& shape) {
	std::vector<uint> steps(shape.size());
	uint step = 1;

	for (size_t i = shape.size(); i > 0; --i) {
		steps[i - 1] = step;
		step *= (uint)shape[i - 1];
	}

	return steps;
}

template <typename _T>
uint Tensor<_T>::calc_addr(
	const nn_shape& start,
	const nn_shape& shape,
	const std::vector<uint>& steps,
	const nn_shape& indices
) {
	if (shape.size() != indices.size()) {
		ErrorExcept(
			"[Tensor::calc_addr()] can't calculate address by indices: %s.",
			dimension_to_str(indices)
		);
	}
	for (int i = 0; i < indices.size(); ++i) {
		if (indices[i] < 0 || indices[i] >= shape[i]) {
			ErrorExcept(
				"[Tensor::calc_addr()] overflowed indices: %s",
				dimension_to_str(indices)
			);
		}
	}

	uint offset = 0;
	for (int i = 0; i < indices.size(); ++i) offset += steps[i] * indices[i] + start[i];

	return offset;
}

template <typename _T>
Tensor<_T>::Tensor() :
	_data(NULL),
	_elem_size(sizeof(_T)),
	_len(0)
{
	id = NULL;
}

template <typename _T>
Tensor<_T>::Tensor(const nn_shape& shape) :
	_data(NULL),
	_elem_size(sizeof(_T)),
	_len(0),
	_shape(shape),
	_start(shape.size(), 0)
{
	try {
		_len = calc_len_size(_shape);
		_steps = calc_steps(_shape);
		_data = new _T[_len];

		if (_data == NULL) {
			ErrorExcept(
				"[Tensor::Tensor()] faild create memory. %s.",
				dimension_to_str(_shape)
			);
		}

		id = linker.Create();
	}
	catch (const Exception& e) {
		_data = NULL;
		_len = 0;
		_shape.clear();
		_start.clear();
		id = NULL;

		e.Put();
	}
}

template <typename _T>
Tensor<_T>::Tensor(const Tensor& p) :
	_data(p._data),
	_elem_size(p._elem_size),
	_len(p._len),
	_steps(p._steps),
	_shape(p._shape),
	_start(p._start)
{
	id = p.id;

	if (id) ++id->ref_cnt;
}

template <typename _T>
Tensor<_T>::Tensor(Tensor&& p) :
	_data(p._data),
	_elem_size(p._elem_size),
	_len(p._len),
	_steps(p._steps),
	_shape(p._shape),
	_start(p._start)
{
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
	_len = p._len;
	_steps = p._steps;
	_shape = p._shape;
	_start = p._start;

	id = p.id;

	if (id) ++id->ref_cnt;

	return *this;
}

template <typename _T>
const Tensor<_T>& Tensor<_T>::operator=(Tensor&& p) {
	clear();

	_data = p._data;
	_len = p._len;
	_steps = p._steps;
	_shape = p._shape;
	_start = p._start;

	id = p.id;

	p._data = NULL;
	p.id = NULL;

	return *this;
}

template <typename _T>
void Tensor<_T>::put(std::ostream& os) const {
	std::vector<uint> indicator;
	uint step = 1;
	bool end_flag = false;

	for (auto iter = _shape.rbegin(); iter != _shape.rend(); ++iter) {
		step *= *iter;
		indicator.push_back(step);
	}

	os << "Tensor: " << dimension_to_str(_shape) << std::endl << std::endl;

	for (uint i = 0; i < _len;) {
		for (const uint& n : indicator) {
			if (i % n == 0) os << '[';
		}

		os << _data[i] << ", ";
		++i;

		for (const uint& n :indicator) {
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
void Tensor<_T>::clear() {
	if (id) {
		if (id->ref_cnt > 1) --id->ref_cnt;
		else {
			delete[] _data;
			linker.Erase(id);
		}
	}

	_data = NULL;
	_len = 0;
	_steps.clear();
	_shape.clear();
	_start.clear();

	id = NULL;
}

template <typename _T>
void Tensor<_T>::set(const nn_shape& shape) {
	try {
		clear();

		_shape = shape;

		_len = calc_len_size(_shape);
		_steps = calc_steps(_shape);
		_start = nn_shape(shape.size(), 0);
		_data = new _T[_len];

		if (_data == NULL) {
			ErrorExcept(
				"[Tensor::set()] faild create memory. %s.",
				dimension_to_str(_shape)
			);
		}

		id = linker.Create();
	}
	catch (const Exception& e) {
		free(_data);

		_shape.clear();
		_start.clear();
		_len = 0;
		_data = NULL;

		throw e;
	}
}

template <typename _T>
template <typename cast_type>
Tensor<cast_type> Tensor<_T>::cast() {
	Tensor<cast_type> tensor(_shape);

	for (uint i = 0; i < _len; ++i) tensor._data[i] = (cast_type)_data[i];

	return tensor;
}

template <typename _T>
_T& Tensor<_T>::get(const nn_shape& indices) {
	uint offset = calc_addr(_start, _shape, _steps, indices);

	return _data[offset];
}

template <typename _T>
Tensor<_T>& Tensor<_T>::slice(nn_shape&& start, nn_shape&& end) {
	// origin: [64, 28, 28, 3]
	// slice: [-1. 9, 9, -1] ~ [-1, 19, 19, -1]
	// out: [64, 10, 10, 3]
	
	return *this;
}

template <typename _T>
Tensor<_T> Tensor<_T>::zeros(const nn_shape& shape) {
	Tensor<_T> tensor(shape);

	for (uint i = 0; i < tensor._len; ++i) tensor._data[i] = _T();

	return tensor;
}

template <typename _T>
template <typename in_type>
Tensor<_T> Tensor<_T>::zeros_like(const Tensor<in_type>& p) {
	Tensor<_T> tensor(p._shape);

	for (uint i = 0; i < tensor._len; ++i) tensor._data[i] = _T();

	return tensor;
}

template <typename _T>
std::ostream& operator<<(std::ostream& os, const Tensor<_T>& tensor) {
	tensor.put(os);

	return os;
}

/**********************************************/
/*                                            */
/*                  NN_Tensor                 */
/*                                            */
/**********************************************/

template <typename _T>
class NN_Tensor : public NN_Shared_Ptr {
protected:
	static uint calc_len_size(const nn_shape& shape);
	static std::vector<uint> calc_steps(const nn_shape& shape);

public:
	static cudaStream_t _s;

	_T* _data;

	cuint _elem_size;
	uint _len;

	nn_shape _shape;
	std::vector<uint> _steps;

	nn_shape _start;

	NN_Tensor();
	NN_Tensor(const nn_shape& shape);
	NN_Tensor(const NN_Tensor& p);
	NN_Tensor(NN_Tensor&& p);
	~NN_Tensor();

	const NN_Tensor& operator=(const NN_Tensor& p);
	const NN_Tensor& operator=(NN_Tensor&& p);

	void clear();
	void set(const nn_shape& shape);

	template <typename cast_type>
	NN_Tensor<cast_type> cast();

	static NN_Tensor zeros(const nn_shape& shape);

	template <typename in_type>
	static NN_Tensor<_T> zeros_like(const NN_Tensor<in_type>& p);
};

template <typename _T>
cudaStream_t NN_Tensor<_T>::_s = NULL;

template <typename _T>
uint NN_Tensor<_T>::calc_len_size(const nn_shape& shape) {
	uint size = 1;

	for (const int& n : shape) {
		if (n < 1) {
			ErrorExcept(
				"[NN_Tensor::calc_mem_size()] invalid shapes %s.",
				dimension_to_str(shape)
			);
		}
		size *= (uint)n;
	}

	return size;
}

template <typename _T>
std::vector<uint> NN_Tensor<_T>::calc_steps(const nn_shape& shape) {
	std::vector<uint> steps;
	uint step = 1;

	for (size_t i = shape.size(); i > 0; --i) {
		steps.push_back(step);
		step *= (uint)shape[i - 1];
	}

	return steps;
}

template <typename _T>
NN_Tensor<_T>::NN_Tensor() :
	_data(NULL),
	_elem_size(sizeof(_T)),
	_len(0)
{
	id = NULL;
}

template <typename _T>
NN_Tensor<_T>::NN_Tensor( const nn_shape& shape) :
	_data(NULL),
	_elem_size(sizeof(_T)),
	_len(0),
	_shape(shape),
	_start(shape.size(), 0)
{
	try {
		_len = calc_len_size(_shape);
		_steps = calc_steps(_shape);
		
		check_cuda(cudaMalloc(&_data, sizeof(_T) * _len));

		if (_data == NULL) {
			ErrorExcept(
				"[Tensor::Tensor()] faild create memory. %s.",
				dimension_to_str(_shape)
			);
		}

		id = linker.Create();
	}
	catch (const Exception& e) {
		_data = NULL;
		_len = 0;
		_shape.clear();
		_start.clear();
		id = NULL;

		e.Put();
	}
}

template <typename _T>
NN_Tensor<_T>::NN_Tensor(const NN_Tensor& p) :
	_data(p._data),
	_elem_size(p._elem_size),
	_len(p._len),
	_steps(p._steps),
	_shape(p._shape),
	_start(p._start)
{
	id = p.id;

	if (id) ++id->ref_cnt;
}

template <typename _T>
NN_Tensor<_T>::NN_Tensor(NN_Tensor&& p) :
	_data(p._data),
	_elem_size(p._elem_size),
	_len(p._len),
	_steps(p._steps),
	_shape(p._shape),
	_start(p._start)
{
	id = p.id;

	p._data = NULL;
	p.id = NULL;
}

template <typename _T>
NN_Tensor<_T>::~NN_Tensor() {
	clear();
}

template <typename _T>
const NN_Tensor<_T>& NN_Tensor<_T>::operator=(const NN_Tensor& p) {
	if (this == &p) return *this;

	clear();

	_data = p._data;
	_len = p._len;
	_steps = p._steps;
	_shape = p._shape;
	_start = p._start;

	id = p.id;

	if (id) ++id->ref_cnt;

	return *this;
}

template <typename _T>
const NN_Tensor<_T>& NN_Tensor<_T>::operator=(NN_Tensor&& p) {
	clear();

	_data = p._data;
	_len = p._len;
	_steps = p._steps;
	_shape = p._shape;
	_start = p._start;

	id = p.id;

	p._data = NULL;
	p.id = NULL;

	return *this;
}

template <typename _T>
void NN_Tensor<_T>::clear() {
	if (id) {
		if (id->ref_cnt > 1) --id->ref_cnt;
		else {
			check_cuda(cudaFree(_data));
			linker.Erase(id);
		}
	}

	_data = NULL;
	_len = 0;
	_steps.clear();
	_shape.clear();
	_start.clear();

	id = NULL;
}

template <typename _T>
void NN_Tensor<_T>::set(const nn_shape& shape) {
	try {
		clear();

		_shape = shape;

		_len = calc_len_size(_shape);
		_steps = calc_steps(_shape);
		_start = nn_shape(shape.size(), 0);
		
		check_cuda(cudaMalloc(&_data, sizeof(_T) * _len));

		if (_data == NULL) {
			ErrorExcept(
				"[Tensor::set()] faild create memory. %s.",
				dimension_to_str(_shape)
			);
		}

		id = linker.Create();
	}
	catch (const Exception& e) {
		free(_data);

		_shape.clear();
		_start.clear();
		_len = 0;
		_data = NULL;

		throw e;
	}
}

template <typename _T>
template <typename cast_type>
NN_Tensor<cast_type> NN_Tensor<_T>::cast() {
	NN_Tensor<cast_type> out_tensor(_shape);

	dtype dst_type = get_type(out_tensor._data);
	dtype src_type = get_type(_data);

	type_cast(_s, dst_type, out_tensor._data, src_type, _data, _len);

	return out_tensor;
}

template <typename _T>
NN_Tensor<_T> NN_Tensor<_T>::zeros(const nn_shape& shape) {
	NN_Tensor<_T> tensor(shape);

	check_cuda(cudaMemset(tensor._data, 0, sizeof(_T) * tensor._len));

	return tensor;
}

template <typename _T>
template <typename in_type>
NN_Tensor<_T> NN_Tensor<_T>::zeros_like(const NN_Tensor<in_type>& p) {
	NN_Tensor<_T> tensor(p._shape);

	check_cuda(cudaMemset(tensor._data, 0, sizeof(_T) * tensor._len));

	return tensor;
}

/**********************************************/
/*                                            */
/*             mem copy function              */
/*                                            */
/**********************************************/

/*          copy Tensor to NN_Tensor          */

template <typename _T>
void copy_to_nn_tensor(const Tensor<_T>& src, NN_Tensor<_T>& dst) {
	if (src._len != dst._len) {
		ErrorExcept(
			"[copy_to_nn_tensor()] src and dst length are different. %d != %d.",
			src._len, dst._len
		);
	}

	check_cuda(cudaMemcpy(dst._data, src._data, sizeof(_T) * src._len, cudaMemcpyHostToDevice));
}

/*          copy NN_Tensor to Tensor          */

template <typename _T>
void copy_to_tensor(const NN_Tensor<_T>& src, Tensor<_T>& dst) {
	if (src._len != dst._len) {
		ErrorExcept(
			"[copy_to_tensor()] src and dst length are different. %d != %d.",
			src._len, dst._len
		);
	}

	check_cuda(cudaMemcpy(dst._data, src._data, sizeof(_T) * src._len, cudaMemcpyDeviceToHost));
}

void set_uniform(NN_Tensor<nn_type>& p);