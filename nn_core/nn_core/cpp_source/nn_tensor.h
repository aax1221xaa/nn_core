#pragma once
#include "cuda_common.h"


bool check_shape(const NN_Shape& shape);
size_t calculate_length(const NN_Shape& shape);

/**********************************************/
/*                                            */
/*                    Tensor                  */
/*                                            */
/**********************************************/

template <typename _T>
class Tensor {
private:
	class _Container {
	public:
		_T* _data;
		int _n_ref;

		_Container() : _data(NULL), _n_ref(0) {}
	};

	_Container* _tensor;
	size_t _len;
	NN_Shape _shape;

	static void put_tensor(std::ostream& os, const Tensor<_T>& tensor, int& current_rank, size_t& counter);

public:

	Tensor();
	Tensor(const NN_Shape& shape);
	Tensor(const Tensor& p);
	Tensor(Tensor&& p);
	~Tensor();

	Tensor& operator=(Tensor& p);
	Tensor& operator=(Tensor&& p);

	_T* get_data() const;
	const size_t& get_len() const;
	const NN_Shape& get_shape() const;

	void clear();
	void set(const NN_Shape& shape);
	void put(std::ostream& os) const;
};

template <typename _T>
void Tensor<_T>::put_tensor(std::ostream& os, const Tensor<_T>& tensor, int& current_rank, size_t& counter) {
	if (current_rank < tensor._shape.get_size() - 1) {
		for (int i = 0; i < tensor._shape[current_rank]; ++i) {
			++current_rank;
			os << '[' << std::endl;
			put_tensor(os, tensor, current_rank, counter);
			os << ']' << std::endl;
			--current_rank;
		}
	}
	else {
		os << '[';
		for (int i = 0; i < tensor._shape[current_rank]; ++i)
			os << tensor._tensor->_data[counter++] << ", ";
		os << ']';
	}
}

template <typename _T>
Tensor<_T>::Tensor() :
	_len(0),
	_tensor(new _Container)
{
}

template <typename _T>
Tensor<_T>::Tensor(const NN_Shape& shape) :
	_tensor(new _Container),
	_len(0)
{
	if (!check_shape(shape)) {
		std::cout << "[Tensor<_T>::Tensor] shape is invalid." << put_shape(shape) << std::endl;
	}
	else {
		_shape = shape;
		_len = calculate_length(shape);
		_tensor->_data = new _T[_len];
	}
}

template <typename _T>
Tensor<_T>::Tensor(const Tensor& p) :
	_tensor(p._tensor),
	_len(p._len),
	_shape(p._shape)
{
	if (_tensor) ++_tensor->_n_ref;
}

template <typename _T>
Tensor<_T>::Tensor(Tensor&& p) :
	_tensor(p._tensor),
	_len(p._len),
	_shape(p._shape)
{
	p._tensor = NULL;
	p._len = 0;
	p._shape.clear();
}

template <typename _T>
Tensor<_T>::~Tensor() {
	clear();
}

template <typename _T>
Tensor<_T>& Tensor<_T>::operator=(Tensor& p) {
	if (this == &p) return *this;

	clear();

	_tensor = p._tensor;
	_len = p._len;
	_shape = p._shape;

	if (_tensor) ++_tensor->_n_ref;

	return *this;
}

template <typename _T>
Tensor<_T>& Tensor<_T>::operator=(Tensor&& p) {
	clear();

	_tensor = p._tensor;
	_len = p._len;
	_shape = p._shape;

	p._tensor = NULL;
	p._len = 0;
	p._shape.clear();

	return *this;
}

template <typename _T>
_T* Tensor<_T>::get_data() const {
	if (_tensor) return _tensor->_data;
	else return NULL;
}

template <typename _T>
const size_t& Tensor<_T>::get_len() const {
	return _len;
}

template <typename _T>
const NN_Shape& Tensor<_T>::get_shape() const {
	return _shape;
}

template <typename _T>
void Tensor<_T>::put(std::ostream& os) const {
	int rank = 0;
	size_t counter = 0;
	
	os << "Dimenstion: " << put_shape(_shape) << std::endl << std::endl;

	if (_shape.get_size() > 0) put_tensor(os, *this, rank, counter);
	else os << "[]\n";
}

template <typename _T>
void Tensor<_T>::clear() {
	if (_tensor) {
		if (_tensor->_n_ref > 0) --_tensor->_n_ref;
		else {
			delete[] _tensor->_data;
			delete _tensor;
		}
	}

	_tensor = NULL;
	_len = 0;
	_shape.clear();
}

template <typename _T>
void Tensor<_T>::set(const NN_Shape& shape) {
	clear();

	if (!check_shape(shape)) {
		ErrorExcept(
			"[Tensor<_T>::set] shape is invalid. %s",
			put_shape(shape)
		);
	}

	_shape = shape;
	_len = calculate_length(shape);

	_tensor = new _Container;
	_tensor->_data = new _T[_len];
}

template <typename _T>
std::ostream& operator<<(std::ostream& os, const Tensor<_T>& tensor) {
	tensor.put(os);

	return os;
}


template <typename _T>
Tensor<_T> zeros(const NN_Shape& shape) {
	Tensor<_T> tensor(shape);

	memset(tensor.get_data(), 0, sizeof(_T) * tensor.get_len());

	return tensor;
}

template <typename _DT, typename _ST>
Tensor<_DT> zeros_like(const Tensor<_ST>& src) {
	Tensor<_DT> tensor(src.get_shape());

	memset(tensor.get_data(), 0, sizeof(_DT) * tensor.get_len());

	return tensor;
}


/**********************************************/
/*                                            */
/*                  GpuTensor                 */
/*                                            */
/**********************************************/

template <typename _T>
class GpuTensor {
private:
	class _Container {
	public:
		_T* _data;
		int _n_ref;
		
		_Container() : _data(NULL), _n_ref(0) {}
	};

	_Container* _tensor;
	size_t _len;
	NN_Shape _shape;

public:
	GpuTensor();
	GpuTensor(const NN_Shape& shape);
	GpuTensor(const GpuTensor& p);
	GpuTensor(GpuTensor&& p);
	~GpuTensor();

	GpuTensor& operator=(GpuTensor& p);
	GpuTensor& operator=(GpuTensor&& p);

	_T* get_data() const;
	const size_t& get_len() const;
	const NN_Shape& get_shape() const;

	void clear();
	void set(const NN_Shape& shape);
};

template <typename _T>
GpuTensor<_T>::GpuTensor() :
	_tensor(new _Container),
	_len(0)
{
}

template <typename _T>
GpuTensor<_T>::GpuTensor(const NN_Shape& shape) :
	_tensor(new _Container),
	_len(0)
{
	if (!check_shape(shape)) {
		std::cout << "[GpuTensor<_T>::GpuTensor] shape is invalid." << put_shape(shape) << std::endl;
	}
	else {
		_shape = shape;
		_len = calculate_length(shape);
		
		cudaMalloc(&(_tensor->_data), sizeof(_T) * _len);
	}
}

template <typename _T>
GpuTensor<_T>::GpuTensor(const GpuTensor& p) :
	_tensor(p._tensor),
	_len(p._len),
	_shape(p._shape)
{
	if (_tensor) ++_tensor->_n_ref;
}

template <typename _T>
GpuTensor<_T>::GpuTensor(GpuTensor&& p) :
	_tensor(p._tensor),
	_len(p._len),
	_shape(p._shape)
{
	p._tensor = NULL;
	p._len = 0;
	p._shape.clear();
}

template <typename _T>
GpuTensor<_T>::~GpuTensor() {
	clear();
}

template <typename _T>
GpuTensor<_T>& GpuTensor<_T>::operator=(GpuTensor& p) {
	if (this == &p) return *this;

	clear();

	_tensor = p._tensor;
	_len = p._len;
	_shape = p._shape;

	if (_tensor) ++_tensor->_n_ref;

	return *this;
}

template <typename _T>
GpuTensor<_T>& GpuTensor<_T>::operator=(GpuTensor&& p) {
	clear();

	_tensor = p._tensor;
	_len = p._len;
	_shape = p._shape;

	p._tensor = NULL;
	p._len = 0;
	p._shape.clear();

	return *this;
}

template <typename _T>
_T* GpuTensor<_T>::get_data() const {
	return _tensor->_data;
}

template <typename _T>
const size_t& GpuTensor<_T>::get_len() const {
	return _len;
}

template <typename _T>
const NN_Shape& GpuTensor<_T>::get_shape() const {
	return _shape;
}

template <typename _T>
void GpuTensor<_T>::clear() {
	if (_tensor) {
		if (_tensor->_n_ref > 0) --_tensor->_n_ref;
		else {
			cudaFree(_tensor->_data);
			delete _tensor;
		}
	}

	_tensor = NULL;
	_len = 0;
	_shape.clear();
}

template <typename _T>
void GpuTensor<_T>::set(const NN_Shape& shape) {
	clear();

	if (!check_shape(shape)) {
		ErrorExcept(
			"[Tensor<_T>::set] shape is invalid. %s",
			put_shape(shape)
		);
	}

	_shape = shape;
	_len = calculate_length(shape);

	_tensor = new _Container;
	cudaMalloc(&(_tensor->_data), sizeof(_T) * _len);
}

template <class _T>
GpuTensor<_T> gpu_zeros(const NN_Shape& shape) {
	GpuTensor<_T> tensor(shape);

	check_cuda(cudaMemset(tensor.get_data(), 0, sizeof(_T) * tensor.get_len()));

	return tensor;
}

template <typename _DT, typename _ST>
GpuTensor<_DT> gpu_zeros_like(const GpuTensor<_ST>& src) {
	GpuTensor<_DT> tensor(src.get_shape());

	check_cuda(cudaMemset(tensor.get_data(), 0, sizeof(_DT) * tensor.get_len()));

	return tensor;
}


/**********************************************/
/*                                            */
/*                 Host & Gpu                 */
/*                                            */
/**********************************************/

template <typename _T>
void host_to_host(const Tensor<_T>& src, Tensor<_T>& dst) {
	if (src.get_len() != dst.get_len()) {
		ErrorExcept(
			"[host_to_host] The size of src and dst are different. %ld != %ld",
			src.get_len(), dst.get_len()
		);
	}

	memcpy_s(dst.get_data(), sizeof(_T) * dst.get_len(), src.get_data(), sizeof(_T) * src.get_len());
}

template <typename _T>
void host_to_gpu(const Tensor<_T>& src, GpuTensor<_T>& dst) {
	if (src.get_len() != dst.get_len()) {
		ErrorExcept(
			"[host_to_gpu] The size of src and dst are different. %ld != %ld",
			src.get_len(), dst.get_len()
		);
	}

	check_cuda(cudaMemcpy(dst.get_data(), src.get_data(), sizeof(_T) * src.get_len(), cudaMemcpyHostToDevice));
}

template <typename _T>
void gpu_to_host(const GpuTensor<_T>& src, Tensor<_T>& dst) {
	if (src.get_len() != dst.get_len()) {
		ErrorExcept(
			"[gpu_to_host] The size of src and dst are different. %ld != %ld",
			src.get_len(), dst.get_len()
		);
	}

	check_cuda(cudaMemcpy(dst.get_data(), src.get_data(), sizeof(_T) * dst.get_len(), cudaMemcpyDeviceToHost));
}

template <typename _T>
void gpu_to_gpu(const GpuTensor<_T>& src, GpuTensor<_T>& dst) {
	if (src.get_len() != dst.get_len()) {
		ErrorExcept(
			"[gpu_to_gpu] The size of src and dst are different. %ld != %ld",
			src.get_len(), dst.get_len()
		);
	}

	check_cuda(cudaMemcpy(dst.get_data(), src.get_data(), sizeof(_T) * dst.get_len(), cudaMemcpyDeviceToDevice));
}


/**********************************************/
/*                                            */
/*					  Random                  */
/*                                            */
/**********************************************/

void set_random_uniform(Tensor<float>& tensor, float a, float b);
void set_random_uniform(GpuTensor<float>& tensor, float a, float b);