#pragma once
#include <tbb/tbb.h>
#include <opencv2/opencv.hpp>
#include <memory>
#include "Exception.h"
#include "nn_list.h"
#include "nn_shape.h"
#include "mat_dtype.h"




/**********************************************/
/*                                            */
/*                    Tensor                  */
/*                                            */
/**********************************************/

template <typename _T>
class GpuTensor;

typedef std::vector<uint> Vect1D;
typedef std::vector<std::vector<uint>> Vect2D;
typedef std::shared_ptr<uint[]> uint_sptr;

template <typename _T>
class Tensor {
	std::shared_ptr<_T[]> _data;
	uint_sptr _indice;
	Vect1D _steps;
	NN_Shape _shape;

	int* _cnt_rank;

	static void copy_unsorted_data(const _T* src, _T* dst, size_t size, cuint* src_indice);
	static void copy_sorted_data(const _T* src, _T* dst, size_t size, cuint* dst_indice);
	static void copy_unsorted_both_data(const _T* src, _T* dst, size_t size, cuint* src_indice, cuint* dst_indice);
	static Vect1D generate_steps(const NN_Shape& shape);
	static Vect2D generate_dims_indice(const NN_Shape& shape);
	static NN_Shape calculate_shape(const Vect2D& dims_indice);
	static void check_shape(const NN_Shape& shape);
	static void change_dim_indice(Vect1D& dim_indice, int begin, int end, int step);
	static void put_tensor(std::ostream& os, const Tensor& tensor, size_t offset, int rank);
	static uint_sptr generate_indice(const uint_sptr& prev_indice, const Vect1D& steps, const Vect2D& dims_indice);
	
	Tensor(const Tensor& p, int cnt_rank);

public:
	
	Tensor();
	Tensor(const NN_Shape& shape);
	Tensor(NN_Shape&& shape);
	Tensor(const int* p_dims, int n_dims);
	Tensor(const Tensor& p);
	Tensor(Tensor&& p);
	Tensor(const GpuTensor<_T>& p);
	Tensor(const cv::Mat& mat);
	~Tensor();

	void clear();
	cv::Mat get_mat() const;
	NN_Shape get_shape() const;
	_T* get_ptr() const;
	
	Tensor reshape(const NN_Shape& shape);
	Tensor reshape(NN_Shape&& shape);

	Tensor transpose(const Vect1D& orders) const;

	Tensor expand_dims(int axis = 0);
	Tensor squeeze(int axis = 0);

	static Tensor zeros(const NN_Shape& shape);
	static Tensor zeros(NN_Shape&& shape);

	template <typename _cT>
	Tensor<_cT> cast() const;

	const Tensor& operator=(const Tensor& p);
	const Tensor& operator=(Tensor&& p);
	const Tensor& operator=(const GpuTensor<_T>& p);
	const Tensor& operator=(const _T& scalar);
	const Tensor& operator=(_T&& scalar);
	const Tensor& operator=(const cv::Mat& mat);

	Tensor operator()(int begin, int end, int step = 1) const;
	Tensor operator()(int index) const;
	Tensor operator()(const std::vector<int>& indice) const;
	Tensor operator[](int index) const;

	std::ostream& put(std::ostream& os) const;
};

template <typename _T>
std::ostream& operator<<(std::ostream& os, const Tensor<_T>& tensor) {
	tensor.put(os);

	return os;
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
void Tensor<_T>::copy_unsorted_data(const _T* src, _T* dst, size_t size, cuint* src_indice) {
	tbb::parallel_for(tbb::blocked_range<size_t>(0, size), [&](const tbb::blocked_range<size_t>& q) {
		for (size_t i = q.begin(); i < q.end(); ++i) dst[i] = src[src_indice[i]];
	}, tbb::auto_partitioner());
}

template <typename _T>
void Tensor<_T>::copy_sorted_data(const _T* src, _T* dst, size_t size, cuint* dst_indice) {
	tbb::parallel_for(tbb::blocked_range<size_t>(0, size), [&](const tbb::blocked_range<size_t>& q) {
		for (size_t i = q.begin(); i < q.end(); ++i) dst[dst_indice[i]] = src[i];
	}, tbb::auto_partitioner());
}

template <typename _T>
void Tensor<_T>::copy_unsorted_both_data(const _T* src, _T* dst, size_t size, cuint* src_indice, cuint* dst_indice) {
	tbb::parallel_for(tbb::blocked_range<size_t>(0, size), [&](const tbb::blocked_range<size_t>& q) {
		for (size_t i = q.begin(); i < q.end(); ++i) dst[dst_indice[i]] = src[src_indice[i]];
	}, tbb::auto_partitioner());
}

template <typename _T>
Vect1D Tensor<_T>::generate_steps(const NN_Shape& shape) {
	Vect1D steps(shape.ranks(), 1);

	NN_Shape::ConstIterator shape_iter = shape.begin();
	Vect1D::iterator step_iter = steps.begin();

	for (; shape_iter != shape.end(); ++shape_iter, ++step_iter) {
		NN_Shape::ConstIterator m_shape_iter = shape_iter + 1;

		for (; m_shape_iter != shape.end(); ++m_shape_iter) *step_iter *= (uint)(*m_shape_iter);
	}

	return steps;
}

template <typename _T>
Vect2D Tensor<_T>::generate_dims_indice(const NN_Shape& shape) {
	Vect2D indice;

	for (const int& n : shape) {
		uint k = 0;
		Vect1D m_indice(n, 0);

		for (uint& m : m_indice) m = k++;

		indice.push_back(m_indice);
	}

	return indice;
}

template <typename _T>
NN_Shape Tensor<_T>::calculate_shape(const Vect2D& dims_indice) {
	NN_Shape shape((int)dims_indice.size());
	NN_Shape::Iterator iter = shape.begin();

	for (const Vect1D& indice : dims_indice) {
		*iter = (int)indice.size();
		++iter;
	}

	return shape;
}

template <typename _T>
void Tensor<_T>::check_shape(const NN_Shape& shape) {
	for (const int& n : shape) {
		if (n < 1) {
			ErrorExcept(
				"This shape is invalid. %s",
				shape_to_str(shape)
			);
		}
	}
}

template <typename _T>
void Tensor<_T>::change_dim_indice(Vect1D& dim_indice, int begin, int end, int step) {
	const size_t n = (size_t)((end - begin + step - 1) / step);
	Vect1D m_indice(n, 0);

	uint i = 0;
	for (uint& m : m_indice) m = dim_indice[begin + step * i++];

	dim_indice = m_indice;
}

template <typename _T>
void Tensor<_T>::put_tensor(std::ostream& os, const Tensor& tensor, size_t offset, int rank) {
	cuint step = tensor._steps[rank];
	
	if (tensor._indice != NULL) {
		if (rank < tensor._shape.ranks() - 1) {
			for (int i = 0; i < tensor._shape[rank]; ++i) {
				os << '[';
				put_tensor(os, tensor, offset + (step * i), rank + 1);
				os << ']' << std::endl;
			}
		}
		else {
			for (int i = 0; i < tensor._shape[rank]; ++i) {
				size_t index = tensor._indice[offset + (step * i)];
				os << tensor._data[index] << ", ";
			}
		}
	}
	else {
		if (rank < tensor._shape.ranks() - 1) {
			for (int i = 0; i < tensor._shape[rank]; ++i) {
				os << '[';
				put_tensor(os, tensor, offset + (step * i), rank + 1);
				os << ']' << std::endl;
			}
		}
		else {
			for (int i = 0; i < tensor._shape[rank]; ++i) {
				os << tensor._data[offset + (step * i)] << ", ";
			}
		}
	}
}

template <typename _T>
uint_sptr Tensor<_T>::generate_indice(const uint_sptr& prev_indice, const Vect1D& steps, const Vect2D& dims_indice) {
	cuint ranks = (uint)dims_indice.size();

	if (ranks == 0) return NULL;

	uint* shape = new uint[ranks];
	uint size = 1;
	uint i = 0;

	for (const Vect1D& curr_indice : dims_indice) {
		cuint dims = (uint)curr_indice.size();

		shape[i++] = dims;
		size *= dims;
	}

	uint* p_indice = new uint[size];

	if (prev_indice == NULL) {
		tbb::parallel_for(tbb::blocked_range<uint>(0, size), [&](const tbb::blocked_range<uint>& q) {
			for (uint i = q.begin(); i < q.end(); ++i) {
				uint n = ranks;
				uint index = 0;
				uint j = i;

				while (n-- > 0) {
					cuint k = shape[n];

					index += steps[n] * dims_indice[n][j % k];
					j /= k;
				}

				p_indice[i] = index;
			}
		}, tbb::auto_partitioner());
	}
	else {
		cuint* p_prev_indice = prev_indice.get();

		tbb::parallel_for(tbb::blocked_range<uint>(0, size), [&](const tbb::blocked_range<uint>& q) {
			for (uint i = q.begin(); i < q.end(); ++i) {
				uint n = ranks;
				uint index = 0;
				uint j = i;

				while (n-- > 0) {
					cuint k = shape[n];

					index += steps[n] * dims_indice[n][j % k];
					j /= k;
				}

				p_indice[i] = p_prev_indice[index];
			}
		}, tbb::auto_partitioner());
	}

	delete[] shape;

	return uint_sptr(p_indice);
}

template <typename _T>
Tensor<_T>::Tensor(const Tensor& p, int cnt_rank) :
	_cnt_rank(new int(cnt_rank)),
	_data(p._data),
	_steps(p._steps),
	_shape(p._shape)
{

}

template <typename _T>
Tensor<_T>::Tensor() :
	_cnt_rank(new int(0))
{

}

template <typename _T>
Tensor<_T>::Tensor(const NN_Shape& shape) :
	_cnt_rank(new int(0))
{
	try {
		check_shape(shape);
		_steps = generate_steps(shape);
		_shape = shape;
		_data = std::shared_ptr<_T[]>(new _T[shape.total_size()]);
	}
	catch (const NN_Exception& e) {
		e.put();
		clear();
		delete _cnt_rank;

		throw e;
	}
}

template <typename _T>
Tensor<_T>::Tensor(NN_Shape&& shape) :
	_cnt_rank(new int(0))
{
	try {
		check_shape(shape);
		_steps = generate_steps(shape);
		_shape = shape;
		_data = std::shared_ptr<_T[]>(new _T[shape.total_size()]);
	}
	catch (const NN_Exception& e) {
		e.put();
		clear();
		delete _cnt_rank;

		throw e;
	}
}

template <typename _T>
Tensor<_T>::Tensor(const int* p_dims, int n_dims) :
	_cnt_rank(new int(0))
{
	try {
		const NN_Shape shape(p_dims, n_dims);

		check_shape(shape);
		_steps = generate_steps(shape);
		_shape = shape;
		_data = std::shared_ptr<_T[]>(new _T[shape.total_size()]);
	}
	catch (const NN_Exception& e) {
		e.put();
		clear();
		delete _cnt_rank;

		throw e;
	}
}

template <typename _T>
Tensor<_T>::Tensor(const Tensor& p) :
	_cnt_rank(new int(0))
{
	try {
		if (p._data != NULL) {

			check_shape(p._shape);
			_steps = generate_steps(p._shape);
			_shape = p._shape;
			_data = std::shared_ptr<_T[]>(new _T[_shape.total_size()]);

			if (_indice != NULL)
				copy_unsorted_data(p._data.get(), _data.get(), _shape.total_size(), p._indice.get());
			else
				memcpy(_data.get(), p._data.get(), sizeof(_T) * _shape.total_size());
		}
	}
	catch (const NN_Exception& e) {
		e.put();
		clear();
		delete _cnt_rank;

		throw e;
	}
}

template <typename _T>
Tensor<_T>::Tensor(Tensor&& p) :
	_cnt_rank(p._cnt_rank),
	_data(p._data),
	_indice(p._indice),
	_steps(p._steps),
	_shape(p._shape)
{
	p._data = NULL;
	p._cnt_rank = NULL;
}

template <typename _T>
Tensor<_T>::Tensor(const GpuTensor<_T>& p) :
	_cnt_rank(new int(0))
{
	try {
		const NN_Shape shape = p.get_shape();

		check_shape(shape);
		_steps = generate_steps(shape);
		_shape = shape;
		_data = std::shared_ptr<_T[]>(new _T[shape.total_size()]);

		check_cuda(cudaMemcpy(_data.get(), p.get_ptr(), sizeof(_T) * shape.total_size(), cudaMemcpyDeviceToHost));
	}
	catch (const NN_Exception& e) {
		e.put();
		clear();
		delete _cnt_rank;

		throw e;
	}
}

template <typename _T>
Tensor<_T>::Tensor(const cv::Mat& mat) :
	_cnt_rank(new int(0))
{
	const int channels = 1 + (mat.type() >> CV_CN_SHIFT);
	const int width = mat.cols;
	const int height = mat.rows;

	try {
		const NN_Shape shape({ height, width, channels });

		if (mat.depth() != what_type(_T())) {
			ErrorExcept(
				"%s != %s",
				cv::typeToString(mat.depth()).c_str(),
				cv::typeToString(what_type(_T())).c_str()
			);
		}

		check_shape(shape);
		_steps = generate_steps(shape);
		_shape = shape;
		_data = std::shared_ptr<_T[]>(new _T[shape.total_size()]);

		memcpy(_data.get(), mat.ptr(), sizeof(_T) * shape.total_size());
	}
	catch (const NN_Exception& e) {
		e.put();
		clear();
		delete _cnt_rank;

		throw e;
	}
}

template <typename _T>
Tensor<_T>::~Tensor() {
	delete _cnt_rank;
}

template <typename _T>
void Tensor<_T>::clear() {
	*_cnt_rank = 0;
	_data = NULL;
	_indice = NULL;
	_steps.clear();
	_shape.clear();
}

template <typename _T>
cv::Mat Tensor<_T>::get_mat() const {
	const int flag = get_type(_T(), _shape[-1]);
	std::vector<int> dims = _shape.get_dims();

	dims.pop_back();

	*_cnt_rank = 0;

	cv::Mat mat(dims, flag);

	if (_indice == NULL) memcpy(mat.ptr(), _data.get(), sizeof(_T) * _shape.total_size());
	else {
		copy_unsorted_data(
			_data.get(),
			(_T*)mat.ptr(),
			_shape.total_size(),
			_indice.get()
		);
	}

	return mat;
}

template <typename _T>
NN_Shape Tensor<_T>::get_shape() const {
	*_cnt_rank = 0;

	return _shape;
}

template <typename _T>
_T* Tensor<_T>::get_ptr() const {
	*_cnt_rank = 0;

	return _data.get();
}

template <typename _T>
Tensor<_T> Tensor<_T>::reshape(const NN_Shape& shape) {
	const size_t old_size = _shape.total_size();
	const size_t new_size = shape.total_size();

	if (old_size != new_size) {
		ErrorExcept(
			"Different shapes. %s != %s",
			shape_to_str(shape),
			shape_to_str(_shape)
		);
	}

	Tensor tensor;

	tensor._data = _data;
	tensor._indice = _indice;
	tensor._steps = generate_steps(shape);
	tensor._shape = shape;

	*_cnt_rank = 0;

	return tensor;
}

template <typename _T>
Tensor<_T> Tensor<_T>::reshape(NN_Shape&& shape) {
	const size_t old_size = _shape.total_size();
	const size_t new_size = shape.total_size();

	if (old_size != new_size) {
		ErrorExcept(
			"Different shapes. %s != %s",
			shape_to_str(shape),
			shape_to_str(_shape)
		);
	}

	Tensor tensor;

	tensor._data = _data;
	tensor._indice = _indice;
	tensor._steps = generate_steps(shape);
	tensor._shape = shape;

	*_cnt_rank = 0;

	return tensor;
}

template <typename _T>
Tensor<_T> Tensor<_T>::transpose(const Vect1D& orders) const {
	Tensor tensor;
	Vect1D m_steps;
	NN_Shape m_shape;

	for (const int& i : orders) {
		m_steps.push_back(_steps[i]);
		m_shape.push_back(_shape[i]);
	}
	
	Vect2D m_dims_indice = generate_dims_indice(m_shape);

	tensor._data = _data;
	tensor._shape = m_shape;
	tensor._steps = generate_steps(m_shape);
	tensor._indice = generate_indice(tensor._indice, m_steps, m_dims_indice);

	*_cnt_rank = 0;

	return tensor;
}

template <typename _T>
Tensor<_T> Tensor<_T>::expand_dims(int axis) {
	const int ranks = _shape.ranks();
	
	axis = axis < 0 ? ranks + axis : axis;

	if (ranks == 0) {
		ErrorExcept(
			"This tensor is empty."
		);
	}
	else if (axis < 0 || axis >= ranks) {
		ErrorExcept(
			"axis is out of range. %d",
			axis
		);
	}

	Tensor tensor(*this, *_cnt_rank);

	tensor._steps.insert(tensor._steps.begin() + axis, 0);
	tensor._shape.insert(tensor._shape.begin() + axis, 1);

	*_cnt_rank = 0;

	return tensor;
}

template <typename _T>
Tensor<_T> Tensor<_T>::squeeze(int axis) {
	const int ranks = _shape.ranks();

	axis = axis < 0 ? ranks + axis : axis;

	if (ranks == 0) {
		ErrorExcept(
			"This tensor is empty."
		);
	}
	else if (axis < 0 || axis >= ranks) {
		ErrorExcept(
			"Axis is out if range. %d",
			axis
		);
	}
	else if(_shape[axis] != 1){
		ErrorExcept(
			"This axis dimension is %d",
			_shape[axis]
		);
	}

	Tensor tensor(*this, *_cnt_rank);

	if (ranks > 1) {
		tensor._steps.erase(tensor._steps.begin() + axis);
		tensor._shape.pop(axis);
	}

	*_cnt_rank = 0;

	return tensor;
}

template <typename _T>
Tensor<_T> Tensor<_T>::zeros(const NN_Shape& shape) {
	Tensor tensor(shape);
	
	memset(tensor._data.get(), 0, sizeof(_T) * shape.total_size());

	return tensor;
}

template <typename _T>
Tensor<_T> Tensor<_T>::zeros(NN_Shape&& shape) {
	Tensor tensor(shape);

	memset(tensor._data.get(), 0, sizeof(_T) * shape.total_size());

	return tensor;
}

template <typename _T>
template <typename _cT>
Tensor<_cT> Tensor<_T>::cast() const {
	if (_data == NULL) {
		ErrorExcept(
			"This tensor is empty."
		);
	}

	Tensor<_cT> tensor(_shape);
	const _T* p_src = _data.get();
	_cT* p_dst = tensor.get_ptr();
	uint* indice = _indice.get();

	if (_indice == NULL) {
		tbb::parallel_for(tbb::blocked_range<size_t>(0, _shape.total_size()), [&](const tbb::blocked_range<size_t>& q) {
			for (size_t i = q.begin(); i < q.end(); ++i) p_dst[i] = (_cT)(p_src[i]);
		}, tbb::auto_partitioner());
	}
	else {
		tbb::parallel_for(tbb::blocked_range<size_t>(0, _shape.total_size()), [&](const tbb::blocked_range<size_t>& q) {
			for (size_t i = q.begin(); i < q.end(); ++i) p_dst[i] = (_cT)(p_src[indice[i]]);
		}, tbb::auto_partitioner());
	}

	return tensor;
}

template <typename _T>
const Tensor<_T>& Tensor<_T>::operator=(const Tensor& p) {
	if (this == &p) return *this;

	if (_data == NULL) {
		if (p._data != NULL) {
			_steps = generate_steps(p._shape);
			_shape = p._shape;

			const size_t size = _shape.total_size();

			_data = std::shared_ptr<_T[]>(new _T[size]);

			if (p._indice == NULL) memcpy(_data.get(), p._data.get(), sizeof(_T) * size);
			else copy_unsorted_data(p._data.get(), _data.get(), size, p._indice.get());
		}
	}
	else {
		if (p._data != NULL) {
			if (p._shape != _shape) {
				ErrorExcept(
					"%s != %s",
					shape_to_str(p._shape),
					shape_to_str(_shape)
				);
			}

			const size_t size = p._shape.total_size();

			if (_indice != NULL) {
				if (p._indice != NULL) {
					copy_unsorted_both_data(
						p._data.get(),
						_data.get(),
						size,
						p._indice.get(),
						_indice.get()
					);
				}
				else {
					copy_sorted_data(
						p._data.get(),
						_data.get(),
						size,
						_indice.get()
					);
				}
			}
			else {
				if (p._indice != NULL) {
					copy_unsorted_data(
						p._data.get(),
						_data.get(),
						size,
						p._indice.get()
					);
				}
				else {
					memcpy(_data.get(), p._data.get(), sizeof(_T) * size);
				}
			}
		}
	}

	*_cnt_rank = 0;

	return *this;
}

template <typename _T>
const Tensor<_T>& Tensor<_T>::operator=(Tensor&& p) {
	if (_data == NULL) {
		_data = p._data;
		_steps = p._steps;
		_shape = p._shape;
		_indice = p._indice;

		p.clear();
	}
	else {
		if (p._data != NULL) {
			if (p._shape != _shape) {
				ErrorExcept(
					"%s != %s",
					shape_to_str(p._shape),
					shape_to_str(_shape)
				);
			}

			const size_t size = _shape.total_size();

			if (_indice != NULL) {
				if (p._indice != NULL) {
					copy_unsorted_both_data(
						p._data.get(),
						_data.get(),
						size,
						p._indice.get(),
						_indice.get()
					);
				}
				else {
					copy_sorted_data(
						p._data.get(),
						_data.get(),
						size,
						_indice.get()
					);
				}
			}
			else {
				if (p._indice != NULL) {
					copy_unsorted_data(
						p._data.get(),
						_data.get(),
						size,
						p._indice.get()
					);
				}
				else {
					memcpy(_data.get(), p._data.get(), sizeof(_T) * size);
				}
			}
		}
	}

	*_cnt_rank = 0;

	return *this;
}

template <typename _T>
const Tensor<_T>& Tensor<_T>::operator=(const GpuTensor<_T>& p) {
	const NN_Shape src_shape = p.get_shape();
	const size_t src_size = src_shape.total_size();

	if (src_size == 0) {
		ErrorExcept(
			"GpuTensor is empty."
		);
	}

	if (_data == NULL) {
		_steps = generate_steps(src_shape);
		_shape = src_shape;
		_data = std::shared_ptr<_T[]>(new _T[src_size]);

		check_cuda(cudaMemcpy(_data.get(), p.get_ptr(), sizeof(_T) * src_size, cudaMemcpyDeviceToHost));
	}
	else {
		if (src_shape != _shape) {
			ErrorExcept(
				"%s != %s",
				shape_to_str(src_shape),
				shape_to_str(_shape)
			);
		}

		if (_indice == NULL)
			check_cuda(cudaMemcpy(_data.get(), p.get_ptr(), sizeof(_T) * src_size, cudaMemcpyDeviceToHost));
		else {
			_T* p_src = new _T[src_size];

			check_cuda(cudaMemcpy(p_src, p.get_ptr(), sizeof(_T) * src_size, cudaMemcpyDeviceToHost));
			copy_sorted_data(p_src, _data.get(), src_size, _indice.get());

			delete[] p_src;
		}
	}

	*_cnt_rank = 0;

	return *this;
}

template <typename _T>
const Tensor<_T>& Tensor<_T>::operator=(const _T& scalar) {
	const size_t size = _shape.total_size();

	if (_data == NULL) {
		ErrorExcept(
			"This tensor is empty."
		);
	}

	_T* dst = _data.get();
	cuint* p_indice = _indice.get();

	if (_indice == NULL) {
		tbb::parallel_for(tbb::blocked_range<size_t>(0, size), [&](const tbb::blocked_range<size_t>& q) {
			for (size_t i = q.begin(); i < q.end(); ++i) dst[i] = scalar;
		}, tbb::auto_partitioner());
	}
	else {
		tbb::parallel_for(tbb::blocked_range<size_t>(0, size), [&](const tbb::blocked_range<size_t>& q) {
			for (size_t i = q.begin(); i < q.end(); ++i) dst[p_indice[i]] = scalar;
		}, tbb::auto_partitioner());
	}

	*_cnt_rank = 0;

	return *this;
}

template <typename _T>
const Tensor<_T>& Tensor<_T>::operator=(_T&& scalar) {
	const size_t size = _shape.total_size();

	if (_data == NULL) {
		ErrorExcept(
			"This tensor is empty."
		);
	}

	_T* dst = _data.get();
	cuint* p_indice = _indice.get();

	if (_indice == NULL) {
		tbb::parallel_for(tbb::blocked_range<size_t>(0, size), [&](const tbb::blocked_range<size_t>& q) {
			for (size_t i = q.begin(); i < q.end(); ++i) dst[i] = scalar;
		}, tbb::auto_partitioner());
	}
	else {
		tbb::parallel_for(tbb::blocked_range<size_t>(0, size), [&](const tbb::blocked_range<size_t>& q) {
			for (size_t i = q.begin(); i < q.end(); ++i) dst[p_indice[i]] = scalar;
		}, tbb::auto_partitioner());
	}

	*_cnt_rank = 0;

	return *this;
}

template <typename _T>
const Tensor<_T>& Tensor<_T>::operator=(const cv::Mat& mat) {
	const int channels = 1 + (mat.type() >> CV_CN_SHIFT);
	const NN_Shape shape({ mat.rows, mat.cols, channels });
	const int depth = mat.depth();
	
	check_shape(shape);

	if (depth != what_type(_T())) {
		ErrorExcept(
			"%s != %s",
			cv::typeToString(mat.depth()).c_str(),
			cv::typeToString(depth).c_str()
		);
	}

	if (_data == NULL) {
		_steps = generate_steps(shape);
		_shape = shape;
		_data = std::shared_ptr<_T[]>(new _T[shape.total_size()]);

		memcpy(_data.get(), mat.ptr(), sizeof(_T) * shape.total_size());
	}
	else {
		if (_shape != shape) {
			ErrorExcept(
				"%s != %s",
				shape_to_str(shape),
				shape_to_str(_shape)
			);
		}

		if (_indice == NULL) memcpy(_data.get(), mat.ptr(), sizeof(_T) * shape.total_size());
		else {
			copy_sorted_data(
				(_T*)mat.ptr(),
				_data.get(),
				shape.total_size(),
				_indice.get()
			);
		}
	}

	*_cnt_rank = 0;

	return *this;
}

template <typename _T>
Tensor<_T> Tensor<_T>::operator()(int begin, int end, int step) const {
	if (*_cnt_rank >= _shape.ranks()) {
		ErrorExcept(
			"%d rank is empty.",
			*_cnt_rank
		);
	}

	const int n = _shape[*_cnt_rank];

	begin = begin < 0 ? n + begin : begin;
	end = end < 0 ? n + end : end;

	if (begin < 0 || end < 0 || begin >= n || end > n) {
		ErrorExcept(
			"begin and end is out of range. begin: %d, end: %d, step: %d",
			begin, end, step
		);
	}

	Tensor tensor(*this, *_cnt_rank + 1);
	Vect2D dims_indice = generate_dims_indice(_shape);

	change_dim_indice(dims_indice[*_cnt_rank], begin, end, step);
	tensor._indice = generate_indice(_indice, _steps, dims_indice);
	tensor._shape[*_cnt_rank] = (int)dims_indice[*_cnt_rank].size();
	tensor._steps = generate_steps(tensor._shape);
	*_cnt_rank = 0;

	return tensor;
}

template <typename _T>
Tensor<_T> Tensor<_T>::operator()(int index) const {
	if (*_cnt_rank >= _shape.ranks()) {
		ErrorExcept(
			"%d rank is empty.",
			*_cnt_rank
		);
	}

	const int n = _shape[*_cnt_rank];

	index = index < 0 ? n + index : index;

	if (index < 0 || index >= n) {
		ErrorExcept(
			"index is out of range. 0 ~ %d",
			n
		);
	}

	Tensor tensor(*this, *_cnt_rank);
	Vect2D m_dims_indice = generate_dims_indice(_shape);
	cuint m_index = m_dims_indice[*_cnt_rank][index];

	m_dims_indice[*_cnt_rank].clear();
	m_dims_indice[*_cnt_rank].push_back(m_index);

	tensor._indice = generate_indice(_indice, _steps, m_dims_indice);

	if (tensor._shape.ranks() > 1) {
		tensor._shape.pop(*_cnt_rank);
		tensor._steps = generate_steps(tensor._shape);
	}
	else {
		tensor._shape[*_cnt_rank] = 1;
	}
	*_cnt_rank = 0;

	return tensor;
}

template <typename _T>
Tensor<_T> Tensor<_T>::operator()(const std::vector<int>& indice) const {
	if (*_cnt_rank >= _shape.ranks()) {
		ErrorExcept(
			"%d rank is empty.",
			*_cnt_rank
		);
	}

	Tensor tensor(*this, *_cnt_rank);

	const int n = _shape[*_cnt_rank];
	Vect1D m_dim_indice(indice.size(), 0);
	Vect2D m_dims_indice = generate_dims_indice(_shape);

	int i = 0;
	for (const int& m : indice) {
		int index = m < 0 ? n + m : m;

		if (index < 0 || index >= n) {
			ErrorExcept(
				"[Tensor<_T>::operator()] indice is out of range."
			);
		}

		m_dim_indice[i++] = (uint)m_dims_indice[*_cnt_rank][index];
	}

	m_dims_indice[*_cnt_rank] = m_dim_indice;
	
	tensor._shape = calculate_shape(m_dims_indice);
	tensor._steps = generate_steps(tensor._shape);
	tensor._indice = generate_indice(_indice, tensor._steps, m_dims_indice);
	*_cnt_rank = 0;

	return tensor;
}

template <typename _T>
Tensor<_T> Tensor<_T>::operator[](int index) const {
	if (*_cnt_rank >= _shape.ranks()) {
		ErrorExcept(
			"%d rank is empty.",
			*_cnt_rank
		);
	}

	const int n = _shape[*_cnt_rank];

	index = index < 0 ? n + index : index;

	if (index < 0 || index >= n) {
		ErrorExcept(
			"index is out of range. 0 ~ %d. %d",
			n, index
		);
	}

	Tensor tensor(*this, *_cnt_rank);

	Vect2D m_dims_indice = generate_dims_indice(_shape);
	cuint m_index = m_dims_indice[*_cnt_rank][index];

	m_dims_indice[*_cnt_rank].clear();
	m_dims_indice[*_cnt_rank].push_back(m_index);

	tensor._indice = generate_indice(_indice, _steps, m_dims_indice);
	
	if (tensor._shape.ranks() > 1) {
		tensor._shape.pop(*_cnt_rank);
		tensor._steps = generate_steps(tensor._shape);
	}
	else {
		tensor._shape[0] = 1;
	}

	*_cnt_rank = 0;

	return tensor;
}

template <typename _T>
std::ostream& Tensor<_T>::put(std::ostream& os) const {
	if (_shape.ranks() > 0) {
		os << '[';
		put_tensor(os, *this, 0, 0);
		os << ']' << std::endl;
		os << "shape: " << shape_to_str(_shape) << std::endl;
	}
	else os << "[]" << std::endl;

	*_cnt_rank = 0;

	return os;
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
		check_cuda(cudaMalloc((void**)&ptr, sizeof(_T) * _shape.total_size()));

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
		check_cuda(cudaMalloc((void**)&dst_ptr, sizeof(_T) * _shape.total_size()));
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
		check_cuda(cudaMalloc((void**)&dst_ptr, sizeof(_T) * _shape.total_size()));
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

		check_cuda(cudaMalloc((void**)&ptr, sizeof(_T) * size));
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