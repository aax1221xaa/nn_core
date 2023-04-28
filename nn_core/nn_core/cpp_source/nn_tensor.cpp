#include "nn_tensor.h"
#include <random>


#ifdef FIX_MODE

void set_uniform(NN_Tensor<nn_type>& p) {
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<float> dis(-0.1f, 0.1f);

	size_t size = p._len;

	float* tmp = new float[size];

	for (size_t i = 0; i < size; ++i) tmp[i] = dis(gen);
	check_cuda(cudaMemcpy(p._data, tmp, sizeof(float) * size, cudaMemcpyHostToDevice));

	delete[] tmp;
}

#endif

#ifndef FIX_MODE
/**********************************************/
/*                                            */
/*                 TensorBase                 */
/*                                            */
/**********************************************/

DType TensorBase::get_type(bool* data) {
	return DType::boolean;
}

DType TensorBase::get_type(char* data) {
	return DType::int8;
}

DType TensorBase::get_type(uchar* data) {
	return DType::uint8;
}

DType TensorBase::get_type(short* data) {
	return DType::int16;
}

DType TensorBase::get_type(ushort* data) {
	return DType::uint16;
}

DType TensorBase::get_type(int* data) {
	return DType::int32;
}

DType TensorBase::get_type(uint* data) {
	return DType::uint32;
}

DType TensorBase::get_type(float* data) {
	return DType::float32;
}

DType TensorBase::get_type(double* data) {
	return DType::float64;
}

size_t TensorBase::get_bytes(DType dtype) {
	size_t bytes = 0;

	switch (dtype)
	{
	case DType::boolean:
		bytes = sizeof(bool);
		break;
	case DType::int8:
		bytes = sizeof(char);
		break;
	case DType::uint8:
		bytes = sizeof(uchar);
		break;
	case DType::int16:
		bytes = sizeof(short);
		break;
	case DType::uint16:
		bytes = sizeof(ushort);
		break;
	case DType::int32:
		bytes = sizeof(int);
		break;
	case DType::uint32:
		bytes = sizeof(uint);
		break;
	case DType::float32:
		bytes = sizeof(float);
		break;
	case DType::float64:
		bytes = sizeof(double);
		break;
	default:
		break;
	}

	return bytes;
}

uint TensorBase::calc_len_size(const nn_shape& shape) {
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

std::vector<uint> TensorBase::calc_steps(const nn_shape& shape) {
	std::vector<uint> steps(shape.size(), 0);
	uint step = 1;

	for (size_t i = shape.size(); i > 0; --i) {
		steps[i - 1] = step;
		step *= (uint)shape[i - 1];
	}

	return steps;
}

TensorBase::TensorBase() :
	_dtype(DType::none),
	_elem_size(0),
	_len(0)
{
}

TensorBase::TensorBase(const TensorBase& p) :
	_dtype(p._dtype),
	_elem_size(p._elem_size),
	_len(p._len),
	_shape(p._shape),
	_steps(p._steps)
{
}

TensorBase::TensorBase(const nn_shape& shape, DType dtype) :
	_shape(shape),
	_dtype(dtype),
	_elem_size(get_bytes(dtype))
{
	try {
		_len = calc_len_size(shape);
		_steps = calc_steps(shape);
	}
	catch (const Exception& p) {
		_shape.clear();
		_steps.clear();
		_dtype = DType::none;
		_elem_size = 0;
		_len = 0;

		p.Put();
	}
}

TensorBase::~TensorBase() {

}

/**********************************************/
/*                                            */
/*                 CPU_Tensor                 */
/*                                            */
/**********************************************/

void CPU_Tensor::left_cast(CPU_Tensor& dst, const CPU_Tensor& src) {
	switch (dst._dtype)
	{
	case DType::boolean:
		right_cast<bool>(dst._data, src._data, dst._indice, src._indice, src._dtype, dst._len);
		break;
	case DType::int8:
		right_cast<char>(dst._data, src._data, dst._indice, src._indice, src._dtype, dst._len);
		break;
	case DType::uint8:
		right_cast<uchar>(dst._data, src._data, dst._indice, src._indice, src._dtype, dst._len);
		break;
	case DType::int16:
		right_cast<short>(dst._data, src._data, dst._indice, src._indice, src._dtype, dst._len);
		break;
	case DType::uint16:
		right_cast<ushort>(dst._data, src._data, dst._indice, src._indice, src._dtype, dst._len);
		break;
	case DType::int32:
		right_cast<int>(dst._data, src._data, dst._indice, src._indice, src._dtype, dst._len);
		break;
	case DType::uint32:
		right_cast<uint>(dst._data, src._data, dst._indice, src._indice, src._dtype, dst._len);
		break;
	case DType::float32:
		right_cast<float>(dst._data, src._data, dst._indice, src._indice, src._dtype, dst._len);
		break;
	case DType::float64:
		right_cast<double>(dst._data, src._data, dst._indice, src._indice, src._dtype, dst._len);
		break;
	default:
		break;
	}
}

template <typename LT>
void CPU_Tensor::right_cast(
	void* dst,
	void* src,
	const uint* dst_indice,
	const uint* src_indice,
	DType src_type,
	uint len
) {
	switch (src_type)
	{
	case DType::boolean:
		trans_cast<LT, bool>(dst, src, len);
		break;
	case DType::int8:
		trans_cast<LT, char>(dst, src, len);
		break;
	case DType::uint8:
		trans_cast<LT, uchar>(dst, src, len);
		break;
	case DType::int16:
		trans_cast<LT, short>(dst, src, len);
		break;
	case DType::uint16:
		trans_cast<LT, ushort>(dst, src, len);
		break;
	case DType::int32:
		trans_cast<LT, int>(dst, src, len);
		break;
	case DType::uint32:
		trans_cast<LT, uint>(dst, src, len);
		break;
	case DType::float32:
		trans_cast<LT, float>(dst, src, len);
		break;
	case DType::float64:
		trans_cast<LT, double>(dst, src, len);
		break;
	default:
		break;
	}
}

template <typename LT, typename RT>
void CPU_Tensor::trans_cast(void* dst, void* src, const uint* dst_indice, const uint* src_indice, uint len) {
	for (uint i = 0; i < len; ++i) ((LT*)dst)[dst_indice[i]] = (LT)((RT*)src)[src_indice[i]];
}

void CPU_Tensor::put_boolean(std::ostream& os, void* data) {
	os << *((bool*)data) << ", ";
}

void CPU_Tensor::put_int8(std::ostream& os, void* data) {
	os << *((char*)data) << ", ";
}

void CPU_Tensor::put_uint8(std::ostream& os, void* data) {
	os << *((uchar*)data) << ", ";
}

void CPU_Tensor::put_int16(std::ostream& os, void* data) {
	os << *((short*)data) << ", ";
}

void CPU_Tensor::put_uint16(std::ostream& os, void* data) {
	os << *((ushort*)data) << ", ";
}

void CPU_Tensor::put_int32(std::ostream& os, void* data) {
	os << *((int*)data) << ", ";
}

void CPU_Tensor::put_uint32(std::ostream& os, void* data) {
	os << *((uint*)data) << ", ";
}

void CPU_Tensor::put_float32(std::ostream& os, void* data) {
	os << *((float*)data) << ", ";
}

void CPU_Tensor::put_float64(std::ostream& os, void* data) {
	os << *((double*)data) << ", ";
}

uint* CPU_Tensor::calculate_indice(
	uint& len,
	std::vector<uint>& steps,
	nn_shape& shape,
	const nn_shape& dims,
	const nn_shape& begin,
	const nn_shape& end,
	bool keep_dims
) {
	if (dims.size() != begin.size() || begin.size() != end.size()) {
		ErrorExcept(
			"[CPU_Tensor::get_indices()] shape, begin and end are different size. shape: %s, begin: %s, end: %s.",
			dimension_to_str(dims), dimension_to_str(begin), dimension_to_str(end)
		);
	}

	shape.clear();
	steps.clear();

	std::vector<int> m_shape;
	std::vector<uint> m_steps;

	uint step = 1;
	size_t i = dims.size();

	while (--i) {
		int n_dim = dims[i];
		int n_begin = begin[i];
		int n_end = end[i];

		if (n_dim < 0) {
			ErrorExcept(
				"[CPU_Tensor::calculate_indice()] dimension could be grater than 0. %s.",
				dimension_to_str(dims)
			);
		}
		else if (n_begin > n_end) {
			ErrorExcept(
				"[CPU_Tensor::calculate_indice()] begin could not greater than end. begin: %s, dims: %s.",
				dimension_to_str(begin), dimension_to_str(end)
			);
		}
		else if (n_begin == n_end) {
			if (n_begin < 0) {
				m_shape.push_back(n_dim);
			}
			else if (n_begin > n_dim) {
				ErrorExcept(
					"[CPU_Tensor::calculate_indices()] begin could not greater than dim. begin: %s, dims: %s.",
					dimension_to_str(begin), dimension_to_str(dims)
				);
			}
			if (keep_dims) m_shape.push_back(1);

			step *= n_dim;
		}
		else {
			const int m_begin = n_begin < 0 ? n_dim - n_begin : n_begin;
			const int m_end = n_end < 0 ? n_dim - n_end : n_end;

			if (m_begin < 0 || m_end < 0 || m_begin > n_dim || m_end > n_dim) {
				ErrorExcept(
					"[CPU_Tensor::get_indices()] begin and end are overflowed. begin: %s, end: %s",
					dimension_to_str(begin), dimension_to_str(end)
				);
			}

			m_shape.push_back(m_end - m_begin);
			step *= m_end - m_begin;
		}
		m_steps.push_back(step);
	}
	
	for (std::vector<int>::reverse_iterator iter = m_shape.rbegin(); iter != m_shape.rend(); ++iter) {
		shape.push_back(*iter);
	}
	for (std::vector<uint>::reverse_iterator iter = m_steps.rbegin(); iter != m_steps.rend(); ++iter) {
		steps.push_back(*iter);
	}

	uint* indice = new uint[step];

	for (uint i = 0; i < step; ++i) {
		uint addr = 0;
		for (size_t j = 0; j < m_shape.size(); ++j) {
			int n_shape = m_shape[j];
		}
	}
}

CPU_Tensor::CPU_Tensor() :
	_data(NULL),
	_indice(NULL),
	_is_shared_indice(false)
{
	id = NULL;
}

CPU_Tensor::CPU_Tensor(const nn_shape& shape, DType dtype) :
	TensorBase(shape, dtype),
	_is_shared_indice(false)
{
	if (_len) {
		_data = malloc(_elem_size * _len);
		_indice = (uint*)malloc(sizeof(uint) * _len);

		for (uint i = 0; i < _len; ++i) _indice[i] = i;

		id = linker.Create();
	}
}

CPU_Tensor::CPU_Tensor(const CPU_Tensor& p) :
	TensorBase(p),
	_data(p._data),
	_indice(p._indice),
	_is_shared_indice(true)
{
	id = p.id;

	if (id) ++id->ref_cnt;
}

CPU_Tensor::CPU_Tensor(CPU_Tensor&& p) :
	TensorBase(p),
	_data(p._data),
	_indice(p._indice),
	_is_shared_indice(false)
{
	id = p.id;

	p._data = NULL;
	p.id = NULL;
}

CPU_Tensor::~CPU_Tensor() {
	clear();
}

const CPU_Tensor& CPU_Tensor::operator=(const CPU_Tensor&p) {
	if (this == &p) return *this;

	clear();

	_data = p._data;
	_indice = p._indice;
	_is_shared_indice = true;

	_dtype = p._dtype;
	_elem_size = p._elem_size;
	_len = p._len;
	_shape = p._shape;
	_steps = p._steps;

	id = p.id;

	if (id) ++id->ref_cnt;

	return *this;
}

const CPU_Tensor& CPU_Tensor::operator=(CPU_Tensor&& p) {
	clear();

	_data = p._data;
	_indice = p._indice;
	_is_shared_indice = false;

	_dtype = p._dtype;
	_elem_size = p._elem_size;
	_len = p._len;
	_shape = p._shape;
	_steps = p._steps;

	id = p.id;

	p._data = NULL;
	p.id = NULL;

	return *this;
}

CPU_Tensor CPU_Tensor::cast(DType dtype) {
	CPU_Tensor tensor(_shape, dtype);

	left_cast(tensor, *this);

	return tensor;
}

void CPU_Tensor::clear() {
	if (id) {
		if (id->ref_cnt > 1) --id->ref_cnt;
		else {
			if (!_is_shared_indice) free(_indice);

			free(_data);
			
			linker.Erase(id);
		}
	}

	_data = NULL;
	_indice = NULL;

	_dtype = DType::none;
	_elem_size = 0;
	_len = 0;
	_shape.clear();
	_steps.clear();
}

void CPU_Tensor::set(const nn_shape& shape, DType dtype) {
	clear();

	_dtype = dtype;
	_shape = shape;
	_elem_size = get_bytes(dtype);
	_len = calc_len_size(shape);

	_data = malloc(_elem_size * _len);

	id = linker.Create();
}

CPU_Tensor CPU_Tensor::slice(const nn_shape& begin, const nn_shape& end, bool keep_dims) {
	CPU_Tensor tensor(*this);

	if (_shape.size() != begin.size() || begin.size() != end.size()) {
		ErrorExcept(
			"[CPU_Tensor::slice()] _shape, start and end size are differents. _shape: %ld, start: %ld, end: %ld.",
			_shape.size(), begin.size(), end.size()
		);
	}

	tensor._shape.clear();
	tensor._steps.clear();

	for (int i = 0; i < _shape.size(); ++i) {
		const int& n_shape = _shape[i];
		const int& n_begin = begin[i];
		const int& n_end = end[i];
		const uint& n_step = _steps[i];

		if (n_begin > n_end) {
			ErrorExcept(
				"[CPU_Tensor::slice()] start could be less than end. start: %s, end: %s.",
				dimension_to_str(begin), dimension_to_str(end)
			);
		}
		else if (n_begin == n_end) {
			if (n_begin < 0) {
				tensor._shape.push_back(n_shape);
				tensor._steps.push_back(n_step);
				tensor._start.push_back(0);
			}
			else {
				if (n_begin > n_shape) {
					ErrorExcept(
						"[CPU_Tensor::slice()] dimensions are overflowd. start: %s, end %s.",
						dimension_to_str(begin), dimension_to_str(end)
					);
				}
				else if (keep_dims) {
					tensor._shape.push_back(1);
					tensor._steps.push_back(n_step);
					tensor._start.push_back((uint)n_begin);
				}
				else {
					tensor._offset += n_step * n_begin;
				}
			}
		}
		else {
			const int m_begin = n_begin < 0 ? n_shape - n_begin : n_begin;
			const int m_end = n_end < 0 ? n_shape - n_end : n_end;

			if (m_begin < 0 || m_end < 0 || m_begin > n_shape || m_end > n_shape) {
				ErrorExcept(
					"[CPU_Tensor::slice()] dimensions are overflowed. start: %s, end: %s",
					dimension_to_str(begin), dimension_to_str(end)
				);
			}
			else {
				tensor._shape.push_back(m_end - m_begin);
				tensor._steps.push_back(n_step);
				tensor._start.push_back((uint)m_begin);
			}
		}
	}

	tensor._len = calc_len_size(tensor._shape);

	return tensor;
}

void CPU_Tensor::put(std::ostream& os) const {
	std::vector<uint> indicator;
	uint step = 1;
	bool end_flag = false;

	void(*put_func)(std::ostream&, void*) = NULL;

	switch (_dtype)
	{
	case DType::none:
		return;
	case DType::boolean:
		put_func = put_boolean;
		break;
	case DType::int8:
		put_func = put_int8;
		break;
	case DType::uint8:
		put_func = put_uint8;
		break;
	case DType::int16:
		put_func = put_int16;
		break;
	case DType::uint16:
		put_func = put_uint16;
		break;
	case DType::int32:
		put_func = put_int32;
		break;
	case DType::uint32:
		put_func = put_uint32;
		break;
	case DType::float32:
		put_func = put_float32;
		break;
	case DType::float64:
		put_func = put_float64;
		break;
	default:
		break;
	}

	for (auto iter = _shape.rbegin(); iter != _shape.rend(); ++iter) {
		step *= *iter;
		indicator.push_back(step);
	}

	os << "Tensor: " << dimension_to_str(_shape) << std::endl << std::endl;

	for (uint i = 0; i < _len;) {
		for (const uint& n : indicator) {
			if (i % n == 0) os << '[';
		}

		(*put_func)(os, ((uchar*)_data + (_elem_size * i)));
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

CPU_Tensor CPU_Tensor::zeros(const nn_shape& shape, DType dtype) {
	CPU_Tensor tensor(shape, dtype);

	memset(tensor._data, 0, tensor._elem_size * tensor._len);

	return tensor;
}

CPU_Tensor CPU_Tensor::zeros_like(const CPU_Tensor tensor, DType dtype) {
	CPU_Tensor m_tensor(tensor._shape, dtype);

	memset(m_tensor._data, 0, m_tensor._elem_size * m_tensor._len);

	return m_tensor;
}

std::ostream& operator<<(std::ostream& os, const CPU_Tensor& tensor) {
	tensor.put(os);

	return os;
}

#endif