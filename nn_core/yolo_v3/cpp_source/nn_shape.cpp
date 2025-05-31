#include "nn_shape.h"
#include "Exception.h"


#define BUFF_LEN			100


char str_buff[BUFF_LEN];
size_t buff_begin = 0;


/**********************************************/
/*                                            */
/*                   NN_Shape				  */
/*                                            */
/**********************************************/

NN_Shape::NN_Shape() {

}

NN_Shape::NN_Shape(size_t len) {
	if (len < 1) {
		ErrorExcept(
			"[NN_Shape::NN_Shape] Len must be greater than 0. but %d",
			len
		);
	}

	_dims.resize(len, 0);
}

NN_Shape::NN_Shape(size_t len, int val) :
	_dims(len, val)
{

}

NN_Shape::NN_Shape(const int* p_dims, int n_dims) {
	for (int i = 0; i < n_dims; ++i) _dims.push_back(p_dims[i]);
}

NN_Shape::NN_Shape(const hsize_t* p_dims, int n_dims) {
	for (int i = 0; i < n_dims; ++i) _dims.push_back((int)p_dims[i]);
}

NN_Shape::NN_Shape(const std::initializer_list<int>& list) {	
	for (const int& n : list) _dims.push_back(n);
}

NN_Shape::NN_Shape(const NN_Shape& p) :
	_dims(p._dims)
{
}

NN_Shape::NN_Shape(NN_Shape&& p) :
	_dims(p._dims)
{
}

NN_Shape& NN_Shape::operator=(const NN_Shape& p) {
	if (this == &p) return *this;

	_dims = p._dims;

	return *this;
}

NN_Shape& NN_Shape::operator=(NN_Shape&& p) {
	if (this == &p) return *this;

	_dims = p._dims;

	return *this;
}

int& NN_Shape::operator[](int index) {
	const int dim_size = (int)_dims.size();

	index = index < 0 ? dim_size + index : index;

	if (index >= dim_size || index < 0) {
		ErrorExcept(
			"[NN_Shape::operator[]] Index is out of range. dim size = %ld, index = %d",
			dim_size,
			index
		);
	}

	return _dims[index];
}

bool NN_Shape::operator!=(const NN_Shape& shape) const {
	bool is_not_equal = false;

	if (_dims.size() != shape._dims.size()) is_not_equal = true;
	else {
		for (int i = 0; i < _dims.size(); ++i) {
			if (_dims[i] != shape._dims[i]) {
				is_not_equal = true;
				break;
			}
		}
	}

	return is_not_equal;
}

const int& NN_Shape::operator[](int index) const {
	index = index < 0 ? (int)_dims.size() + index : index;

	if (index >= (int)_dims.size() || index < 0) {
		ErrorExcept(
			"[NN_Shape::operator[]] Index is out of range. dim size = %ld, index = %d",
			_dims.size(),
			index
		);
	}

	return _dims[index];
}

int NN_Shape::ranks() const {
	return (int)_dims.size();
}

size_t NN_Shape::total_size() const {
	size_t size = 1;

	for (const uint& n : _dims) size *= n;

	return size;
}

std::ostream& NN_Shape::put_shape(std::ostream& os) const {
	os << '[';

	for (const int& n : _dims) os << std::to_string(n) << ", ";

	os << ']';

	return os;
}

bool NN_Shape::is_empty() const {
	return _dims.empty();
}

void NN_Shape::clear() {
	_dims.clear();
}

NN_Shape::Iterator NN_Shape::begin() {
	return _dims.begin();
}

NN_Shape::Iterator NN_Shape::end() {
	return _dims.end();
}

NN_Shape::ConstIterator NN_Shape::begin() const {
	return _dims.cbegin();
}

NN_Shape::ConstIterator NN_Shape::end() const {
	return _dims.cend();
}

void NN_Shape::push_front(int n) {
	_dims.insert(_dims.begin(), n);
}

void NN_Shape::push_front(const NN_Shape& p) {
	_dims.insert(_dims.begin(), p._dims.begin(), p._dims.end());
}

void NN_Shape::push_front(const std::initializer_list<int>& list) {
	_dims.insert(_dims.begin(), list);
}

void NN_Shape::push_back(int n) {
	_dims.push_back(n);
}

void NN_Shape::push_back(const NN_Shape& p) {
	_dims.insert(_dims.end(), p._dims.begin(), p._dims.end());
}

void NN_Shape::push_back(const std::initializer_list<int>& list) {
	_dims.insert(_dims.end(), list);
}

int NN_Shape::pop(int index) {
	int n = _dims[index];
	_dims.erase(_dims.cbegin() + index);

	return n;
}

int NN_Shape::pop(ConstIterator iter) {
	int n = *iter;
	_dims.erase(iter);

	return n;
}

void NN_Shape::insert(int index, int n) {
	_dims.insert(_dims.cbegin() + index, n);
}

void NN_Shape::insert(ConstIterator iter, int n) {
	_dims.insert(iter, n);
}

const std::vector<int>& NN_Shape::get_dims() const {
	return _dims;
}

const std::vector<uint> NN_Shape::get_udims() const {
	std::vector<uint> udims;

	for (const int& n : _dims) udims.push_back((cuint)n);

	return udims;
}

NN_Tensor4dShape NN_Shape::get_4d_shape() {
	if (_dims.size() > 4) {
		ErrorExcept(
			"[NN_Tensor4dShape::get_4d_shape()] Can't return to tensor4d shape. because ranks are %ld",
			_dims.size()
		);
	}
	
	NN_Tensor4dShape shape;

	shape._n = _dims[0];
	shape._h = _dims[1];
	shape._w = _dims[2];
	shape._c = _dims[3];

	return shape;
}

NN_Tensor4dShape NN_Shape::get_4d_shape() const {
	if (_dims.size() > 4) {
		ErrorExcept(
			"[NN_Tensor4dShape::get_4d_shape()] Can't return to tensor4d shape. because ranks are %ld",
			_dims.size()
		);
	}

	NN_Tensor4dShape shape;

	shape._n = _dims[0];
	shape._h = _dims[1];
	shape._w = _dims[2];
	shape._c = _dims[3];

	return shape;
}

NN_Filter4dShape NN_Shape::get_filter_shape() {
	if (_dims.size() > 4) {
		ErrorExcept(
			"[NN_Tensor4dShape::get_filter_shape()] Can't return to tensor4d shape. because ranks are %ld",
			_dims.size()
		);
	}

	NN_Filter4dShape shape;

	shape._h = _dims[0];
	shape._w = _dims[1];
	shape._in_c = _dims[2];
	shape._out_c = _dims[3];

	return shape;
}

NN_Filter4dShape NN_Shape::get_filter_shape() const {
	if (_dims.size() > 4) {
		ErrorExcept(
			"[NN_Tensor4dShape::get_filter_shape()] Can't return to tensor4d shape. because ranks are %ld",
			_dims.size()
		);
	}

	NN_Filter4dShape shape;

	shape._h = _dims[0];
	shape._w = _dims[1];
	shape._in_c = _dims[2];
	shape._out_c = _dims[3];

	return shape;
}

const char* shape_to_str(const NN_Shape& shape) {
	std::string buff = "[";

	if (!shape.is_empty()) {
		for (const int& n : shape) {
			buff += std::to_string(n) + ", ";
		}
	}

	buff += "]";

	char* str_out = NULL;
	size_t size = strlen(buff.c_str()) + 1;

	if (buff_begin + size < BUFF_LEN) {
		str_out = str_buff + buff_begin;
		strcpy_s(str_out, sizeof(char) * BUFF_LEN, buff.c_str());
		buff_begin += size;
	}
	else {
		str_out = str_buff;
		strcpy_s(str_buff, buff.c_str());
		buff_begin = size;
	}

	return str_out;
}

std::ostream& operator<<(std::ostream& os, const NN_Shape& shape) {
	return shape.put_shape(os);
}