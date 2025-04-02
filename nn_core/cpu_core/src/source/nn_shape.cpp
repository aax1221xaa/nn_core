#include "../header/nn_shape.h"
#include "../header/Exception.h"


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

NN_Shape::NN_Shape(const std::initializer_list<int>& list) :
	_dims(list.size())
{	
	int i = 0;
	for (const int& n : list) _dims[i++] = n;
}

NN_Shape::NN_Shape(const int* dims, int ranks) :
	_dims(ranks, 0)
{
	for (int i = 0; i < ranks; ++i) _dims[i] = dims[i];
}

NN_Shape::NN_Shape(const NN_Shape& p) :
	_dims(p._dims)
{
}

NN_Shape::NN_Shape(NN_Shape&& p) :
	_dims(std::move(p._dims))
{
}

NN_Shape& NN_Shape::operator=(const NN_Shape& p) {
	if (this == &p) return *this;

	_dims = p._dims;

	return *this;
}

NN_Shape& NN_Shape::operator=(NN_Shape&& p) {
	if (this == &p) return *this;

	_dims = std::move(p._dims);

	return *this;
}

int& NN_Shape::operator[](int index) {
	if (index >= (int)_dims.size() || index < 0) {
		ErrorExcept(
			"[NN_Shape::operator[]] Index is out of range."
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
	if (index >= (int)_dims.size() || index < 0) {
		ErrorExcept(
			"[NN_Shape::operator[]] Index is out of range."
		);
	}

	return _dims[index];
}

int NN_Shape::ranks() {
	return (int)_dims.size();
}

int NN_Shape::ranks() const {
	return (int)_dims.size();
}

size_t NN_Shape::total_size() {
	size_t size = 1;

	for (int& n : _dims) size *= n;

	return size;
}

size_t NN_Shape::total_size() const {
	size_t size = 1;

	for (const int& n : _dims) size *= n;

	return size;
}

std::ostream& NN_Shape::put_shape(std::ostream& os) {
	os << '[';

	for (const int& n : _dims) os << std::to_string(n) << ", ";

	os << ']';

	return os;
}

std::ostream& NN_Shape::put_shape(std::ostream& os) const {
	os << '[';

	for (const int& n : _dims) os << std::to_string(n) << ", ";

	os << ']';

	return os;
}

bool NN_Shape::is_empty() {
	return _dims.empty();
}

bool NN_Shape::is_empty() const {
	return _dims.empty();
}

void NN_Shape::clear() {
	_dims.clear();
}

NN_Shape::iter NN_Shape::begin() {
	return _dims.begin();
}

NN_Shape::iter NN_Shape::end() {
	return _dims.end();
}

NN_Shape::r_iter NN_Shape::rbegin() {
	return _dims.rbegin();
}

NN_Shape::r_iter NN_Shape::rend() {
	return _dims.rend();
}

NN_Shape::c_iter NN_Shape::begin() const {
	return _dims.cbegin();
}

NN_Shape::c_iter NN_Shape::end() const {
	return _dims.cend();
}

NN_Shape::cr_iter NN_Shape::rbegin() const {
	return _dims.crbegin();
}

NN_Shape::cr_iter NN_Shape::rend() const {
	return _dims.crend();
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

std::vector<int>& NN_Shape::get_dims() {
	return _dims;
}

const std::vector<int>& NN_Shape::get_dims() const {
	return _dims;
}

NN_Shape4D NN_Shape::get_4dims() {
	if (_dims.size() > 4) {
		ErrorExcept(
			"[NN_Shape::get_4dims] This shape is %ld dimensions.",
			_dims.size()
		);
	}

	NN_Shape4D shape;

	shape._n = _dims[0];
	shape._h = _dims[1];
	shape._w = _dims[2];
	shape._c = _dims[3];

	return shape;
}

NN_Shape2D NN_Shape::get_2dims() {
	if (_dims.size() > 2) {
		ErrorExcept(
			"[NN_Shape::get_2dims] This shape is %ld dimensions.",
			_dims.size()
		);
	}

	NN_Shape2D shape;

	shape._n = _dims[0];
	shape._c = _dims[1];

	return shape;
}

NN_Shape4D NN_Shape::get_4dims() const {
	if (_dims.size() > 4) {
		ErrorExcept(
			"[NN_Shape::get_4dims] This shape is %ld dimensions.",
			_dims.size()
		);
	}

	NN_Shape4D shape;

	shape._n = _dims[0];
	shape._h = _dims[1];
	shape._w = _dims[2];
	shape._c = _dims[3];

	return shape;
}

NN_Shape2D NN_Shape::get_2dims() const {
	if (_dims.size() > 2) {
		ErrorExcept(
			"[NN_Shape::get_2dims] This shape is %ld dimensions.",
			_dims.size()
		);
	}

	NN_Shape2D shape;

	shape._n = _dims[0];
	shape._c = _dims[1];

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