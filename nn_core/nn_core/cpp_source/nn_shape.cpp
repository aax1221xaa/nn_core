#include "nn_shape.h"



NN_Shape::NN_Shape() {

}

NN_Shape::NN_Shape(int len) :
	NN_Shape()
{
	if (len < 1) {
		ErrorExcept(
			"[NN_Shape::NN_Shape] Len must be greater than 0. but %d",
			len
		);
	}

	_dims.resize(len, 0);
}

NN_Shape::NN_Shape(const std::initializer_list<int>& list) :
	_dims(list)
{
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

const int& NN_Shape::operator[](int index) const {
	if (index >= (int)_dims.size() || index < 0) {
		ErrorExcept(
			"[NN_Shape::operator[]] Index is out of range."
		);
	}

	return _dims[index];
}

const int NN_Shape::get_len() const {
	return (int)_dims.size();
}

size_t NN_Shape::total_size() const {
	size_t size = _dims.size() > 0 ? 1 : 0;

	for (size_t i = 0; i < _dims.size(); ++i) {
		if (_dims[i] < 1) size = 0;

		size *= (size_t)_dims[i];
	}

	return size;
}

std::ostream& NN_Shape::put_shape(std::ostream& os) const {
	os << '[';

	for (const int& n : _dims) os << std::to_string(n) << ", ";

	os << ']' << std::endl;

	return os;
}

std::vector<int>::iterator NN_Shape::begin() {
	return _dims.begin();
}

std::vector<int>::iterator NN_Shape::end() {
	return _dims.end();
}

std::vector<int>::const_iterator NN_Shape::begin() const {
	return _dims.cbegin();
}

std::vector<int>::const_iterator NN_Shape::end() const {
	return _dims.cend();
}

bool NN_Shape::is_empty() const {
	return _dims.empty();
}

void NN_Shape::clear() {
	_dims.clear();
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

const char* shape_to_str(const NN_Shape& shape) {
	std::string buff = "[";

	if (shape.is_empty()) {
		for (const int& n : shape) {
			buff += std::to_string(n) + ", ";
		}
	}

	buff += "]";

	return buff.c_str();
}

std::ostream& operator<<(std::ostream& os, const NN_Shape& shape) {
	return shape.put_shape(os);
}