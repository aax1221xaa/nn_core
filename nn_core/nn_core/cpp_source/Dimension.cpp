#include "Dimension.h"


char str_buffer[STR_MAX] = { '\0' };
int str_idx = 0;


NN_Shape::Iterator::Iterator(int* _shape, int _index) :
	p_shape(_shape),
	index(_index)
{
}

NN_Shape::Iterator::Iterator(const NN_Shape::Iterator& p) : 
	p_shape(p.p_shape),
	index(p.index)
{
}

const NN_Shape::Iterator& NN_Shape::Iterator::operator++() {
	++index;
	
	return *this;
}

const NN_Shape::Iterator& NN_Shape::Iterator::operator--() {
	--index;

	return *this;
}

bool NN_Shape::Iterator::operator!=(const NN_Shape::Iterator& p) const {
	return index != p.index;
}

bool NN_Shape::Iterator::operator==(const NN_Shape::Iterator& p) const {
	return index == p.index;
}

int& NN_Shape::Iterator::operator*() const {
	return p_shape[index];
}

NN_Shape::NN_Shape() {
	shape = NULL;
	len = 0;
}

NN_Shape::NN_Shape(const NN_Shape& p) {
	id = p.id;

	shape = p.shape;
	len = p.len;

	if (id) ++id->ref_cnt;
}

NN_Shape::NN_Shape(const initializer_list<int>& _shape) {
	len = _shape.size();

	shape = new int[len];

	int i = 0;
	for (const int& n : _shape) shape[i++] = n;

	id = linker.Create();
}

NN_Shape::~NN_Shape() {
	clear();
}

void NN_Shape::set(const initializer_list<int>& _shape) {
	clear();

	len = _shape.size();

	shape = new int[len];

	int i = 0;
	for (const int& n : _shape) shape[i++] = n;

	id = linker.Create();
}

void NN_Shape::clear() {
	if (id) {
		if (id->ref_cnt > 1) --id->ref_cnt;
		else {
			delete[] shape;
			linker.Erase(id);
		}
	}

	id = NULL;
	shape = NULL;
	len = 0;
}
int& NN_Shape::operator[](int axis) const {
	int index = 0;

	if (axis < 0) {
		index = len + axis;
	}
	else {
		index = axis;
	}

	if (index < 0 || index >= len) {
		ErrorExcept(
			"[NN_Shape::operator[]] axis is out of range."
		);
	}

	return shape[index];
}

const NN_Shape::Iterator NN_Shape::begin() const {
	return NN_Shape::Iterator(shape, 0);
}
const NN_Shape::Iterator NN_Shape::end() const {
	return NN_Shape::Iterator(shape, len);
}

bool NN_Shape::operator==(const NN_Shape& p) {
	bool equal_flag = true;

	if (len != p.len) {
		ErrorExcept(
			"[NN_Shape::operator==] different ranks. %d, %d.",
			len, p.len
		);
	}

	for (int i = 0; i < len; ++i) {
		if (shape[i] != p.shape[i]) equal_flag = false;
	}

	return equal_flag;
}

const char* NN_Shape::get_str() const {
	char tmp_buff[128] = { '\0', };
	char elem[16] = { '\0', };

	sprintf_s(tmp_buff, "[");

	for (int i = 0; i < len; ++i) {
		sprintf_s(elem, "%d, ", shape[i]);
		strcat_s(tmp_buff, elem);
	}

	strcat_s(tmp_buff, "]");

	int str_size = strlen(tmp_buff) + 1;
	int clearance = STR_MAX - str_idx;
	char* p_buff = NULL;

	if (clearance >= str_size) {
		p_buff = &str_buffer[str_idx];
		str_idx += str_size;
	}
	else {
		p_buff = str_buffer;
		str_idx = 0;
	}

	strcpy_s(p_buff, sizeof(char) * str_size, tmp_buff);

	return p_buff;
}
const NN_Shape& NN_Shape::operator=(const NN_Shape& p) {
	if (this == &p) return *this;

	clear();

	id = p.id;
	shape = p.shape;
	len = p.len;

	if (id) ++id->ref_cnt;

	return *this;
}

const size_t NN_Shape::get_elem_size() const {
	size_t elem_size = 1;

	for (int i = 0; i < len; ++i) {
		int n = shape[i];

		if (n < 1) {
			elem_size = 0;
			break;
		}
		else elem_size *= n;
	}

	return elem_size;
}