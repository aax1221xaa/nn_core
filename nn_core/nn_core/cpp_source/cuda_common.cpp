#include "cuda_common.h"


char str_buffer[STR_MAX] = { '\0', };
int str_idx = 0;

dim3 get_grid_size(const dim3 block, unsigned int x, unsigned int y, unsigned int z) {
	if (x < 0 || y < 0 || z < 0) ErrorExcept("[get_grid_size()] x, y, z can't less than 0. but x = %d, y = %d, z = %d.", x, y, z);

	dim3 grid(
		(x + block.x - 1) / block.x,
		(y + block.y - 1) / block.y,
		(z + block.z - 1) / block.z
	);

	return grid;
}


/**********************************************/
/*                                            */
/*                   NN_Shape                 */
/*                                            */
/**********************************************/

NN_Shape::Iterator::Iterator(int* p_dims, int index) :
	_p_dims(p_dims),
	_index(index)
{
}

NN_Shape::Iterator::Iterator(const typename NN_Shape::Iterator& p) :
	_p_dims(p._p_dims),
	_index(p._index)
{
}

void NN_Shape::Iterator::operator++() {
	++_index;
}

bool NN_Shape::Iterator::operator!=(const typename NN_Shape::Iterator& p) const {
	return _index != p._index;
}

int& NN_Shape::Iterator::operator*() const {
	return _p_dims[_index];
}

NN_Shape::NN_Shape() :
	_data(new _Container()),
	_len(0)
{
}

NN_Shape::NN_Shape(int len) :
	_data(new _Container()),
	_len(len)
{
	_data->_dims = new int[len];
	memset(_data->_dims, 0, sizeof(int) * _len);
}

NN_Shape::NN_Shape(const std::initializer_list<int>& dims) :
	_data(new _Container()),
	_len(0)
{
	_data->_dims = new int[dims.size()];

	for (const int& n : dims) _data->_dims[_len++] = n;
}

NN_Shape::NN_Shape(const NN_Shape& p) :
	_data(p._data),
	_len(p._len)
{
	if (_data) ++_data->_n_ref;
}

NN_Shape::NN_Shape(NN_Shape&& p) :
	_data(p._data),
	_len(p._len)
{
	p._data = NULL;
	p._len = 0;
}

NN_Shape::~NN_Shape() {
	clear();
}

NN_Shape& NN_Shape::operator=(const NN_Shape& p) {
	if (this == &p) return *this;

	clear();

	_data = p._data;
	_len = p._len;

	if (_data) ++_data->_n_ref;

	return *this;
}

NN_Shape& NN_Shape::operator=(NN_Shape&& p) {
	if (this == &p) return *this;

	_data = p._data;
	_len = p._len;

	p._data = NULL;
	p._len = 0;

	return *this;
}

int& NN_Shape::operator[](const int& index) const {
	if (index < 0 || index >= _len) {
		ErrorExcept(
			"[NN_Shape::operator[]] Invalid index."
		);
	}

	return _data->_dims[index];
}

void NN_Shape::clear() {
	if (_data) {
		if (_data->_n_ref > 0) --_data->_n_ref;
		else {
			delete[] _data->_dims;
			delete _data;
		}
	}

	_data = NULL;
	_len = 0;
}

void NN_Shape::set(const std::initializer_list<int>& dims) {
	clear();

	_data = new _Container();
	_data->_dims = new int[dims.size()];
	_len = 0;

	for (const int& n : dims) _data->_dims[_len++] = n;
}

void NN_Shape::resize(int len) {
	clear();

	_data = new _Container();
	_data->_dims = new int[len];
	_len = len;

	memset(_data->_dims, 0, sizeof(int) * len);
}

void NN_Shape::push_back(const int dim) {
	int* tmp = new int[_len + 1];

	memcpy_s(tmp, sizeof(int) * _len, _data->_dims, sizeof(int) * _len);
	tmp[_len] = dim;
	++_len;

	delete[] _data->_dims;
	_data->_dims = tmp;
}

void NN_Shape::push_front(const int dim) {
	int* tmp = new int[_len + 1];

	memcpy_s(&tmp[1], sizeof(int) * _len, _data->_dims, sizeof(int) * _len);
	tmp[0] = dim;
	++_len;

	delete[] _data->_dims;
	_data->_dims = tmp;
}

const int& NN_Shape::get_size() const {
	return _len;
}

int* NN_Shape::get_dims() const {
	if (_data) return _data->_dims;
	else return NULL;
}

void NN_Shape::copy_to(NN_Shape& shape) const {
	if (_len != shape.get_size()) shape = NN_Shape(_len);

	memcpy_s(shape.get_dims(), sizeof(int) * _len, _data->_dims, sizeof(int) * _len);
}

typename NN_Shape::Iterator NN_Shape::begin() const {
	return NN_Shape::Iterator(_data->_dims, 0);
}

typename NN_Shape::Iterator NN_Shape::end() const {
	return NN_Shape::Iterator(_data->_dims, _len);
}


const char* put_shape(const NN_Shape& tensor) {
	char tmp_buff[128] = { '[', '\0', };
	char tmp_dim[16] = { '\0', };

	for (const int& n : tensor) {
		sprintf_s(tmp_dim, "%d, ", n);
		strcat_s(tmp_buff, tmp_dim);
	}
	strcat_s(tmp_buff, "]");

	int str_size = (int)strlen(tmp_buff) + 1;
	int least = STR_MAX - str_idx;
	char* p_buff = NULL;

	if (least >= str_size) {
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

std::ostream& operator<<(std::ostream& os, List<int>& list) {
	list.put(os);

	return os;
}