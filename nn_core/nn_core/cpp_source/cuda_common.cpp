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

const char* put_shape(const nn_shape& tensor) {
	char tmp_buff[128] = { '[', '\0', };
	char tmp_dim[16] = { '\0', };

	for (nn_shape& n : tensor) {
		sprintf_s(tmp_dim, "%d, ", n.get_val());
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

ptrManager NN_Shared_Ptr::linker;

NN_Shared_Ptr::NN_Shared_Ptr() :
	id(NULL)
{
}

std::ostream& operator<<(std::ostream& os, List<int>& list) {
	list.put(os);

	return os;
}

/**********************************************/
/*                                            */
/*                   nn_shape                 */
/*                                            */
/**********************************************/
/*
NN_Shape::NN_Shape() :
	_shape(NULL)
{	
}

NN_Shape::NN_Shape(const std::initializer_list<int> shape) {
	_shape = new NN_Shape::Container;

	_shape->_ref = 1;
	_shape->_rank = (int)shape.size();
	_shape->_nums = new int[shape.size()];

	int i = 0;
	for (const int& n : shape) _shape->_nums[i++] = n;
}

NN_Shape::~NN_Shape() {
	if (_shape) {
		if (_shape->_ref > 1) --_shape->_ref;
		else {
			delete[] _shape->_nums;
			delete _shape;
		}
	}

	_shape = NULL;
}

void NN_Shape::push_front(int num) {
	if (!_shape) {
		_shape = new NN_Shape::Container;
		_shape->_nums = NULL;
		_shape->_rank = 0;
		_shape->_ref = 1;
	}

	int* tmp = new int[_shape->_rank + 1];

	tmp[0] = num;

	for (int i = 0; i < _shape->_rank; ++i)
		tmp[i + 1] = _shape->_nums[i];

	delete[] _shape->_nums;
	_shape->_nums = tmp;

	++_shape->_rank;
}

void NN_Shape::push_back(int num) {
	if (!_shape) {
		_shape = new NN_Shape::Container;
		_shape->_nums = NULL;
		_shape->_rank = 0;
		_shape->_ref = 1;
	}

	int* tmp = new int[_shape->_rank + 1];

	tmp[_shape->_rank] = num;

	for (int i = 0; i < _shape->_rank; ++i)
		tmp[i] = _shape->_nums[i];

	delete[] _shape->_nums;
	_shape->_nums = tmp;

	++_shape->_rank;
}

void NN_Shape::erase(int index) {
	if (!_shape || _shape->_rank <= index || index < 0) {
		ErrorExcept(
			"[NN_Shape::erase] invalid index(%d).",
			index
		);
	}

	if (_shape->_rank > 1) {
		int* tmp = new int[_shape->_rank - 1];

		for (int i = 0, j = 0; i < _shape->_rank; ++i) {
			if (i != index) tmp[j++] = _shape->_nums[i];
		}

		delete[] _shape->_nums;
		_shape->_nums = tmp;

		--_shape->_rank;
	}
	else {
		delete[] _shape->_nums;
		delete _shape;

		_shape = NULL;
	}
}

void NN_Shape::resize(int size) {
	if (!_shape) {
		_shape = new NN_Shape::Container;

		_shape->_rank = size;
		_shape->_ref = 1;
	}
}
*/