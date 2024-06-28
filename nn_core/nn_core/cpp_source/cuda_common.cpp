#include "cuda_common.h"
#include <random>



dim3 get_grid_size(const dim3 block, unsigned int x, unsigned int y, unsigned int z) {
	if (x < 0 || y < 0 || z < 0) ErrorExcept("[get_grid_size()] x, y, z can't less than 0. but x = %d, y = %d, z = %d.", x, y, z);

	dim3 grid(
		(x + block.x - 1) / block.x,
		(y + block.y - 1) / block.y,
		(z + block.z - 1) / block.z
	);

	return grid;
}

std::vector<int> random_choice(int min, int max, int amounts, bool replace) {
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<int> dist(min, max - 1);

	std::vector<int> indice(amounts);

	if (replace) {
		for (int i = 0; i < amounts; ++i) {
			indice[i] = dist(gen);
		}
	}
	else {
		std::vector<bool> mask(labs(max - min), false);

		for (int i = 0; i < amounts; ++i) {
			while (true) {
				int num = dist(gen);

				if (!mask[num - min]) {
					mask[num - min] = true;
					indice[i] = num;

					break;
				}
			}
		}
	}

	return indice;
}


/**********************************************/
/*                                            */
/*                  NN_Stream                 */
/*                                            */
/**********************************************/

void NN_Stream::destroy() {
	if (_ptr) {
		if (_ptr->_n_ref > 0) --_ptr->_n_ref;
		else {
			for (int i = 0; i < _ptr->_amounts; ++i)
				check_cuda(cudaStreamDestroy(_ptr->_st[i]));

			delete[] _ptr->_st;
			delete _ptr;
		}
	}

	_ptr = NULL;
}

NN_Stream::NN_Stream(int amounts) {
	_ptr = new Container;

	_ptr->_amounts = amounts;
	_ptr->_n_ref = 0;
	_ptr->_st = new cudaStream_t[amounts];

	for (int i = 0; i < amounts; ++i) _ptr->_st[i] = NULL;

	try {
		for (int i = 0; i < amounts; ++i) check_cuda(cudaStreamCreate(&(_ptr->_st[i])));
	}
	catch (const NN_Exception& e) {
		for (int i = 0; i < amounts; ++i) cudaStreamDestroy(_ptr->_st[i]);

		delete[] _ptr->_st;
		delete _ptr;

		NN_Check::set_flag(false);

		throw e;
	}
}

NN_Stream::NN_Stream(const NN_Stream& p) :
	_ptr(p._ptr)
{
	if (_ptr) ++_ptr->_n_ref;
}

NN_Stream::NN_Stream(NN_Stream&& p) :
	_ptr(p._ptr)
{
	p._ptr = NULL;
}

NN_Stream::~NN_Stream() {
	try {
		clear();
	}
	catch (const NN_Exception& e) {
		NN_Check::set_flag(false);

		e.put();
	}
}

NN_Stream& NN_Stream::operator=(const NN_Stream& p) {
	if (this == &p) return *this;

	_ptr = p._ptr;

	if (_ptr) ++_ptr->_n_ref;

	return *this;
}

NN_Stream& NN_Stream::operator=(NN_Stream&& p) {
	if (this == &p) return *this;

	_ptr = p._ptr;
	p._ptr = NULL;

	return *this;
}

cudaStream_t& NN_Stream::operator[](int index) {
	if (index < 0 || index >= _ptr->_amounts) {
		ErrorExcept(
			"[NN_Stream::operator[]] Index is out of range."
		);
	}

	return _ptr->_st[index];
}

void NN_Stream::clear() {
	if (_ptr) {
		if (_ptr->_n_ref > 0) --_ptr->_n_ref;
		else {
			for (int i = 0; i < _ptr->_amounts; ++i)
				check_cuda(cudaStreamDestroy(_ptr->_st[i]));
			delete[] _ptr->_st;
			delete _ptr;
		}
	}
	_ptr = NULL;
}

cudaStream_t* NN_Stream::get_stream() const {
	return _ptr->_st;
}
