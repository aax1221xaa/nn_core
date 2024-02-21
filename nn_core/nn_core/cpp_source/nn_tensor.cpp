#include "nn_tensor.h"
#include <random>


/**********************************************/
/*                                            */
/*                 TensorBase                 */
/*                                            */
/**********************************************/

TensorBase::TensorBase() :
	_is_valid(false),
	_len(0)
{
}

TensorBase::TensorBase(const nn_shape& shape) :
	_len(0)
{
	_is_valid = check_shape(shape);

	if (_is_valid) {
		_shape = shape;
		_len = get_len(shape);
	}
}

size_t TensorBase::get_len(const nn_shape& shape) {
	size_t len = 1;

	for (const nn_shape& n : shape) {
		if (n._val > 0) len *= n._val;
	}

	return len;
}

bool TensorBase::check_shape(const nn_shape& shape) {
	bool is_valid = true;

	if (shape.size() > 0) {
		for (const nn_shape& n : shape) {
			if (n._val < 1) is_valid = false;
		}
	}
	else is_valid = false;

	return is_valid;
}