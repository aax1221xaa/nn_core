#include "cuda_misc.h"


/*******************************************

				tensor4dParam

*******************************************/

char str_buffer[STR_MAX] = { '\0', };
int str_idx = 0;

tensor4d::tensor4d() :
	_n(0),
	_c(0),
	_h(0),
	_w(0),
	_is_valid(false)
{
}

tensor4d::tensor4d(int n, int c, int h, int w) {
	set(n, c, h, w);
}

void tensor4d::set(int n, int c, int h, int w) {
	if (n < 1 || c < 1 || h < 1 || w < 1) _is_valid = false;
	else {
		_n = n;
		_c = c;
		_h = h;
		_w = w;

		_is_valid = true;
	}
}

size_t tensor4d::get_size() {
	return sizeof(nn_type) * _n * _c * _h * _w;
}

const char* tensor4d::shape_to_str(const tensor4d& tensor) {
	char tmp_buff[128] = { '\0', };

	sprintf_s(
		tmp_buff,
		"[%d, %d, %d, %d]",
		tensor._n,
		tensor._c,
		tensor._h,
		tensor._w
	);

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