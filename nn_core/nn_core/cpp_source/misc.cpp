#include "misc.h"

/*******************************************

				tensor4dParam

*******************************************/

extern char str_buffer[STR_MAX];
extern int str_idx;

tensor4d::tensor4d() :
	_n(1),
	_c(1),
	_h(1),
	_w(1)
{
}

tensor4d::tensor4d(int n, int c, int h, int w) :
	_n(n),
	_c(c),
	_h(h),
	_w(w)
{
}

void tensor4d::set(int n, int c, int h, int w) {
	_n = n;
	_c = c;
	_h = h;
	_w = w;
}

const size_t tensor4d::get_size() const {
	return sizeof(nn_type) * _n * _c * _h * _w;
}

const bool tensor4d::is_valid() const {
	return (_n > 0 && _c > 0 && _h > 0 && _w > 0);
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

/**********************************************

				  solutionBase

**********************************************/

solutionBase::solutionBase() :
	_is_calculated(false)
{
}

const tensor4d solutionBase::calculate_size() {
	return tensor4d();
}

const size_t solutionBase::get_workspace_size() {
	return 0;
}

void solutionBase::clear_flag() {
	_is_calculated = false;
}