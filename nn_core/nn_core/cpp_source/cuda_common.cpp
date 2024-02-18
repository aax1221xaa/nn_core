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

	for (cint n : tensor) {
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

ptrManager NN_Shared_Ptr::linker;

NN_Shared_Ptr::NN_Shared_Ptr() :
	id(NULL)
{
}