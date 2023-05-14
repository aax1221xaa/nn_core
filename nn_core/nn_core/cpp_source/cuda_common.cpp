#include "cuda_common.h"


char str_buffer[STR_MAX] = { '\0' };
int str_idx = 0;


const char* dimension_to_str(const nn_shape& shape) {
	char tmp_buff[128] = { '\0', };
	char elem[16] = { '\0', };

	sprintf_s(tmp_buff, "[");

	for (int i = 0; i < shape.size(); ++i) {
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

dim3 get_grid_size(const dim3 block, unsigned int x, unsigned int y, unsigned int z) {
	dim3 grid(
		(x + block.x - 1) / block.x,
		(y + block.y - 1) / block.y,
		(z + block.z - 1) / block.z
	);

	return grid;
}

ptrManager NN_Shared_Ptr::linker;

NN_Shared_Ptr::NN_Shared_Ptr() :
	id(NULL)
{
}