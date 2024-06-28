#pragma once
#include "nn_list.h"
#include "nn_tensor.h"


class NN_Loss {
public:
	virtual ~NN_Loss();

	virtual void run(
		const NN_List<GpuTensor<nn_type>>& output,
		const NN_List<GpuTensor<nn_type>>& y,
		NN_List<GpuTensor<nn_type>>& loss
	);
};