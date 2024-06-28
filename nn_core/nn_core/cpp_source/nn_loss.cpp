#include "nn_loss.h"


NN_Loss::~NN_Loss() {

}

void NN_Loss::run(
	const NN_List<GpuTensor<nn_type>>& output,
	const NN_List<GpuTensor<nn_type>>& y,
	NN_List<GpuTensor<nn_type>>& loss
) {

}