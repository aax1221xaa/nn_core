#pragma once
#include "../cpp_source/gpu_tensor_misc.h"


/**********************************************/
/*                                            */
/*                     Add                    */
/*                                            */
/**********************************************/

class Add : public OperatorBase {
public:
	Add();
	GpuTensor<nn_type> run(const GpuTensor<nn_type>& a, const GpuTensor<nn_type>& b);
	OperatorBase* create_forward();
};