#pragma once
#include "../cpp_source/nn_list.h"
#include "../cpp_source/nn_tensor.h"


/**********************************************/
/*											  */
/*				     NN_Loss			      */
/*										      */
/**********************************************/

class NN_Loss {
public:
	const std::string _loss_name;

	NN_Loss(const std::string& loss_name);
	virtual ~NN_Loss();

	virtual void run(
		NN_Stream& st,
		const NN_List<GpuTensor<nn_type>>& true_y,
		const NN_List<GpuTensor<nn_type>>& pred_y,
		NN_List<GpuTensor<nn_type>>& loss,
		NN_List<GpuTensor<nn_type>>& doutput
	);
};