#include "nn_loss.cuh"

#ifndef __CUDACC__
#define __CUDACC__
#endif

#include <device_functions.h>
#include <device_launch_parameters.h>

/**********************************************/
/*											  */
/*				 kernel function			  */
/*										      */
/**********************************************/


/**********************************************/
/*											  */
/*				     NN_Loss			      */
/*										      */
/**********************************************/

NN_Loss::NN_Loss(const std::string& loss_name) :
	_loss_name(loss_name)
{
}

NN_Loss::~NN_Loss() {

}

void NN_Loss::run(
	NN_Stream& st,
	const NN_List<GpuTensor<nn_type>>& true_y,
	const NN_List<GpuTensor<nn_type>>& pred_y,
	NN_List<GpuTensor<nn_type>>& loss,
	NN_List<GpuTensor<nn_type>>& doutput
) {

}