#ifndef DENSE_CUH
#define DENSE_CUH

#include "../cpp_source/nn_tensor.h"



/**********************************************/
/*											  */
/*				  host function 			  */
/*										      */
/**********************************************/

void dens(
	const cudaStream_t st,
	const NN_Tensor& input,
	const NN_Tensor& weight,
	NN_Tensor& output
);


#endif