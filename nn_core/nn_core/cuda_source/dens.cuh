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
	const NN_Tensor4D input,
	const NN_Tensor4D weight,
	NN_Tensor4D output
);


#endif