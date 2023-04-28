#ifndef DENSE_CUH
#define DENSE_CUH

#include "../cpp_source/nn_tensor.h"



/**********************************************/
/*											  */
/*				  host function 			  */
/*										      */
/**********************************************/

void dens(
	cudaStream_t st,
	const CudaTensor input,
	const CudaTensor weight,
	CudaTensor output
);


#endif