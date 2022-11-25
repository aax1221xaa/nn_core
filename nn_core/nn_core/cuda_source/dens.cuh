#ifndef DENSE_CUH
#define DENSE_CUH

#include "../cpp_source/cuda_common.h"



/**********************************************/
/*											  */
/*				  host function 			  */
/*										      */
/**********************************************/

void dens(
	const cudaStream_t st,
	const Tensor& input,
	const Tensor& weight,
	Tensor& output
);


#endif