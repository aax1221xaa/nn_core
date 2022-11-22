#ifndef DENSE_CUH
#define DENSE_CUH

#include "../cpp_source/cuda_common.h"


/**********************************************/
/*											  */
/*				 kernel function			  */
/*										      */
/**********************************************/

__global__ void __matmul(
	float* a,
	float* b,
	float* c,
	const uint m,
	const uint n,
	const uint k
);


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