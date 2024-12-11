#pragma once
#include "nn_lambda.h"


/**********************************************/
/*                                            */
/*					 NN_Div                   */
/*                                            */
/**********************************************/

class NN_Div : public NN_Operator {
	nn_type _val;

public:
	NN_Div();

	void op(NN_Stream& st, const GpuTensor<nn_type>& a, const GpuTensor<nn_type>& b, GpuTensor<nn_type>& c);
	void op(NN_Stream& st, const GpuTensor<nn_type>& a, nn_type b, GpuTensor<nn_type>& c);
	void op(NN_Stream& st, nn_type a, const GpuTensor<nn_type>& b, GpuTensor<nn_type>& c);

	void set_const_value(nn_type val, int status);
	nn_type get_val();
};