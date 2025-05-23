#pragma once
#include "nn_lambda.h"


/**********************************************/
/*                                            */
/*					 OpInput                  */
/*                                            */
/**********************************************/

class OpInput : public NN_Operator {
public:
	OpInput();

	void run(NN_Stream& st, const NN_List<GpuTensor<nn_type>>& input, NN_List<GpuTensor<nn_type>>& output);
};


/**********************************************/
/*                                            */
/*					  OpDiv                   */
/*                                            */
/**********************************************/

class OpDiv : public NN_Operator {
	nn_type _val;

public:
	OpDiv();

	void run(NN_Stream& st, const NN_List<GpuTensor<nn_type>>& input, NN_List<GpuTensor<nn_type>>& output);
	void set_const_value(nn_type val, int status);
	nn_type get_val();
};