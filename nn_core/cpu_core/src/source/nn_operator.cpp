#include "nn_operator.h"


/**********************************************/
/*                                            */
/*					 NN_Div                   */
/*                                            */
/**********************************************/

NN_Div::NN_Div() :
	NN_Operator("div"),
	_val(0.f)
{

}

void NN_Div::run(NN_Stream& st, const NN_List<GpuTensor<nn_type>>& input, NN_List<GpuTensor<nn_type>>& output) {
	switch (_status)
	{
	case 1:
	{
		NN_Tensor<nn_type> a_input = input.val();
		output.val() = a_input / _val;
	}
	break;
	case 2:
	{
		NN_Tensor<nn_type> a_input = input.val();
		output.val() = a_input.inverse(_val);
	}
	break;
	case 3:
	{
		NN_Tensor<nn_type> a_input = input[0].val();
		NN_Tensor<nn_type> b_input = input[1].val();
		output.val() = a_input / b_input;
	}
	break;
	default:
		break;
	}
}

void NN_Div::set_const_value(nn_type val, int status) {
	_val = val;
	_status = status;
}

nn_type NN_Div::get_val() {
	return _val;
}